"""Tests for the statsapi per-start ingest's default-window logic
(mlb/build_starts_statsapi.py), especially the fix for delayed/suspended
games that finalize after their date was first pulled."""

from __future__ import annotations

import csv
import json

import pytest

from mlb import build_starts_statsapi as m


# --- default_window: pure function, no network ------------------------

def test_no_prior_data_starts_at_initial_boundary():
    assert m.default_window(None, "2026-06-15") == m.INITIAL_START


def test_recent_last_statsapi_uses_rolling_overlap():
    # last_statsapi is within the OVERLAP_DAYS window of `end`: the rolling
    # tail should win, reaching back further than "day after last" so
    # already-covered dates get rescanned (catches delayed/suspended games).
    end = "2026-06-15"
    last = "2026-06-13"  # 2 days before end
    start = m.default_window(last, end)
    expected_overlap_start = "2026-06-09"  # end - (OVERLAP_DAYS - 1) = -6d
    assert start == expected_overlap_start
    assert start < last  # rescans dates already on file

    from datetime import date
    assert (date.fromisoformat(end) - date.fromisoformat(start)).days == (
        m.OVERLAP_DAYS - 1)


def test_stale_last_statsapi_catches_up_instead_of_skipping_gap():
    # Gap since last_statsapi exceeds OVERLAP_DAYS (job was down a while):
    # start the day after last_statsapi, not the trailing week, so the
    # missed days in between are not silently skipped.
    end = "2026-06-30"
    last = "2026-06-01"  # 29 days before end
    start = m.default_window(last, end)
    assert start == "2026-06-02"


def test_never_starts_before_initial_boundary():
    end = "2026-03-05"
    last = "2026-03-02"
    start = m.default_window(last, end)
    assert start >= m.INITIAL_START


# --- ingest(): full pipeline with mocked network, real parsing ---------

def _schedule_payload(date_str: str, games: list[dict]) -> dict:
    return {"dates": [{"date": date_str, "games": games}]} if games else {"dates": []}


def _game(game_pk: int, state: str = "F", game_number: int = 1) -> dict:
    return {
        "gamePk": game_pk,
        "gameNumber": game_number,
        "status": {"codedGameState": state},
        "teams": {
            "away": {"team": {"name": "Houston Astros"}},
            "home": {"team": {"name": "Texas Rangers"}},
        },
    }


def _boxscore_payload(pitcher_id: int, name: str) -> dict:
    line = {
        "person": {"fullName": name},
        "stats": {"pitching": {
            "gamesStarted": 1, "outs": 18, "hits": 5, "runs": 2,
            "earnedRuns": 2, "baseOnBalls": 1, "strikeOuts": 6,
        }},
    }
    return {"teams": {
        "away": {"pitchers": [pitcher_id], "players": {f"ID{pitcher_id}": line}},
        "home": {"pitchers": [pitcher_id + 1],
                 "players": {f"ID{pitcher_id + 1}": line}},
    }}


@pytest.fixture
def statsapi_env(tmp_path, monkeypatch):
    out = tmp_path / "pitcher_starts.csv"
    xwalk = tmp_path / "crosswalk.csv"
    checkpoint = tmp_path / "checkpoint.jsonl"
    with open(xwalk, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["retro_id", "mlbam_id", "bbref_id",
                                           "name_first", "name_last"])
        w.writeheader()

    monkeypatch.setattr(m, "OUT", out)
    monkeypatch.setattr(m, "XWALK", xwalk)
    monkeypatch.setattr(m, "CHECKPOINT", checkpoint)
    monkeypatch.setattr(m, "yesterday_et", lambda: "2026-06-15")
    monkeypatch.setattr(m.time, "sleep", lambda *_: None)
    return out


def test_delayed_final_is_picked_up_on_later_overlapping_run_no_dupes(
        statsapi_env, monkeypatch):
    out = statsapi_env

    # Day D0 (older, within the rolling window): game 100 was suspended when
    # first pulled, so it never made it into the CSV. Day D2 (more recent):
    # game 200 was already final and already ingested.
    D0, D2 = "2026-06-10", "2026-06-13"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        from mlb.build_starts import FIELDS
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for team, opp, home in (("TEX", "HOU", 1), ("HOU", "TEX", 0)):
            w.writerow({
                "game_id": "200", "source": "statsapi", "date": D2,
                "season": 2026, "game_num": 1, "team": team, "opponent": opp,
                "home": home, "retro_id": "", "mlbam_id": "9001",
                "name": "Existing Pitcher", "outs": 18, "h": 5, "r": 2,
                "er": 2, "bb": 1, "so": 6,
            })

    calls = {"schedule": 0}

    def fake_get(url: str):
        if "schedule" in url:
            calls["schedule"] += 1
            if calls["schedule"] == 1:
                # First run: D0's game is suspended (not final) -> absent
                # from final_games(); D2's game is final.
                return {"dates": [
                    _schedule_payload(D0, [_game(100, state="D")])["dates"][0],
                    _schedule_payload(D2, [_game(200, state="F")])["dates"][0],
                ]}
            # Second run: D0's game has since gone final.
            return {"dates": [
                _schedule_payload(D0, [_game(100, state="F")])["dates"][0],
                _schedule_payload(D2, [_game(200, state="F")])["dates"][0],
            ]}
        pk = int(url.rsplit("/", 2)[-2])
        return _boxscore_payload(pk, f"Pitcher {pk}")

    monkeypatch.setattr(m, "_get", fake_get)

    # Run 1: default window (no --start/--end) covers D0..yesterday; D0's
    # game is not yet final, so it is correctly skipped this run.
    assert m.ingest(None, None) == 0
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["game_id"] for r in rows} == {"200"}

    # Run 2 (later, still within the overlap window): D0's game is now
    # final. The rolling window re-scans D0 and picks it up.
    assert m.ingest(None, None) == 0
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    game_ids = [r["game_id"] for r in rows]
    assert set(game_ids) == {"100", "200"}
    # No duplication: game 200 (re-scanned both runs, already checkpointed)
    # still has exactly one row per side.
    assert game_ids.count("200") == 2
    assert game_ids.count("100") == 2


def test_explicit_start_end_bypass_default_window(statsapi_env, monkeypatch):
    """--start/--end must be honored as-is, ignoring last_statsapi/overlap."""
    out = statsapi_env
    calls = {"schedule": 0}

    def fake_get(url):
        if "schedule" in url:
            calls["schedule"] += 1
            return {"dates": [_schedule_payload(
                "2026-05-01", [_game(300, state="F")])["dates"][0]]}
        pk = int(url.rsplit("/", 2)[-2])
        return _boxscore_payload(pk, "Explicit Pitcher")

    monkeypatch.setattr(m, "_get", fake_get)
    assert m.ingest("2026-05-01", "2026-05-01") == 0
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["game_id"] for r in rows} == {"300"}
