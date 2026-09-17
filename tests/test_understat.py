"""Tests for the Understat fetch/parse layer (soccer/clubs/data/understat.py).

The site is unreachable from the development sandbox, so the parser is
exercised on fixtures of every response shape it accepts: the JSON
endpoint with plain objects, the same with JSON-encoded string values,
and the legacy HTML page with embedded `JSON.parse` blobs. The fetch
path is tested with a stubbed session.
"""

import json

import pandas as pd
import pytest

from soccer.clubs.data import understat as u


def _match(mid, d, h, a, xg_h, xg_a, gh=1, ga=0, result=True):
    return {"id": str(mid), "isResult": result, "datetime": f"{d} 15:00:00",
            "h": {"id": "1", "title": h}, "a": {"id": "2", "title": a},
            "xG": {"h": str(xg_h), "a": str(xg_a)}, "goals": {"h": str(gh), "a": str(ga)}}


def _line(d, h_a, npxg, ppda_att, ppda_def, deep, xpts):
    return {"h_a": h_a, "xG": "1.5", "xGA": "0.8", "npxG": str(npxg), "npxGA": "0.7",
            "ppda": {"att": ppda_att, "def": ppda_def},
            "ppda_allowed": {"att": 200, "def": 20}, "deep": deep, "deep_allowed": 3,
            "scored": 1, "missed": 0, "xpts": str(xpts), "result": "w",
            "date": f"{d} 15:00:00", "wins": 1, "draws": 0, "loses": 0, "pts": 3,
            "npxGD": "0.5"}


def _payload_objects():
    return {
        "dates": [
            _match(1, "2024-08-17", "Arsenal", "Wolverhampton Wanderers", 2.1, 0.4),
            _match(2, "2024-08-24", "Aston Villa", "Arsenal", 0.9, 1.7, gh=0, ga=2),
            _match(3, "2025-05-25", "Arsenal", "Chelsea", 1.0, 1.0, result=False),
        ],
        "teams": {
            "83": {"id": "83", "title": "Arsenal", "history": [
                _line("2024-08-17", "h", 1.9, 250, 30, 12, 2.6),
                _line("2024-08-24", "a", 1.4, 180, 25, 8, 2.1),
            ]},
            "229": {"id": "229", "title": "Wolverhampton Wanderers", "history": [
                _line("2024-08-17", "a", 0.4, 120, 30, 2, 0.2),
            ]},
            "71": {"id": "71", "title": "Aston Villa", "history": [
                _line("2024-08-24", "h", 0.9, 140, 22, 5, 0.6),
            ]},
        },
    }


class TestParsePayload:
    def test_json_objects(self):
        matches, teams = u.parse_payload(_payload_objects())
        assert len(matches) == 3 and set(teams) == {"Arsenal", "Wolverhampton Wanderers", "Aston Villa"}

    def test_json_encoded_strings_and_legacy_key_names(self):
        p = _payload_objects()
        encoded = {"datesData": json.dumps(p["dates"]),
                   "teamsData": json.dumps(p["teams"])}
        matches, teams = u.parse_payload(encoded)
        assert len(matches) == 3 and len(teams["Arsenal"]) == 2

    def test_top_level_list_is_matches_only(self):
        payload = u._payload_from_response(json.dumps(_payload_objects()["dates"]))
        matches, teams = u.parse_payload(payload)
        assert len(matches) == 3 and teams == {}

    def test_legacy_html_blobs(self):
        p = _payload_objects()
        def blob(obj):
            raw = json.dumps(obj)
            return "".join(f"\\x{ord(c):02x}" if ord(c) < 128 else c for c in raw)
        html = ("<html><script>var datesData = JSON.parse('" + blob(p["dates"]) +
                "'); var teamsData = JSON.parse('" + blob(p["teams"]) + "');</script></html>")
        payload = u._payload_from_response(html)
        assert set(payload) == {"datesData", "teamsData"}
        matches, teams = u.parse_payload(payload)
        assert len(matches) == 3 and "Arsenal" in teams

    def test_garbage_is_none(self):
        assert u._payload_from_response("<html>nothing here</html>") is None


class TestBuildRows:
    def test_both_sides_joined_and_fixtures_skipped(self):
        matches, teams = u.parse_payload(_payload_objects())
        rows, unmatched = u.build_rows("epl", 2024, matches, teams)
        assert unmatched == set()
        assert len(rows) == 2                       # the unplayed fixture is skipped
        r = rows[0]
        assert r["season"] == "2024-25" and r["match_id"] == "1"
        assert r["home_team"] == "Arsenal FC" and r["away_team"] == "Wolverhampton Wanderers FC"
        assert r["xg_home"] == 2.1 and r["xg_away"] == 0.4
        assert r["npxg_home"] == 1.9 and r["npxg_away"] == 0.4
        assert r["ppda_att_home"] == 250 and r["ppda_def_home"] == 30
        assert r["deep_home"] == 12 and r["deep_away"] == 2
        assert r["xpts_home"] == 2.6 and r["xpts_away"] == 0.2
        # Arsenal away the following week: joined on (title, date, 'a').
        assert rows[1]["npxg_away"] == 1.4 and rows[1]["deep_away"] == 8

    def test_missing_team_history_leaves_advanced_fields_empty(self):
        matches, _ = u.parse_payload(_payload_objects())
        rows, _ = u.build_rows("epl", 2024, matches, {})
        assert rows[0]["xg_home"] == 2.1 and rows[0]["npxg_home"] is None

    def test_unmapped_name_is_refused_and_reported(self):
        p = _payload_objects()
        p["dates"].append(_match(9, "2024-09-01", "Arsenal", "Not A Club", 1, 1))
        matches, teams = u.parse_payload(p)
        rows, unmatched = u.build_rows("epl", 2024, matches, teams)
        assert unmatched == {"Not A Club"} and len(rows) == 2


class TestWrite:
    def test_processed_and_legacy_files_round_trip(self, tmp_path):
        matches, teams = u.parse_payload(_payload_objects())
        rows, _ = u.build_rows("epl", 2024, matches, teams)
        out = tmp_path / "understat_matches.csv"
        legacy = tmp_path / "xg_matches.csv"
        assert u.write_processed(rows, out) == 2
        assert u.write_legacy_xg(rows, legacy) == 2
        df = u.load_processed(out)
        assert list(df.columns) == u.COLUMNS and len(df) == 2
        assert df["npxg_home"].iloc[0] == pytest.approx(1.9)
        lg = pd.read_csv(legacy)
        assert list(lg.columns) == ["league", "date", "home_team", "away_team", "xg_home", "xg_away"]
        # A re-fetch replaces same-keyed rows rather than duplicating them.
        rows[0]["xg_home"] = 9.9
        assert u.write_processed(rows, out) == 2
        assert u.load_processed(out)["xg_home"].max() == pytest.approx(9.9)

    def test_load_processed_without_a_file_is_an_empty_typed_frame(self, tmp_path):
        df = u.load_processed(tmp_path / "nope.csv")
        assert list(df.columns) == u.COLUMNS and df.empty


class TestFetch:
    def test_json_endpoint_then_page_fallback_and_raw_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(u, "RAW_DIR", tmp_path)
        calls = []

        class Resp:
            def __init__(self, text): self.text = text; self.status_code = 200
            def raise_for_status(self): pass

        class Session:
            def get(self, url, timeout, headers):
                calls.append((url, headers.get("X-Requested-With")))
                if "getLeagueData" in url:
                    return Resp("<html>not json</html>")
                p = _payload_objects()
                def blob(obj):
                    return "".join(f"\\x{ord(c):02x}" for c in json.dumps(obj))
                return Resp("var datesData = JSON.parse('" + blob(p["dates"]) + "');")

        payload = u.fetch_league_season("epl", 2024, session=Session())
        assert "getLeagueData/EPL/2024" in calls[0][0] and calls[0][1] == "XMLHttpRequest"
        assert "league/EPL/2024" in calls[1][0]
        assert "datesData" in payload
        assert (tmp_path / "epl_2024.json").exists()

    def test_season_helpers(self):
        assert u.season_label(2014) == "2014-15"
        from datetime import date
        assert u.current_understat_season(date(2026, 9, 17)) == 2026
        assert u.current_understat_season(date(2026, 3, 1)) == 2025
