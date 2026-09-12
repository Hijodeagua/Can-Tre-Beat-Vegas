"""Tests for the shot-form feature layer and its fetcher
(soccer/clubs/model/shots.py, soccer/clubs/data/fetch_shots.py)."""

import pandas as pd

from soccer.clubs.data import fetch_shots
from soccer.clubs.model.form import RollingForm, attach, replay
from soccer.clubs.model.shots import (
    MAX_AGE_DAYS,
    MIN_MATCHES,
    WINDOW,
    _Form,
    attach_shots,
    slate_diff,
)


def push_n(form, n, league="epl", home="A", away="B", start="2024-01-01",
           home_val=6.0, away_val=2.0):
    for i in range(n):
        d = (pd.Timestamp(start) + pd.Timedelta(days=7 * i)).date().isoformat()
        form.push(league, home, away, d, home_val, away_val)
    return d


class TestForm:
    def test_no_form_below_min_matches(self):
        f = _Form()
        push_n(f, MIN_MATCHES - 1)
        assert f.net("epl", "A", "2024-06-01") is None

    def test_net_is_rolling_mean_of_for_minus_against(self):
        f = _Form()
        last = push_n(f, MIN_MATCHES)   # A: 6 on target for, 2 against
        assert abs(f.net("epl", "A", last) - 4.0) < 1e-9
        assert abs(f.net("epl", "B", last) - (-4.0)) < 1e-9

    def test_window_caps_history(self):
        f = _Form()
        push_n(f, WINDOW, home_val=0.0, away_val=0.0)
        last = push_n(f, WINDOW, start="2025-01-01", home_val=9.0, away_val=0.0)
        assert abs(f.net("epl", "A", last) - 9.0) < 1e-9

    def test_staleness_guard_voids_old_form(self):
        f = _Form()
        last = push_n(f, MIN_MATCHES)
        stale = (pd.Timestamp(last) + pd.Timedelta(days=MAX_AGE_DAYS + 1)).date().isoformat()
        assert f.net("epl", "A", stale) is None

    def test_one_sided_form_is_zero(self):
        f = _Form()
        last = push_n(f, MIN_MATCHES, home="A", away="B")
        assert slate_diff(f, "epl", "A", "C", last) == 0.0

    def test_two_sided_form_diff(self):
        f = _Form()
        push_n(f, MIN_MATCHES, home="A", away="B", home_val=7.0, away_val=3.0)
        last = push_n(f, MIN_MATCHES, home="C", away="D", home_val=4.0, away_val=4.0)
        assert abs(slate_diff(f, "epl", "A", "C", last) - 4.0) < 1e-9


class TestAttachShots:
    def test_uncovered_league_rows_get_zero(self):
        hist = pd.DataFrame([
            {"date": "2024-01-01", "league": "serie_b",
             "home_team": "X", "away_team": "Y"},
        ])
        assert (attach_shots(hist)["sot_net_diff"] == 0.0).all()

    def test_feature_is_strictly_pre_match(self):
        # A club's very first covered match must feature 0 — no lookahead.
        hist = pd.DataFrame([
            {"date": "2010-08-14", "league": "epl",
             "home_team": "Aston Villa FC", "away_team": "West Ham United FC"},
        ])
        assert attach_shots(hist)["sot_net_diff"].iloc[0] == 0.0

    def test_attach_uses_only_earlier_matches(self):
        """Row 2 must see row 1's shots and nothing else."""
        rows = [
            {"date": f"2024-0{1 + i // 4}-{1 + 7 * (i % 4):02d}", "league": "epl",
             "home_team": "A", "away_team": "B"}
            for i in range(MIN_MATCHES + 1)
        ]
        hist = pd.DataFrame(rows)
        values = {("epl", r["date"], "A", "B"): (8.0, 1.0) for r in rows}
        out = attach(hist, "sot_net_diff", values,
                     lambda: RollingForm(WINDOW, MIN_MATCHES, MAX_AGE_DAYS))
        # the first MIN_MATCHES rows are warm-up; the last one sees form
        assert list(out["sot_net_diff"][:MIN_MATCHES]) == [0.0] * MIN_MATCHES
        assert abs(out["sot_net_diff"].iloc[-1] - 14.0) < 1e-9  # +7 vs -7


class TestCommittedData:
    def test_backfill_joins_onto_results(self):
        """Every committed shot row must key onto a played results.csv
        match — the join `attach` does is exact, so a drifted date or an
        unmapped name silently becomes a feature of 0."""
        shots = pd.read_csv(fetch_shots.OUT_CSV)
        res = pd.read_csv(fetch_shots.DATA_DIR / "results.csv")
        played = set(zip(res["league"], res["date"],
                         res["home_team"], res["away_team"]))
        keys = set(zip(shots["league"], shots["date"],
                       shots["home_team"], shots["away_team"]))
        assert keys <= played
        assert len(keys) == len(shots)      # no duplicate matches

    def test_counts_are_sane(self):
        shots = pd.read_csv(fetch_shots.OUT_CSV)
        assert (shots["sot_home"] <= shots["shots_home"]).all()
        assert (shots["sot_away"] <= shots["shots_away"]).all()
        assert (shots[["shots_home", "shots_away"]] >= 0).all().all()


class TestFetcher:
    def test_season_code(self):
        assert fetch_shots.season_code("2024-25") == "2425"
        assert fetch_shots.season_code("2009-10") == "0910"

    def test_iso_date_accepts_both_layers_spellings(self):
        assert fetch_shots._iso_date("2024-08-16") == "2024-08-16"
        assert fetch_shots._iso_date("16/08/2024") == "2024-08-16"
        assert fetch_shots._iso_date("16/08/24") == "2024-08-16"
        assert fetch_shots._iso_date("") is None

    def test_parse_skips_unmapped_names(self):
        text = ("Date,HomeTeam,AwayTeam,FTHG,FTAG,HS,AS,HST,AST\n"
                "2024-08-16,Man United,Fulham,1,0,14,10,5,2\n"
                "2024-08-16,Nowhere Town,Fulham,1,0,14,10,5,2\n")
        rows, unmatched = fetch_shots.parse_csv(text, "epl", "2024-25")
        assert len(rows) == 1
        assert rows[0]["home_team"] == "Manchester United FC"
        assert unmatched == {"Nowhere Town"}

    def test_parse_skips_rows_without_counts(self):
        text = ("Date,HomeTeam,AwayTeam,FTHG,FTAG,HS,AS,HST,AST\n"
                "2024-08-16,Man United,Fulham,,,,,,\n")
        rows, unmatched = fetch_shots.parse_csv(text, "epl", "2024-25")
        assert rows == [] and unmatched == set()

    def test_align_takes_the_date_from_results_and_checks_the_score(self):
        index = {("epl", "2024-25", "Manchester United FC", "Fulham FC"):
                 ("2024-08-16", 1, 0)}
        rows = [{"league": "epl", "season": "2024-25", "date": "2024-08-17",
                 "home_team": "Manchester United FC", "away_team": "Fulham FC",
                 "shots_home": 14, "shots_away": 10, "sot_home": 5,
                 "sot_away": 2, "_score": (1, 0)}]
        kept, unknown, mismatched = fetch_shots.align(list(rows), index)
        assert kept[0]["date"] == "2024-08-16"     # snapped off the day drift
        assert (unknown, mismatched) == (0, 0)

        wrong = [dict(rows[0], _score=(2, 0))]
        kept, unknown, mismatched = fetch_shots.align(wrong, index)
        assert kept == [] and mismatched == 1

    def test_merge_replaces_a_league_season_wholesale(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "shots.csv"
            def write(rows):
                import csv as _csv
                with out.open("w", newline="", encoding="utf-8") as f:
                    w = _csv.DictWriter(f, fieldnames=fetch_shots.COLUMNS)
                    w.writeheader()
                    w.writerows(rows)

            def row(league, season, date):
                return {"league": league, "season": season, "date": date,
                        "home_team": "A", "away_team": "B", "shots_home": 1,
                        "shots_away": 1, "sot_home": 1, "sot_away": 1}

            write([row("epl", "2024-25", "2024-08-16"),
                   row("epl", "2025-26", "2025-08-16")])
            merged = fetch_shots.merge([row("epl", "2025-26", "2025-08-17")], out)
            seasons = sorted((r["league"], r["season"], r["date"]) for r in merged)
            assert seasons == [("epl", "2024-25", "2024-08-16"),
                               ("epl", "2025-26", "2025-08-17")]


class TestReplay:
    def test_replay_feeds_in_date_order(self):
        values = {
            ("epl", "2024-03-01", "A", "B"): (9.0, 0.0),
            ("epl", "2024-01-01", "A", "B"): (0.0, 0.0),
        }
        form = replay(values, lambda: RollingForm(WINDOW, 1, MAX_AGE_DAYS))
        assert form.last_date[("epl", "A")] == "2024-03-01"
