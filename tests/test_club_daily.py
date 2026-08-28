"""Tests for the club-soccer daily pipeline + UEFA glue + txt parser."""

import math

import pandas as pd
import pytest

from soccer.clubs.daily import grade, scoring, simulate
from soccer.clubs.data import football_txt
from soccer.clubs.model.elo import ClubEloEngine
from soccer.clubs.model.europe import cross_update


class TestFootballTxtParser:
    def test_group_restarts_do_not_advance_the_year(self):
        text = """= UEFA Champions League 2014/15

▪ Group A
  Tue Sep 16 2014
    20:45  Alpha FC (ESP)          v Beta FC (GER)            1-0 (1-0)
  Tue Dec 9
    20:45  Beta FC (GER)           v Alpha FC (ESP)           2-2

▪ Group B
  Tue Sep 16
    20:45  Gamma FC (ITA)          v Delta FC (ENG)           0-3 (0-1)
"""
        ms = football_txt.parse(text, "2014-15")
        assert [m.date for m in ms] == ["2014-09-16", "2014-12-09", "2014-09-16"]
        assert ms[2].country1 == "ITA"

    def test_january_dates_land_in_second_year(self):
        text = """▪ Matchday 20
  Sat Jan 10
    15:00  Alpha FC v Beta FC  2-1 (0-1)
"""
        (m,) = football_txt.parse(text, "2026-27")
        assert m.date == "2027-01-10"

    def test_pens_line_takes_the_extra_time_score(self):
        text = """▪ Final
  Wed May 31 2023
    21:00  Alpha FC (ESP)          v Beta FC (ITA)            4-1 pen. 1-1 a.e.t. (1-1, 0-1)
"""
        (m,) = football_txt.parse(text, "2022-23")
        assert (m.score1, m.score2) == (1, 1)

    def test_aet_without_pens_takes_first_score(self):
        text = """▪ Semifinals
  Thu May 18 2023
    21:00  Alpha FC (ESP)          v Beta FC (ITA)            2-1 a.e.t. (1-1, 0-0)
"""
        (m,) = football_txt.parse(text, "2022-23")
        assert (m.score1, m.score2) == (2, 1)

    def test_unplayed_fixture_has_no_score(self):
        text = """▪ Matchday 1
  Fri Aug 21 2026
    20:00  Arsenal FC              v Coventry City FC
"""
        (m,) = football_txt.parse(text, "2026-27")
        assert m.score1 is None
        assert m.team2 == "Coventry City FC"

    def test_goal_scorer_lines_are_skipped(self):
        text = """▪ Final
  Sat Jun 6 2015
    20:45  Alpha FC (ITA)          v Beta FC (ESP)            1-3 (0-1)
            (Morata 55'; Rakitic 4' Suárez 68' Neymar 90'+7')
"""
        ms = football_txt.parse(text, "2014-15")
        assert len(ms) == 1


class TestScoring:
    def test_grid_is_a_distribution(self):
        g = scoring.score_grid(1.4, 1.1)
        assert g.sum() == pytest.approx(1.0)
        assert g.shape == (scoring.MAX_GOALS + 1, scoring.MAX_GOALS + 1)

    def test_most_likely_score_tilts_with_lambdas(self):
        assert scoring.most_likely_score(2.6, 0.4) == (2, 0)
        assert scoring.most_likely_score(0.4, 2.6) == (0, 2)

    def test_lambdas_split_total_and_margin(self):
        p = scoring.ScoreParams(margin_a=-0.5, margin_b=1.6, league_total={"epl": 2.8})
        lam_h, lam_a = p.lambdas("epl", 0.75)
        assert lam_h + lam_a == pytest.approx(2.8)
        assert lam_h > lam_a


class TestUefaGlue:
    def _row(self, neutral=False):
        return pd.Series(
            {
                "date": "2025-09-16", "season": "2025-26", "competition": "ucl",
                "neutral": neutral, "home_team": "A", "home_league": "epl",
                "away_team": "B", "away_league": "serie_a",
                "home_score": 2, "away_score": 0,
            }
        )

    def test_cross_update_moves_points_between_pools(self):
        engines = {
            "epl": ClubEloEngine(k=10, home_advantage=50, entry_rating=1500),
            "serie_a": ClubEloEngine(k=20, home_advantage=40, entry_rating=1500),
        }
        rec = cross_update(engines, self._row(), weight=1.0)
        gained = engines["epl"].ratings["A"] - 1500
        lost = 1500 - engines["serie_a"].ratings["B"]
        assert gained == pytest.approx(lost)   # zero-sum across pools
        assert gained > 0
        assert rec["league"] == "uefa:ucl"
        # K is the mean of the two leagues' Ks (x1.5 for the 2-goal margin)
        assert gained == pytest.approx(15 * 1.5 * (1 - rec["exp_home"]))

    def test_neutral_final_drops_home_advantage(self):
        engines = {
            "epl": ClubEloEngine(k=10, home_advantage=80, entry_rating=1500),
            "serie_a": ClubEloEngine(k=10, home_advantage=80, entry_rating=1500),
        }
        rec = cross_update(engines, self._row(neutral=True), weight=1.0)
        assert rec["exp_home"] == pytest.approx(0.5)


def make_results(played):
    rows = []
    for date, league, home, away, hs, as_ in played:
        rows.append(
            {"date": date, "season": "2026-27", "league": league,
             "home_team": home, "away_team": away,
             "home_score": hs, "away_score": as_}
        )
    return pd.DataFrame(rows)


class TestGrading:
    @pytest.fixture(autouse=True)
    def _tmp_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(grade, "PREDICTIONS_DIR", tmp_path)
        monkeypatch.setattr(grade, "GRADES_CSV", tmp_path / "grades.csv")
        self.dir = tmp_path

    def _write_slate(self, name="slate_2026-08-22.csv"):
        slate = pd.DataFrame(
            [
                {"date": "2026-08-22", "league": "epl", "season": "2026-27",
                 "home_team": "A", "away_team": "B",
                 "p_H": 0.5, "p_D": 0.3, "p_A": 0.2, "pick": "H"},
                {"date": "2026-08-23", "league": "epl", "season": "2026-27",
                 "home_team": "C", "away_team": "D",
                 "p_H": 0.2, "p_D": 0.3, "p_A": 0.5, "pick": "A"},
            ]
        )
        slate.to_csv(self.dir / name, index=False)

    def test_grades_only_played_matches(self):
        self._write_slate()
        results = make_results([("2026-08-22", "epl", "A", "B", 3, 1)])
        graded = grade.grade_all(results, "2026-08-23")
        assert len(graded) == 1
        row = graded.iloc[0]
        assert row["outcome"] == "H" and bool(row["pick_correct"])
        assert row["log_loss"] == pytest.approx(-math.log(0.5), abs=1e-3)

    def test_idempotent_and_catches_late_results(self):
        self._write_slate()
        results = make_results([("2026-08-22", "epl", "A", "B", 0, 0)])
        first = grade.grade_all(results, "2026-08-23")
        assert len(first) == 1 and first.iloc[0]["outcome"] == "D"
        again = grade.grade_all(results, "2026-08-24")
        assert len(again) == 0            # never regraded
        late = make_results(
            [("2026-08-22", "epl", "A", "B", 0, 0),
             ("2026-08-23", "epl", "C", "D", 1, 2)]
        )
        third = grade.grade_all(late, "2026-09-10")
        assert len(third) == 1 and bool(third.iloc[0]["pick_correct"])
        summary = grade.ledger_summary()
        assert summary["graded"] == 2


class TestSimulateHelpers:
    def test_current_table_points(self):
        results = make_results(
            [("2026-08-22", "epl", "A", "B", 2, 0),
             ("2026-08-23", "epl", "C", "A", 1, 1)]
        )
        pts = simulate._current_table(results, "epl", "2026-27")
        assert pts == {"A": 4, "B": 0, "C": 1}


class TestRollingLedger:
    @pytest.fixture(autouse=True)
    def _tmp_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(grade, "PREDICTIONS_DIR", tmp_path)
        monkeypatch.setattr(grade, "GRADES_CSV", tmp_path / "grades.csv")
        self.dir = tmp_path

    def _seed_grades(self):
        rows = []
        for date, correct, ll in (
            ("2026-08-01", True, 0.5),   # outside the 7d window
            ("2026-08-25", True, 0.6),
            ("2026-08-27", False, 1.4),
        ):
            rows.append(
                {"date": date, "league": "epl", "season": "2026-27",
                 "home_team": "A", "away_team": "B", "pick": "H",
                 "p_H": 0.5, "p_D": 0.3, "p_A": 0.2, "outcome": "H",
                 "home_score": 1, "away_score": 0,
                 "pick_correct": correct, "log_loss": ll,
                 "graded_on": date}
            )
        pd.DataFrame(rows, columns=grade.LEDGER_COLUMNS).to_csv(
            self.dir / "grades.csv", index=False)

    def test_rolling_windows_split_by_match_date(self):
        self._seed_grades()
        summary = grade.ledger_summary("2026-08-28")
        assert summary["graded"] == 3
        assert summary["rolling"]["7d"]["graded"] == 2
        assert summary["rolling"]["7d"]["accuracy"] == pytest.approx(0.5)
        assert summary["rolling"]["30d"]["graded"] == 3

    def test_no_run_date_omits_rolling(self):
        self._seed_grades()
        assert "rolling" not in grade.ledger_summary()

    def test_recent_grades_window(self):
        self._seed_grades()
        recent = grade.recent_grades("2026-08-28", days=7)
        assert list(recent["date"]) == ["2026-08-25", "2026-08-27"]


class TestSimulateLeague:
    """simulate_league on a tiny synthetic 7-club league: shape and
    probability-mass identities of the Opta-style outputs."""

    def _state(self):
        from soccer.clubs.daily.scoring import ScoreParams
        from soccer.clubs.daily.state import DailyState

        clubs = list("ABCDEFG")
        engine = ClubEloEngine(k=10, home_advantage=60)
        engine.current_season = "2026-27"
        for i, c in enumerate(clubs):
            engine.ratings[c] = 1650 - 50 * i
            engine.last_league[c] = "epl"
            engine.last_season[c] = "2026-27"

        played = [("2026-08-22", "epl", "A", "B", 2, 0)]
        rows = make_results(played).to_dict(orient="records")
        for h, a in [(h, a) for h in clubs for a in clubs if h != a][:12]:
            rows.append(
                {"date": "2026-09-05", "season": "2026-27", "league": "epl",
                 "home_team": h, "away_team": a,
                 "home_score": None, "away_score": None}
            )
        results = pd.DataFrame(rows)
        params = ScoreParams(0.0, 0.8, {"epl": 2.6})
        return DailyState(
            engines={"epl": engine}, history=None, results=results,
            outcome_model=None, score_params=params, xg_form=None,
        )

    def test_outputs_carry_opta_fields_and_mass(self):
        sim = simulate.simulate_league(
            self._state(), "epl", "2026-27", n_sims=200, seed=1)
        clubs = sim["clubs"]
        assert len(clubs) == 7
        for key in ("exp_position", "p_uel", "p_title", "p_top4",
                    "p_relegation", "exp_points"):
            assert all(key in c for c in clubs)
        assert sum(c["p_title"] for c in clubs) == pytest.approx(1.0, abs=0.02)
        assert sum(c["p_top4"] for c in clubs) == pytest.approx(4.0, abs=0.05)
        assert sum(c["p_uel"] for c in clubs) == pytest.approx(2.0, abs=0.05)
        assert sum(c["p_relegation"] for c in clubs) == pytest.approx(3.0, abs=0.05)
        assert sum(c["exp_position"] for c in clubs) == pytest.approx(28.0, abs=0.1)
        # table ordered by expected finish, best first
        positions = [c["exp_position"] for c in clubs]
        assert positions == sorted(positions)
