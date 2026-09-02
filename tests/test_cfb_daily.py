"""Tests for the college-football daily pipeline: score model, slate,
idempotent grading with the paired always-home baseline, rolling ledger
windows, and the season Monte Carlo's probability-mass identities."""

import math

import pandas as pd
import pytest

from CFB.daily import grade, predict, scoring, simulate
from CFB.daily.config import ALWAYS_HOME_P, BOWL_ELIGIBLE_WINS
from CFB.daily.state import DailyState, as_of
from CFB.model.elo import CfbEloEngine

COLS = ["game_id", "season", "week", "season_type", "date", "start_utc",
        "home_team", "away_team", "home_conference", "away_conference",
        "home_division", "away_division", "home_points", "away_points",
        "neutral_site", "conference_game", "completed", "notes"]


def _row(gid, date, home, away, hp=None, ap=None, hconf="SEC", aconf="SEC",
         hdiv="fbs", adiv="fbs", neutral=False, notes="", week=1,
         season_type="regular"):
    completed = hp is not None
    return [gid, 2026, week, season_type, date, f"{date}T20:00:00Z", home, away,
            hconf, aconf, hdiv, adiv, hp, ap, neutral,
            hconf == aconf and hdiv == "fbs" and adiv == "fbs", completed, notes]


def _games(rows):
    df = pd.DataFrame(rows, columns=COLS)
    df["completed"] = df["completed"].astype(bool)
    return df


def _state(games, ratings, conferences):
    engine = CfbEloEngine(k=30.0, home_advantage=60.0, fcs_rating=1000.0,
                          entry_rating=1300.0)
    engine.roll_season(2026, conferences)
    engine.ratings.update(ratings)
    params = scoring.ScoreParams(margin_a=0.0, margin_b=0.05, league_total=52.0)
    return DailyState(engine=engine, history=None, games=games,
                      score_params=params, rates=scoring.TeamRates(), season=2026)


class TestScoring:
    def test_expected_score_splits_total_and_margin(self):
        p = scoring.ScoreParams(margin_a=0.0, margin_b=0.05, league_total=52.0)
        home, away = p.expected_score(200.0)          # +10 points
        assert home + away == pytest.approx(52.0, abs=0.11)
        assert home - away == pytest.approx(10.0, abs=0.11)
        assert p.elo_per_point == pytest.approx(20.0)

    def test_team_rates_shrink_toward_league_and_clip(self):
        r = scoring.TeamRates()
        assert r.attack("Nobody") == pytest.approx(1.0)
        for _ in range(30):
            r.observe("Hot", "Cold", 60, 3)
        assert r.attack("Hot") > 1.3 and r.defense("Hot") < 0.6
        assert scoring.TOTAL_MIN <= r.matchup_total("Hot", "Cold") <= scoring.TOTAL_MAX


class TestAsOf:
    def test_results_on_or_after_run_date_are_masked(self):
        g = _games([_row(1, "2026-09-05", "A", "B", 20, 10),
                    _row(2, "2026-09-06", "C", "D", 7, 3)])
        m = as_of(g, "2026-09-06")
        assert bool(m.loc[0, "completed"]) and not bool(m.loc[1, "completed"])
        assert pd.isna(m.loc[1, "home_points"])


class TestSlate:
    def test_slate_window_pick_and_fcs_flag(self):
        g = _games([
            _row(1, "2026-09-05", "A", "B"),
            _row(2, "2026-09-05", "C", "Podunk", adiv="fcs", aconf=None),
            _row(3, "2026-09-08", "A", "C"),          # outside the 2-day window
            _row(4, "2026-09-04", "B", "C", 21, 7),   # already played
        ])
        st = _state(g, {"A": 1500.0, "B": 1650.0, "C": 1500.0},
                    {"A": "SEC", "B": "SEC", "C": "SEC"})
        slate = predict.build_slate(st, "2026-09-05")
        assert list(slate["game_id"]) == [1, 2]
        ab = slate.iloc[0]
        assert ab["pick"] == "B" and ab["p_home"] < 0.5
        assert ab["pick_prob"] == pytest.approx(1 - ab["p_home"], abs=1e-4)
        fcs = slate.iloc[1]
        assert bool(fcs["away_fcs"]) and fcs["elo_away_pre"] == 1000.0
        assert fcs["pred_home_score"] > fcs["pred_away_score"]


class TestGrading:
    @pytest.fixture(autouse=True)
    def _tmp_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(grade, "PREDICTIONS_DIR", tmp_path)
        monkeypatch.setattr(grade, "GRADES_CSV", tmp_path / "grades.csv")
        self.dir = tmp_path

    def _write_slate(self, name="slate_2026-09-05.csv"):
        pd.DataFrame([
            {"game_id": 1, "date": "2026-09-05", "season": 2026, "week": 1,
             "home_team": "A", "away_team": "B", "neutral": False,
             "home_fcs": False, "away_fcs": False,
             "pick": "A", "p_home": 0.6, "pick_prob": 0.6,
             "pred_home_score": 30.0, "pred_away_score": 20.0},
            {"game_id": 2, "date": "2026-09-06", "season": 2026, "week": 1,
             "home_team": "C", "away_team": "D", "neutral": True,
             "home_fcs": False, "away_fcs": False,
             "pick": "D", "p_home": 0.3, "pick_prob": 0.7,
             "pred_home_score": 20.0, "pred_away_score": 28.0},
        ]).to_csv(self.dir / name, index=False)

    def test_grades_only_played_and_scores_paired_baseline(self):
        self._write_slate()
        games = _games([_row(1, "2026-09-05", "A", "B", 24, 21),
                        _row(2, "2026-09-06", "C", "D")])
        graded = grade.grade_all(games, "2026-09-06")
        assert len(graded) == 1
        r = graded.iloc[0]
        assert bool(r["pick_correct"]) and r["home_win"] == 1.0
        assert r["log_loss"] == pytest.approx(-math.log(0.6), abs=1e-3)
        assert r["home_log_loss"] == pytest.approx(-math.log(ALWAYS_HOME_P), abs=1e-3)
        assert r["d_ll"] == pytest.approx(r["log_loss"] - r["home_log_loss"], abs=1e-3)
        assert r["margin_err"] == pytest.approx(7.0)

    def test_neutral_baseline_is_coin_flip_and_grading_is_idempotent(self):
        self._write_slate()
        games = _games([_row(1, "2026-09-05", "A", "B", 24, 21),
                        _row(2, "2026-09-06", "C", "D", 10, 31, neutral=True)])
        first = grade.grade_all(games, "2026-09-07")
        assert len(first) == 2
        neutral = first[first["game_id"] == 2].iloc[0]
        assert neutral["home_log_loss"] == pytest.approx(-math.log(0.5), abs=1e-3)
        assert bool(neutral["pick_correct"])
        again = grade.grade_all(games, "2026-09-08")
        assert len(again) == 0
        # A second slate file carrying the same game (overlapping windows)
        # must not double count either.
        self._write_slate("slate_2026-09-06.csv")
        assert len(grade.grade_all(games, "2026-09-09")) == 0
        summary = grade.ledger_summary("2026-09-09")
        assert summary["graded"] == 2 and summary["correct"] == 2
        assert summary["rolling"]["7d"]["graded"] == 2
        assert summary["d_ll_se"] is not None

    def test_recent_grades_window(self):
        self._write_slate()
        games = _games([_row(1, "2026-09-05", "A", "B", 24, 21),
                        _row(2, "2026-09-06", "C", "D", 10, 31, neutral=True)])
        grade.grade_all(games, "2026-09-07")
        recent = grade.recent_grades("2026-09-12", days=7)
        assert list(recent["game_id"]) == [1, 2]
        assert grade.recent_grades("2026-09-20", days=7).empty


class TestSimulate:
    def _league(self):
        teams = list("ABCDEFGH")
        confs = {t: ("East" if t in "ABCD" else "West") for t in teams}
        rows = [_row(1, "2026-09-05", "A", "B", 28, 10, hconf="East", aconf="East")]
        gid = 2
        for h in teams:
            for a in teams:
                if h != a and (h, a) not in (("A", "B"), ("B", "A")) and gid <= 30:
                    rows.append(_row(gid, "2026-10-01", h, a,
                                     hconf=confs[h], aconf=confs[a]))
                    gid += 1
        rows.append(_row(99, "2026-11-20", "A", "Podunk", adiv="fcs", aconf=None,
                         hconf="East"))
        games = _games(rows)
        ratings = {t: 1650 - 40 * i for i, t in enumerate(teams)}
        return _state(games, ratings, confs)

    def test_records_and_mass_identities(self):
        st = self._league()
        rec = simulate.current_records(st.games, 2026, st.fbs_teams()).set_index("team")
        assert rec.loc["A", "wins"] == 1 and rec.loc["A", "conf_wins"] == 1
        assert rec.loc["B", "losses"] == 1 and rec.loc["B", "pts_diff"] == -18
        sim = simulate.simulate_season(st, n_sims=400, seed=3)
        teams = sim["teams"]
        assert len(teams) == 8
        by_conf = {}
        for t in teams:
            by_conf.setdefault(t["conference"], []).append(t)
        for conf, members in by_conf.items():
            assert sum(m["p_conf_title"] for m in members) == pytest.approx(1.0, abs=1e-6)
            assert sum(m["p_ccg"] for m in members) == pytest.approx(2.0, abs=1e-6)
        a = next(t for t in teams if t["team"] == "A")
        assert a["wins"] == 1 and a["exp_wins"] >= 1.0
        assert 0.0 <= a["p_bowl"] <= 1.0 and 0.0 <= a["p_undefeated"] <= 1.0
        # Strongest team should be the most likely East champion.
        east = sorted(by_conf["East"], key=lambda t: -t["p_conf_title"])
        assert east[0]["team"] == "A"

    def test_played_ccg_is_final(self):
        st = self._league()
        extra = _games([_row(200, "2026-12-05", "C", "A", 30, 27, hconf="East",
                             aconf="East", neutral=True, notes="East Championship",
                             week=14)])
        st.games = pd.concat([st.games, extra], ignore_index=True)
        sim = simulate.simulate_season(st, n_sims=50, seed=1)
        got = {t["team"]: t for t in sim["teams"]}
        assert got["C"]["p_conf_title"] == 1.0 and got["A"]["p_conf_title"] == 0.0
        assert got["A"]["p_ccg"] == 1.0 and got["B"]["p_ccg"] == 0.0
        assert BOWL_ELIGIBLE_WINS == 6
