"""Tests for the NFL daily pipeline: score model, the next-week slate,
idempotent grading with the paired always-home baseline (ties included),
rolling ledger windows, and the season Monte Carlo's probability-mass
identities through the playoff bracket."""

import math

import pandas as pd
import pytest

from NFL.daily import grade, predict, scoring, simulate
from NFL.daily.config import ALWAYS_HOME_P
from NFL.daily.state import DailyState, as_of
from NFL.elo.engine import NflEloEngine
from NFL.elo.teams import CONFERENCE_OF, DIVISION_OF, DIVISIONS, TEAMS

COLS = ["game_id", "season", "week", "game_type", "date", "gametime", "weekday",
        "home_team", "away_team", "home_score", "away_score", "location",
        "home_rest", "away_rest", "div_game", "neutral", "playoff", "completed"]


def _row(gid, date, home, away, hs=None, as_=None, week=1, gt="REG",
         neutral=False, h_rest=7, a_rest=7, season=2026):
    return [gid, season, week, gt, date, "13:00", "Sunday", home, away, hs, as_,
            "Neutral" if neutral else "Home", h_rest, a_rest,
            int(DIVISION_OF.get(home) == DIVISION_OF.get(away)), neutral,
            gt != "REG", hs is not None]


def _games(rows):
    df = pd.DataFrame(rows, columns=COLS)
    df["completed"] = df["completed"].astype(bool)
    return df


def _state(games, ratings=None):
    engine = NflEloEngine(k=20.0, home_advantage=50.0, rest_bonus=25.0)
    engine.roll_season(2026)
    engine.ratings.update(ratings or {})
    params = scoring.ScoreParams(margin_a=0.0, margin_b=0.04, league_total=46.0, mov_mean=2.0)
    return DailyState(engine=engine, history=None, games=games,
                      score_params=params, rates=scoring.TeamRates(), season=2026)


class TestScoring:
    def test_expected_score_splits_total_and_margin(self):
        p = scoring.ScoreParams(margin_a=0.0, margin_b=0.04, league_total=46.0)
        home, away = p.expected_score(100.0)          # +4 points
        assert home + away == pytest.approx(46.0, abs=0.11)
        assert home - away == pytest.approx(4.0, abs=0.11)
        assert p.elo_per_point == pytest.approx(25.0)

    def test_team_rates_shrink_toward_league_and_clip(self):
        r = scoring.TeamRates()
        assert r.attack("Nobody") == pytest.approx(1.0)
        for _ in range(20):
            r.observe("Hot", "Cold", 45, 3)
        assert r.attack("Hot") > 1.3 and r.defense("Hot") < 0.6
        assert scoring.TOTAL_MIN <= r.matchup_total("Hot", "Cold") <= scoring.TOTAL_MAX


class TestAsOf:
    def test_results_on_or_after_run_date_are_masked(self):
        g = _games([_row("a", "2026-09-13", "KC", "DEN", 20, 10),
                    _row("b", "2026-09-14", "SEA", "SF", 7, 3)])
        m = as_of(g, "2026-09-14")
        assert bool(m.loc[0, "completed"]) and not bool(m.loc[1, "completed"])
        assert pd.isna(m.loc[1, "home_score"])


class TestSlate:
    def test_slate_is_the_next_week_with_pick_and_line(self):
        g = _games([
            _row("w1a", "2026-09-10", "KC", "DEN", 24, 17, week=1),   # played
            _row("w1b", "2026-09-13", "SEA", "SF", week=1),
            _row("w1c", "2026-09-14", "NE", "BUF", week=1, a_rest=14),
            _row("w2a", "2026-09-20", "KC", "SEA", week=2),           # next week
        ])
        st = _state(g, {"SEA": 1500.0, "SF": 1650.0, "NE": 1500.0, "BUF": 1500.0})
        slate = predict.build_slate(st, "2026-09-12")
        assert list(slate["game_id"]) == ["w1b", "w1c"]
        sea = slate.iloc[0]
        assert sea["pick"] == "SF" and sea["p_home"] < 0.5
        assert sea["pick_prob"] == pytest.approx(1 - sea["p_home"], abs=1e-4)
        assert sea["elo_spread"] < 0 and sea["pred_away_score"] > sea["pred_home_score"]
        assert sea["week_label"] == "Week 1"
        # Buffalo off its bye: the away side gets the rest edge, so the
        # home team is no longer favoured by the full home advantage.
        ne = slate.iloc[1]
        assert 0.5 < ne["p_home"] < st.engine.pregame("NE", "BUF")[2]
        assert predict.next_week(st, "2026-09-15") == (2026, 2)
        assert predict.build_slate(st, "2026-09-15")["game_id"].tolist() == ["w2a"]
        assert predict.build_slate(st, "2026-09-21").empty


class TestGrading:
    @pytest.fixture(autouse=True)
    def _tmp_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(grade, "PREDICTIONS_DIR", tmp_path)
        monkeypatch.setattr(grade, "GRADES_CSV", tmp_path / "grades.csv")
        self.dir = tmp_path

    def _write_slate(self, name="slate_2026-09-08.csv", p_home_a=0.6):
        pd.DataFrame([
            {"game_id": "a", "date": "2026-09-13", "season": 2026, "week": 1,
             "game_type": "REG", "home_team": "KC", "away_team": "DEN", "neutral": False,
             "pick": "KC" if p_home_a >= 0.5 else "DEN", "p_home": p_home_a,
             "pick_prob": max(p_home_a, 1 - p_home_a), "elo_spread": 3.0,
             "pred_home_score": 26.0, "pred_away_score": 23.0},
            {"game_id": "b", "date": "2026-09-14", "season": 2026, "week": 1,
             "game_type": "REG", "home_team": "SEA", "away_team": "SF", "neutral": True,
             "pick": "SF", "p_home": 0.3, "pick_prob": 0.7, "elo_spread": -5.0,
             "pred_home_score": 20.0, "pred_away_score": 25.0},
        ]).to_csv(self.dir / name, index=False)

    def test_grades_only_played_and_scores_paired_baseline(self):
        self._write_slate()
        games = _games([_row("a", "2026-09-13", "KC", "DEN", 24, 21),
                        _row("b", "2026-09-14", "SEA", "SF")])
        graded = grade.grade_all(games, "2026-09-14")
        assert len(graded) == 1
        r = graded.iloc[0]
        assert bool(r["pick_correct"]) and r["home_win"] == 1.0 and not bool(r["tie"])
        assert r["log_loss"] == pytest.approx(-math.log(0.6), abs=1e-3)
        assert r["home_log_loss"] == pytest.approx(-math.log(ALWAYS_HOME_P), abs=1e-3)
        assert r["d_ll"] == pytest.approx(r["log_loss"] - r["home_log_loss"], abs=1e-3)
        assert r["margin_err"] == pytest.approx(0.0)
        assert r["week_label"] == "Week 1"

    def test_earliest_slate_wins_and_grading_is_idempotent(self):
        self._write_slate("slate_2026-09-08.csv", p_home_a=0.6)
        self._write_slate("slate_2026-09-12.csv", p_home_a=0.9)   # re-predicted later
        games = _games([_row("a", "2026-09-13", "KC", "DEN", 24, 21),
                        _row("b", "2026-09-14", "SEA", "SF", 10, 31, neutral=True)])
        first = grade.grade_all(games, "2026-09-15")
        assert len(first) == 2
        assert first[first["game_id"] == "a"].iloc[0]["p_home"] == pytest.approx(0.6)
        neutral = first[first["game_id"] == "b"].iloc[0]
        assert neutral["home_log_loss"] == pytest.approx(-math.log(0.5), abs=1e-3)
        assert bool(neutral["pick_correct"])
        assert len(grade.grade_all(games, "2026-09-16")) == 0
        summary = grade.ledger_summary("2026-09-16")
        assert summary["graded"] == 2 and summary["correct"] == 2
        assert summary["rolling"]["7d"]["graded"] == 2
        assert summary["by_week"]["Week 1"]["graded"] == 2
        assert summary["d_ll_se"] is not None

    def test_tie_is_graded_as_half_and_never_correct(self):
        self._write_slate()
        games = _games([_row("a", "2026-09-13", "KC", "DEN", 20, 20)])
        graded = grade.grade_all(games, "2026-09-14")
        r = graded.iloc[0]
        assert bool(r["tie"]) and r["home_win"] == 0.5 and not bool(r["pick_correct"])
        assert r["log_loss"] == pytest.approx(-(0.5 * math.log(0.6) + 0.5 * math.log(0.4)), abs=1e-3)
        assert grade.ledger_summary()["ties"] == 1

    def test_recent_grades_window(self):
        self._write_slate()
        games = _games([_row("a", "2026-09-13", "KC", "DEN", 24, 21),
                        _row("b", "2026-09-14", "SEA", "SF", 10, 31, neutral=True)])
        grade.grade_all(games, "2026-09-15")
        recent = grade.recent_grades("2026-09-20", days=7)
        assert list(recent["game_id"]) == ["a", "b"]
        assert grade.recent_grades("2026-09-30", days=7).empty


class TestSimulate:
    def _league(self, played_playoffs=False):
        rows = [_row("p1", "2026-09-10", "KC", "DEN", 28, 10, week=1)]
        gid = 2
        # Every team plays the other three in its division once (played
        # nothing else), then a handful of cross-division games.
        for div, members in DIVISIONS.items():
            for i, h in enumerate(members):
                for a in members[i + 1:]:
                    if {h, a} == {"KC", "DEN"}:
                        continue
                    rows.append(_row(f"g{gid}", "2026-10-01", h, a, week=4))
                    gid += 1
        teams = list(TEAMS)
        for i in range(0, 32, 2):
            rows.append(_row(f"x{gid}", "2026-11-01", teams[i], teams[(i + 5) % 32], week=9))
            gid += 1
        games = _games(rows)
        if played_playoffs:
            games = pd.concat([games, _games([
                _row("wc1", "2027-01-10", "KC", "LV", 30, 27, week=19, gt="WC")])],
                ignore_index=True)
        ratings = {t: 1600 - 6 * i for i, t in enumerate(teams)}
        ratings["KC"] = 1750.0
        return _state(games, ratings)

    def test_records(self):
        st = self._league()
        rec = simulate.current_records(st.games, 2026).set_index("team")
        assert rec.loc["KC", "wins"] == 1 and rec.loc["KC", "div_wins"] == 1
        assert rec.loc["DEN", "losses"] == 1 and rec.loc["DEN", "pts_diff"] == -18
        assert rec.loc["DEN", "conf_losses"] == 1

    def test_probability_mass_identities(self):
        st = self._league()
        sim = simulate.simulate_season(st, n_sims=500, seed=3)
        teams = sim["teams"]
        assert len(teams) == 32
        by_div, by_conf = {}, {}
        for t in teams:
            by_div.setdefault(t["division"], []).append(t)
            by_conf.setdefault(t["conference"], []).append(t)
        for members in by_div.values():
            assert sum(m["p_division"] for m in members) == pytest.approx(1.0, abs=2e-3)
        for members in by_conf.values():
            assert sum(m["p_playoffs"] for m in members) == pytest.approx(7.0, abs=5e-3)
            assert sum(m["p_top_seed"] for m in members) == pytest.approx(1.0, abs=2e-3)
            assert sum(m["p_conf"] for m in members) == pytest.approx(1.0, abs=2e-3)
        assert sum(t["p_sb"] for t in teams) == pytest.approx(1.0, abs=2e-3)
        kc = next(t for t in teams if t["team"] == "KC")
        assert kc["wins"] == 1 and kc["exp_wins"] >= 1.0
        assert kc["p_playoffs"] <= 1.0 and kc["p_division"] >= kc["p_top_seed"]
        # Strongest team is the most likely AFC West champion and the table
        # is ordered by Super Bowl odds.
        west = sorted(by_div["AFC West"], key=lambda t: -t["p_division"])
        assert west[0]["team"] == "KC"
        assert teams[0]["p_sb"] >= teams[-1]["p_sb"]

    def test_played_playoff_game_is_honoured(self):
        st = self._league(played_playoffs=True)
        sim = simulate.simulate_season(st, n_sims=300, seed=1)
        lv = next(t for t in sim["teams"] if t["team"] == "LV")
        kc = next(t for t in sim["teams"] if t["team"] == "KC")
        # LV lost to KC in the real Wild Card game: whenever the sim
        # produces that matchup LV goes out, so LV can never beat KC's
        # conference-title odds via that path.
        assert lv["p_conf"] <= kc["p_conf"]
        assert sum(t["p_sb"] for t in sim["teams"]) == pytest.approx(1.0, abs=2e-3)
