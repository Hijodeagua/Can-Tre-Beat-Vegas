"""Tests for the NFL Elo engine: franchise continuity, the situational
edges (home, neutral, bye-week rest), ties, playoff K, the margin cap,
season regression, and the tuner's fast replay agreeing with the engine."""

import math

import pandas as pd
import pytest

from NFL.elo.engine import (
    BASE_RATING, REST_BONUS_DAYS, NflEloEngine, expected_score, fast_replay,
    load_games, replay,
)
from NFL.elo.teams import DIVISION_OF, TEAMS, canonical, week_label


def _engine(**kw):
    params = dict(k=20.0, home_advantage=50.0, season_regression=0.4,
                  playoff_k_mult=1.5, margin_cap=35.0, rest_bonus=25.0)
    params.update(kw)
    return NflEloEngine(**params)


class TestTeams:
    def test_thirty_two_teams_in_eight_divisions(self):
        assert len(TEAMS) == 32
        assert len({DIVISION_OF[t] for t in TEAMS}) == 8

    def test_relocations_map_to_current_franchise(self):
        assert canonical("STL") == "LA" and canonical("SD") == "LAC" and canonical("OAK") == "LV"
        assert canonical("KC") == "KC"

    def test_week_labels(self):
        assert week_label(7, "REG") == "Week 7"
        assert week_label(19, "WC") == "Wild Card"
        assert week_label(22, "SB") == "Super Bowl"


class TestEngineRules:
    def test_home_advantage_dropped_at_neutral_site(self):
        e = _engine()
        e.roll_season(2026)
        _, _, p_home = e.pregame("A", "B", neutral=False)
        _, _, p_neutral = e.pregame("A", "B", neutral=True)
        assert p_home > 0.5
        assert p_neutral == pytest.approx(0.5)

    def test_bye_week_rest_bonus_applies_at_the_threshold(self):
        e = _engine(rest_bonus=25.0)
        e.roll_season(2026)
        _, _, p_plain = e.pregame("A", "B", home_rest=7, away_rest=7)
        _, _, p_bye = e.pregame("A", "B", home_rest=REST_BONUS_DAYS, away_rest=7)
        _, _, p_away_bye = e.pregame("A", "B", home_rest=7, away_rest=14)
        assert p_bye > p_plain > p_away_bye
        assert p_bye == pytest.approx(expected_score(1500 + 50 + 25, 1500))
        # Both off a bye cancels out.
        _, _, p_both = e.pregame("A", "B", home_rest=14, away_rest=14)
        assert p_both == pytest.approx(p_plain)

    def test_zero_sum_update_and_favourite_loses_more(self):
        e = _engine()
        e.roll_season(2026)
        e.ratings["A"], e.ratings["B"] = 1600.0, 1450.0
        e.update("A", "B", 17, 24)
        assert e.ratings["A"] + e.ratings["B"] == pytest.approx(3050.0)
        assert e.ratings["A"] < 1600.0

    def test_tie_moves_the_favourite_down_and_scores_half(self):
        e = _engine()
        e.roll_season(2026)
        e.ratings["A"], e.ratings["B"] = 1600.0, 1500.0
        rec = e.update("A", "B", 20, 20)
        assert rec["home_win"] == 0.5
        assert e.ratings["A"] < 1600.0 and e.ratings["B"] > 1500.0
        # A tie between equals at a neutral site changes nothing.
        f = _engine()
        f.roll_season(2026)
        f.update("C", "D", 10, 10, neutral=True)
        assert f.ratings["C"] == pytest.approx(BASE_RATING)

    def test_playoff_multiplier_scales_the_update(self):
        reg, po = _engine(playoff_k_mult=1.5), _engine(playoff_k_mult=1.5)
        for e in (reg, po):
            e.roll_season(2026)
        reg.update("A", "B", 27, 20)
        po.update("A", "B", 27, 20, playoff=True)
        assert (po.ratings["A"] - 1500) == pytest.approx(1.5 * (reg.ratings["A"] - 1500))

    def test_margin_is_capped_before_the_multiplier(self):
        big, small = _engine(margin_cap=35.0), _engine(margin_cap=35.0)
        for e in (big, small):
            e.roll_season(2026)
        big.update("A", "B", 59, 0)
        small.update("A", "B", 35, 0)
        assert big.ratings["A"] == pytest.approx(small.ratings["A"])
        uncapped = _engine(margin_cap=60.0)
        uncapped.roll_season(2026)
        uncapped.update("A", "B", 59, 0)
        assert uncapped.ratings["A"] > big.ratings["A"]

    def test_season_regression_toward_base_is_idempotent(self):
        e = _engine(season_regression=0.4)
        e.roll_season(2025)
        e.ratings["A"] = 1700.0
        e.roll_season(2026)
        assert e.ratings["A"] == pytest.approx(1700 - 0.4 * 200)
        e.roll_season(2026)
        assert e.ratings["A"] == pytest.approx(1620.0)


def _spine(rows):
    cols = ["game_id", "season", "week", "game_type", "gameday", "gametime",
            "home_team", "away_team", "home_score", "away_score", "location",
            "home_rest", "away_rest"]
    return pd.DataFrame(rows, columns=cols)


def _row(gid, season, date, home, away, hs, ap, gt="REG", location="Home",
         h_rest=7, a_rest=7, week=1):
    return [gid, season, week, gt, date, "13:00", home, away, hs, ap, location, h_rest, a_rest]


class TestReplay:
    def test_load_games_canonicalises_and_flags(self, tmp_path):
        path = tmp_path / "games.csv"
        _spine([_row("a", 2015, "2015-09-13", "STL", "SEA", 34, 31),
                _row("b", 2015, "2016-02-07", "CAR", "DEN", 10, 24, gt="SB", location="Neutral"),
                _row("c", 2016, "2016-09-11", "LA", "SF", None, None)]).to_csv(path, index=False)
        g = load_games(path)
        assert list(g["home_team"]) == ["LA", "CAR", "LA"]
        assert list(g["completed"]) == [True, True, False]
        assert bool(g.loc[1, "neutral"]) and bool(g.loc[1, "playoff"])
        assert g.loc[0, "date"] == "2015-09-13"

    def test_replay_rolls_seasons_and_carries_the_franchise(self, tmp_path):
        path = tmp_path / "games.csv"
        _spine([_row("a", 2015, "2015-09-13", "STL", "SEA", 34, 31),
                _row("b", 2016, "2016-09-11", "LA", "SF", 28, 0)]).to_csv(path, index=False)
        g = load_games(path)
        e = _engine(season_regression=0.5)
        engine, hist = replay(g, engine=e)
        assert list(hist["game_id"]) == ["a", "b"]
        # The 2016 Rams start from the regressed 2015 STL rating, not 1500.
        after_2015 = hist.loc[0, "elo_home_pre"] + (hist.loc[1, "elo_home_pre"] - 1500) / 0.5
        assert hist.loc[1, "elo_home_pre"] != pytest.approx(1500.0)
        assert engine.current_season == 2016
        assert "STL" not in engine.ratings

    def test_fast_replay_matches_the_engine(self, tmp_path):
        path = tmp_path / "games.csv"
        rows = [_row("a", 2015, "2015-09-13", "A", "B", 34, 31, h_rest=14),
                _row("b", 2015, "2015-09-20", "B", "C", 17, 17),
                _row("c", 2015, "2016-01-10", "A", "C", 20, 27, gt="WC"),
                _row("d", 2016, "2016-09-11", "C", "A", 28, 0, location="Neutral")]
        _spine(rows).to_csv(path, index=False)
        g = load_games(path)
        params = dict(k=20.0, home_advantage=50.0, season_regression=0.4,
                      playoff_k_mult=1.5, margin_cap=35.0, rest_bonus=25.0)
        engine, hist = replay(g, engine=NflEloEngine(**params))
        fast_rows = list(zip(g["season"], g["home_team"], g["away_team"], g["neutral"],
                             g["playoff"], g["home_rest"] >= REST_BONUS_DAYS,
                             g["away_rest"] >= REST_BONUS_DAYS,
                             g["home_score"], g["away_score"]))
        ll, n = fast_replay(fast_rows, params, 2015, 2017)
        assert n == 4
        expected = 0.0
        for r in hist.itertuples():
            p = min(max(r.p_home, 1e-6), 1 - 1e-6)
            expected -= r.home_win * math.log(p) + (1 - r.home_win) * math.log(1 - p)
        assert ll == pytest.approx(expected / 4, abs=1e-9)
