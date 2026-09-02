"""Tests for the college-football Elo engine and the cfbfastR spine
normalizer: the four college-specific rules (conference-mean regression,
pooled FCS opponent, FBS entry rating, capped margin) plus the parse
conventions that would silently corrupt the replay if they drifted."""

import pandas as pd
import pytest

from CFB.data.fetch_schedule import normalize, parse_scoreboard
from CFB.model.elo import (
    BASE_RATING, INDEPENDENT, CfbEloEngine, expected_score, fast_replay,
    replay, season_conferences,
)


def _engine(**kw):
    params = dict(k=30.0, home_advantage=60.0, season_regression=0.5,
                  conf_weight=1.0, fcs_rating=1000.0, entry_rating=1300.0,
                  margin_cap=35.0)
    params.update(kw)
    return CfbEloEngine(**params)


class TestEngineRules:
    def test_entry_rating_for_first_fbs_game_and_fcs_pool(self):
        e = _engine()
        e.roll_season(2026, {"A": "SEC", "B": "SEC"})
        assert e.rating_for("A") == 1300.0
        assert e.rating_for("Some FCS", "fcs") == 1000.0
        assert e.rating_for("Some FCS", None) == 1000.0

    def test_fcs_side_never_updates_and_fbs_side_does(self):
        e = _engine()
        e.roll_season(2026, {"A": "SEC"})
        e.update("A", "Podunk", 45, 7, home_division="fbs", away_division="fcs")
        assert "Podunk" not in e.ratings
        assert e.ratings["A"] > 1300.0
        assert e.games_played["A"] == 1

    def test_home_advantage_dropped_at_neutral_site(self):
        e = _engine()
        e.roll_season(2026, {"A": "SEC", "B": "SEC"})
        e.ratings["A"] = e.ratings["B"] = 1500.0
        _, _, p_home = e.pregame("A", "B", "fbs", "fbs", neutral=False)
        _, _, p_neutral = e.pregame("A", "B", "fbs", "fbs", neutral=True)
        assert p_home > 0.5
        assert p_neutral == pytest.approx(0.5)

    def test_zero_sum_update(self):
        e = _engine()
        e.roll_season(2026, {"A": "SEC", "B": "SEC"})
        e.ratings["A"], e.ratings["B"] = 1600.0, 1450.0
        e.update("A", "B", 21, 24)
        assert e.ratings["A"] + e.ratings["B"] == pytest.approx(3050.0)
        assert e.ratings["A"] < 1600.0  # favourite lost

    def test_margin_is_capped_before_the_multiplier(self):
        big = _engine(margin_cap=35.0)
        small = _engine(margin_cap=35.0)
        for e in (big, small):
            e.roll_season(2026, {"A": "SEC", "B": "SEC"})
            e.ratings["A"] = e.ratings["B"] = 1500.0
        big.update("A", "B", 70, 0)
        small.update("A", "B", 35, 0)
        assert big.ratings["A"] == pytest.approx(small.ratings["A"])
        uncapped = _engine(margin_cap=100.0)
        uncapped.roll_season(2026, {"A": "SEC", "B": "SEC"})
        uncapped.ratings["A"] = uncapped.ratings["B"] = 1500.0
        uncapped.update("A", "B", 70, 0)
        assert uncapped.ratings["A"] > big.ratings["A"]

    def test_conference_mean_regression_uses_new_membership(self):
        e = _engine(season_regression=0.5, conf_weight=1.0)
        e.roll_season(2025, {"A": "Big 12", "B": "Big 12", "C": "SEC", "D": "SEC"})
        e.ratings.update({"A": 1700.0, "B": 1500.0, "C": 1400.0, "D": 1400.0})
        # A realigns to the SEC for 2026: it regresses toward the SEC mean
        # computed with A as a member (1500), not the Big 12's.
        e.roll_season(2026, {"A": "SEC", "B": "Big 12", "C": "SEC", "D": "SEC"})
        sec_mean = (1700 + 1400 + 1400) / 3
        assert e.ratings["A"] == pytest.approx(1700 + 0.5 * (sec_mean - 1700))
        assert e.ratings["B"] == pytest.approx(1500.0)      # alone at its mean
        assert e.conference["A"] == "SEC"

    def test_conf_weight_zero_is_flat_regression_to_base(self):
        e = _engine(season_regression=0.4, conf_weight=0.0)
        e.roll_season(2025, {"A": "SEC", "B": "SEC"})
        e.ratings.update({"A": 1800.0, "B": 1800.0})
        e.roll_season(2026, {"A": "SEC", "B": "SEC"})
        assert e.ratings["A"] == pytest.approx(BASE_RATING + 0.6 * 300)

    def test_independents_and_dropped_programs_regress_to_base(self):
        e = _engine(season_regression=1.0, conf_weight=1.0)
        e.roll_season(2025, {"ND": INDEPENDENT, "X": "MAC"})
        e.ratings.update({"ND": 1800.0, "X": 1300.0})
        e.roll_season(2026, {"ND": INDEPENDENT})  # X left FBS
        assert e.ratings["ND"] == pytest.approx(BASE_RATING)
        assert e.ratings["X"] == pytest.approx(BASE_RATING)

    def test_same_season_does_not_re_regress(self):
        e = _engine(season_regression=0.5)
        e.roll_season(2026, {"A": "SEC"})
        e.ratings["A"] = 1700.0
        e.roll_season(2026, {"A": "SEC"})
        assert e.ratings["A"] == 1700.0


def _spine(rows):
    cols = ["game_id", "season", "week", "season_type", "date", "start_utc",
            "home_team", "away_team", "home_conference", "away_conference",
            "home_division", "away_division", "home_points", "away_points",
            "neutral_site", "conference_game", "completed", "notes"]
    return pd.DataFrame(rows, columns=cols)


def _row(gid, season, date, home, away, hp, ap, hconf="SEC", aconf="SEC",
         hdiv="fbs", adiv="fbs", neutral=False, completed=True):
    return [gid, season, 1, "regular", date, f"{date}T20:00:00Z", home, away,
            hconf, aconf, hdiv, adiv, hp, ap, neutral, hconf == aconf, completed, ""]


class TestReplay:
    def test_fast_replay_matches_engine(self):
        games = _spine([
            _row(1, 2024, "2024-09-01", "A", "B", 31, 10),
            _row(2, 2024, "2024-09-08", "B", "C", 17, 20, aconf="ACC"),
            _row(3, 2024, "2024-09-15", "C", "Podunk", 42, 0, hconf="ACC", adiv="fcs", aconf=None),
            _row(4, 2025, "2025-09-01", "A", "C", 24, 27, aconf="ACC", neutral=True),
            _row(5, 2025, "2025-09-08", "B", "A", 14, 35),
            _row(6, 2026, "2026-09-05", "A", "B", None, None, completed=False),
        ])
        params = dict(k=30.0, home_advantage=60.0, season_regression=0.5,
                      conf_weight=0.75, fcs_rating=1000.0, entry_rating=1300.0,
                      margin_cap=35.0)
        engine, history = replay(games, engine=CfbEloEngine(**params))
        assert len(history) == 5                       # unplayed game skipped
        rows = list(zip(history["season"], history["home_team"], history["away_team"],
                        ~history["home_fcs"], ~history["away_fcs"], history["neutral"],
                        history["home_points"], history["away_points"]))
        import numpy as np
        p = history["p_home"].clip(1e-6, 1 - 1e-6)
        y = history["home_win"]
        ll_engine = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        ll_fast, n = fast_replay(rows, season_conferences(games), params, 2024, 2027)
        assert n == 5
        assert ll_fast == pytest.approx(ll_engine, abs=1e-9)

    def test_end_cuts_the_replay_walk_forward(self):
        games = _spine([
            _row(1, 2024, "2024-09-01", "A", "B", 31, 10),
            _row(2, 2024, "2024-09-08", "A", "B", 31, 10),
        ])
        _, history = replay(games, engine=_engine(), end="2024-09-08")
        assert list(history["game_id"]) == [1]

    def test_season_conferences_reads_both_sides(self):
        games = _spine([_row(1, 2024, "2024-09-01", "A", "B", 3, 0, hconf="SEC", aconf="ACC")])
        assert season_conferences(games) == {2024: {"A": "SEC", "B": "ACC"}}


class TestNormalize:
    def _raw(self, **over):
        base = {
            "game_id": 401000001, "season": 2026, "week": 1, "season_type": "regular",
            "start_date": "2026-08-30T02:00:00.000Z", "start_time_tbd": "FALSE",
            "completed": "TRUE", "neutral_site": "FALSE", "conference_game": "FALSE",
            "home_team": "UNLV", "home_division": "fbs", "home_conference": "Mountain West",
            "home_points": 21, "away_team": "Memphis", "away_division": "fbs",
            "away_conference": "American Athletic", "away_points": 27, "notes": None,
        }
        base.update(over)
        return pd.DataFrame([base])

    def test_kickoff_date_is_eastern(self):
        # 02:00Z on Aug 30 is 10pm ET on Aug 29 — the slate date.
        out = normalize(self._raw())
        assert out.loc[0, "date"] == "2026-08-29"
        assert out.loc[0, "home_points"] == 21 and out.loc[0, "completed"]

    def test_non_fbs_games_dropped_but_fbs_vs_fcs_kept(self):
        kept = normalize(self._raw(away_division="fcs"))
        dropped = normalize(self._raw(home_division="fcs", away_division="fcs"))
        assert len(kept) == 1 and len(dropped) == 0

    def test_completed_without_score_is_unplayed(self):
        out = normalize(self._raw(home_points=None, away_points=None))
        assert not out.loc[0, "completed"]

    def test_aliases_applied(self):
        out = normalize(self._raw(home_team="Connecticut"))
        assert out.loc[0, "home_team"] == "UConn"


class TestScoreboard:
    def test_parse_scoreboard_keeps_completed_only(self):
        payload = {"events": [
            {"id": "401000001", "competitions": [{
                "neutralSite": False,
                "status": {"type": {"completed": True}},
                "competitors": [
                    {"homeAway": "home", "score": "21", "team": {"displayName": "UNLV"}},
                    {"homeAway": "away", "score": "27", "team": {"displayName": "Memphis"}},
                ]}]},
            {"id": "401000002", "competitions": [{
                "status": {"type": {"completed": False}},
                "competitors": [
                    {"homeAway": "home", "score": "7"},
                    {"homeAway": "away", "score": "3"},
                ]}]},
        ]}
        finals = parse_scoreboard(payload)
        assert finals == {401000001: (21.0, 27.0, False)}
