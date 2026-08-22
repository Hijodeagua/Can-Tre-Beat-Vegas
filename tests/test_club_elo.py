"""Tests for the top-5-league club Elo models (soccer/clubs/)."""

import pandas as pd
import pytest

from soccer.clubs.data import fetch_results
from soccer.clubs.data.leagues import (
    ALIASES,
    LEAGUES,
    canonical,
    next_season,
    season_for_date,
)
from soccer.clubs.model.elo import ClubEloEngine, expected_score, mov_multiplier


def match(season, home, away, hs, as_, league="epl", date="2020-09-12"):
    return pd.Series(
        {
            "date": date,
            "season": season,
            "league": league,
            "home_team": home,
            "away_team": away,
            "home_score": hs,
            "away_score": as_,
        }
    )


class TestLeagues:
    def test_all_ten_leagues_registered(self):
        assert set(LEAGUES) == {
            "epl", "championship", "bundesliga", "bundesliga_2",
            "la_liga", "la_liga_2", "serie_a", "serie_b",
            "ligue_1", "ligue_2",
        }

    def test_second_divisions_share_their_country_pool(self):
        from soccer.clubs.data.leagues import POOLS, pool_of
        assert pool_of("championship") == "epl"
        assert pool_of("serie_b") == "serie_a"
        assert set(POOLS) == {"epl", "bundesliga", "la_liga", "serie_a", "ligue_1"}
        assert POOLS["epl"] == ["epl", "championship"]

    def test_d2_aliases_map_into_the_pool_canon(self):
        assert canonical("championship", "Leeds United") == "Leeds United FC"
        assert canonical("serie_b", "US Palermo") == "Palermo FC"
        assert canonical("serie_a", "US Palermo") == "Palermo FC"
        assert canonical("la_liga_2", "Espanyol Barcelona") == "RCD Espanyol de Barcelona"

    def test_canonical_maps_renamed_clubs(self):
        assert canonical("epl", "Manchester City") == "Manchester City FC"
        assert canonical("serie_a", "Inter") == "FC Internazionale Milano"
        assert canonical("ligue_1", "Paris Saint-Germain") == "Paris Saint-Germain FC"

    def test_canonical_passes_through_unknown_names(self):
        assert canonical("epl", "Arsenal FC") == "Arsenal FC"

    def test_aliases_never_map_to_another_alias(self):
        # A chain (old -> mid -> new) would leave mid-era rows uncanonicalized.
        for league, aliases in ALIASES.items():
            for target in aliases.values():
                assert target not in aliases, f"{league}: {target} is both key and value"

    def test_season_helpers(self):
        assert next_season("2024-25") == "2025-26"
        assert next_season("1999-00") == "2000-01"
        assert season_for_date("2026-08-20") == "2026-27"
        assert season_for_date("2026-05-24") == "2025-26"


class TestClubEloEngine:
    def test_update_is_zero_sum_and_symmetric(self):
        e = ClubEloEngine(k=20, home_advantage=0, entry_rating=1500)
        rec = e.update(match("2020-21", "A", "B", 2, 0))
        assert e.ratings["A"] + e.ratings["B"] == pytest.approx(3000)
        assert e.ratings["A"] > 1500 > e.ratings["B"]
        assert rec["outcome"] == "H"
        assert rec["elo_gap"] == 0

    def test_draw_moves_equal_ratings_nowhere_without_home_edge(self):
        e = ClubEloEngine(k=20, home_advantage=0, entry_rating=1500)
        e.update(match("2020-21", "A", "B", 1, 1))
        assert e.ratings["A"] == pytest.approx(1500)

    def test_home_advantage_enters_expectation_only(self):
        e = ClubEloEngine(k=20, home_advantage=100, entry_rating=1500)
        rec = e.update(match("2020-21", "A", "B", 1, 1))
        # Home was favored, so a draw costs the home side rating points.
        assert e.ratings["A"] < 1500 < e.ratings["B"]
        assert rec["exp_home"] > 0.5

    def test_mov_multiplier_convention(self):
        assert mov_multiplier(1) == 1.0
        assert mov_multiplier(-2) == 1.5
        assert mov_multiplier(3) == pytest.approx(14 / 8)
        assert mov_multiplier(5) == pytest.approx(2.0)

    def test_new_club_enters_at_entry_rating(self):
        e = ClubEloEngine(k=20, home_advantage=0, entry_rating=1420)
        rec = e.update(match("2020-21", "Promoted", "AlsoNew", 0, 1))
        assert rec["elo_home_pre"] == 1420
        assert rec["elo_away_pre"] == 1420

    def test_season_rollover_regresses_everyone_toward_base(self):
        e = ClubEloEngine(k=20, home_advantage=0, season_regression=0.5, entry_rating=1500)
        e.update(match("2020-21", "A", "B", 3, 0))
        high, low = e.ratings["A"], e.ratings["B"]
        rec = e.update(match("2021-22", "A", "B", 1, 1))
        # The pre-match ratings of the second game are the regressed values.
        assert e.current_season == "2021-22"
        assert rec["elo_home_pre"] == pytest.approx(1500 + 0.5 * (high - 1500))
        assert rec["elo_away_pre"] == pytest.approx(1500 + 0.5 * (low - 1500))

    def test_expected_score_is_logistic(self):
        assert expected_score(1500, 1500) == 0.5
        assert expected_score(1900, 1500) == pytest.approx(10 / 11)

    def test_tier2_entry_rating(self):
        e = ClubEloEngine(k=20, home_advantage=0,
                          entry_rating=1420, entry_rating_t2=1250)
        rec = e.update(match("2020-21", "NewD2A", "NewD2B", 1, 1,
                             league="championship"))
        assert rec["elo_home_pre"] == 1250
        assert rec["elo_away_pre"] == 1250

    def test_division_switch_blends_toward_entry(self):
        e = ClubEloEngine(k=0, home_advantage=0, entry_rating=1400,
                          entry_rating_t2=1250, division_carry=0.5)
        # K=0: ratings only move via the switch blend, so the math is exact.
        e.ratings["Promoted"] = 1500
        e.last_league["Promoted"] = "championship"
        e.ratings["Incumbent"] = 1600
        e.last_league["Incumbent"] = "epl"
        rec = e.update(match("2021-22", "Promoted", "Incumbent", 0, 0, league="epl"))
        assert rec["elo_home_pre"] == pytest.approx(1400 + 0.5 * (1500 - 1400))
        assert rec["elo_away_pre"] == 1600  # no switch, no blend
        assert e.last_league["Promoted"] == "epl"

    def test_rating_for_previews_the_blend_without_mutating(self):
        e = ClubEloEngine(entry_rating=1400, entry_rating_t2=1250,
                          division_carry=0.25)
        e.ratings["Up"] = 1500
        e.last_league["Up"] = "la_liga_2"
        assert e.rating_for("Up", "la_liga") == pytest.approx(1400 + 0.25 * 100)
        assert e.rating_for("Up", "la_liga_2") == 1500  # same tier: unchanged
        assert e.ratings["Up"] == 1500                  # non-mutating
        assert e.rating_for("Unknown", "la_liga_2") == 1250


class TestFetchNormalization:
    def _rows(self, monkeypatch, matches):
        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"matches": matches}

        monkeypatch.setattr(fetch_results.requests, "get", lambda *a, **k: Resp())
        return fetch_results.fetch_season_json("epl", "2025-26")

    def test_accepts_both_score_shapes_and_flags_unplayed(self, monkeypatch):
        rows = self._rows(
            monkeypatch,
            [
                {"date": "2025-08-16", "team1": "Arsenal FC", "team2": "Fulham FC",
                 "score": {"ft": [2, 1], "ht": [1, 0]}},
                {"date": "2025-08-16", "team1": "Aston Villa FC",
                 "team2": "Newcastle United FC", "score": [0, 0]},
                {"date": "2026-05-24", "team1": "Everton FC", "team2": "Burnley FC",
                 "status": "cancelled"},
            ],
        )
        assert (rows[0]["home_score"], rows[0]["away_score"]) == (2, 1)
        assert (rows[1]["home_score"], rows[1]["away_score"]) == (0, 0)
        assert rows[2]["home_score"] == ""

    def test_canonicalizes_team_names(self, monkeypatch):
        rows = self._rows(
            monkeypatch,
            [{"date": "2019-08-10", "team1": "Manchester City", "team2": "West Ham United",
              "score": {"ft": [5, 0]}}],
        )
        assert rows[0]["home_team"] == "Manchester City FC"
        assert rows[0]["away_team"] == "West Ham United FC"
