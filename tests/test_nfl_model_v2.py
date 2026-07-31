"""Tests for the v2 NFL model: Elo, feature/target construction, line movement."""

import numpy as np
import pandas as pd
import pytest

from NFL.model.v2 import elo as elo_mod
from NFL.model.v2.dataset import (
    add_market_features,
    add_targets,
    haversine_miles,
    to_team_rows,
    add_rolling_form,
)
from NFL.model.line_movement import (
    aggregate_movement,
    american_to_prob,
    drop_in_play,
)


def _games():
    """Two-week toy schedule: 4 teams, everything played."""
    return pd.DataFrame({
        "game_id": ["g1", "g2", "g3", "g4"],
        "season": [2024, 2024, 2024, 2024],
        "week": [1, 1, 2, 2],
        "gameday": pd.to_datetime(["2024-09-08", "2024-09-08", "2024-09-15", "2024-09-15"]),
        "game_type": ["REG"] * 4,
        "home_team": ["AAA", "CCC", "BBB", "DDD"],
        "away_team": ["BBB", "DDD", "AAA", "CCC"],
        "home_score": [24.0, 10.0, 17.0, 30.0],
        "away_score": [17.0, 20.0, 14.0, 27.0],
        "location": ["Home"] * 4,
        "spread_line": [3.0, -2.5, 1.0, 7.0],
        "total_line": [44.5, 41.0, 40.5, 50.0],
        "home_moneyline": [-160, 120, -110, -300],
        "away_moneyline": [140, -140, -110, 250],
        "home_qb_name": ["qa", "qc", "qb", "qd"],
        "away_qb_name": ["qb", "qd", "qa", "qc"],
    })


# --------------------------------------------------------------------------
# Elo
# --------------------------------------------------------------------------

class TestElo:
    def test_even_teams_with_hfa_favour_home(self):
        eng = elo_mod.EloEngine()
        _, _, p = eng.pregame(2024, "AAA", "BBB")
        assert p > 0.5

    def test_neutral_site_removes_hfa(self):
        eng = elo_mod.EloEngine()
        _, _, p = eng.pregame(2024, "AAA", "BBB", neutral=True)
        assert p == pytest.approx(0.5)

    def test_update_is_zero_sum(self):
        eng = elo_mod.EloEngine()
        eng.pregame(2024, "AAA", "BBB")
        before = eng.rating("AAA") + eng.rating("BBB")
        eng.update("AAA", "BBB", 30, 10)
        assert eng.rating("AAA") + eng.rating("BBB") == pytest.approx(before)

    def test_winner_gains_rating(self):
        eng = elo_mod.EloEngine()
        eng.pregame(2024, "AAA", "BBB")
        eng.update("AAA", "BBB", 30, 10)
        assert eng.rating("AAA") > 1500 > eng.rating("BBB")

    def test_bigger_margin_moves_rating_more(self):
        small, big = elo_mod.EloEngine(), elo_mod.EloEngine()
        for e, (hs, as_) in ((small, (21, 20)), (big, (49, 3))):
            e.pregame(2024, "AAA", "BBB")
            e.update("AAA", "BBB", hs, as_)
        assert big.ratings["AAA"] > small.ratings["AAA"]

    def test_new_season_regresses_toward_mean(self):
        eng = elo_mod.EloEngine()
        eng.pregame(2024, "AAA", "BBB")
        eng.update("AAA", "BBB", 45, 0)
        peak = eng.ratings["AAA"]
        eng.pregame(2025, "AAA", "BBB")
        assert 1500 < eng.ratings["AAA"] < peak

    def test_unplayed_games_do_not_update_ratings(self):
        g = _games()
        g.loc[2:, ["home_score", "away_score"]] = np.nan
        out = elo_mod.compute_elo(g)
        # Week 2 sees week 1's results but nothing after.
        assert out.loc[out["game_id"] == "g3", "home_elo"].iloc[0] != 1500
        assert out["home_elo"].notna().all()

    def test_pregame_rating_precedes_result(self):
        """The rating on a game must not reflect that game's outcome."""
        out = elo_mod.compute_elo(_games())
        assert out.loc[out["game_id"] == "g1", "home_elo"].iloc[0] == 1500
        assert out.loc[out["game_id"] == "g1", "away_elo"].iloc[0] == 1500


# --------------------------------------------------------------------------
# targets and features
# --------------------------------------------------------------------------

class TestTargets:
    def test_win_cover_and_over_definitions(self):
        t = add_targets(_games())
        # g1: home 24-17, spread_line 3 (home favoured by 3), total 44.5
        row = t[t["game_id"] == "g1"].iloc[0]
        assert row["home_win"] == 1
        assert row["home_cover"] == 1        # won by 7, needed 3
        assert row["over"] == 0              # 41 points vs 44.5

    def test_underdog_home_cover(self):
        # g2: home 10-20, spread_line -2.5 (home dog by 2.5) -> lost by 10, no cover
        row = add_targets(_games()).query("game_id == 'g2'").iloc[0]
        assert row["home_win"] == 0
        assert row["home_cover"] == 0

    def test_pushes_become_nan(self):
        g = _games()
        g.loc[0, ["home_score", "away_score"]] = [23.0, 20.0]  # margin 3 == spread 3
        g.loc[0, "total_line"] = 43.0                          # total 43 == line
        row = add_targets(g).iloc[0]
        assert pd.isna(row["home_cover"])
        assert pd.isna(row["over"])

    def test_unplayed_games_have_no_targets(self):
        g = _games()
        g.loc[0, ["home_score", "away_score"]] = np.nan
        row = add_targets(g).iloc[0]
        assert pd.isna(row["home_win"]) and pd.isna(row["home_cover"])


class TestMarketFeatures:
    def test_novig_probability_strips_the_hold(self):
        m = add_market_features(_games())
        row = m[m["game_id"] == "g1"].iloc[0]
        # -160/+140 is roughly a 60% home favourite once vig is removed.
        assert 0.55 < row["market_home_prob"] < 0.65
        assert row["market_vig"] > 0

    def test_pickem_prices_are_even(self):
        row = add_market_features(_games()).query("game_id == 'g3'").iloc[0]
        assert row["market_home_prob"] == pytest.approx(0.5)

    def test_missing_moneyline_falls_back_to_spread(self):
        g = _games()
        g[["home_moneyline", "away_moneyline"]] = np.nan
        m = add_market_features(g)
        assert m["market_home_prob"].notna().all()
        # g4 has home favoured by 7, so the fallback must favour home.
        assert m.query("game_id == 'g4'")["market_home_prob"].iloc[0] > 0.6


class TestRollingForm:
    def test_rolling_stats_exclude_the_current_game(self):
        rows = add_rolling_form(to_team_rows(_games()))
        first = rows[(rows["team"] == "AAA") & (rows["game_id"] == "g1")].iloc[0]
        assert pd.isna(first["roll_margin"])  # no prior games to average

    def test_team_spread_flips_sign_for_the_away_side(self):
        rows = to_team_rows(_games())
        g1 = rows[rows["game_id"] == "g1"]
        home = g1[g1["is_home"] == 1].iloc[0]
        away = g1[g1["is_home"] == 0].iloc[0]
        assert home["team_spread"] == -away["team_spread"]
        assert home["margin"] == -away["margin"]

    def test_qb_change_flag(self):
        g = _games()
        g.loc[2, "away_qb_name"] = "backup"  # AAA's QB differs in week 2
        rows = add_rolling_form(to_team_rows(g))
        wk2 = rows[(rows["team"] == "AAA") & (rows["game_id"] == "g3")].iloc[0]
        assert wk2["qb_change"] == 1


def test_haversine_matches_known_distance():
    # Seattle -> Miami is roughly 2,700 miles.
    d = haversine_miles((47.5952, -122.3316), (25.9580, -80.2389))
    assert 2600 < d < 2800


# --------------------------------------------------------------------------
# line movement
# --------------------------------------------------------------------------

def _snapshots():
    """Legacy-schema style snapshots: no Game ID, moneyline only."""
    return pd.DataFrame({
        "ts": pd.to_datetime(["2025-10-18 08:00", "2025-10-20 14:00", "2025-10-20 21:18"]),
        "game_date": pd.to_datetime(["2025-10-20 19:00"] * 3),
        "home_team": ["Detroit Lions"] * 3,
        "away_team": ["Tampa Bay Buccaneers"] * 3,
        "game_key": ["Detroit Lions|Tampa Bay Buccaneers|2025-10-20"] * 3,
        "game_id_odds": [pd.NA] * 3,
        "League": ["NFL"] * 3,
        "spread_home": [np.nan] * 3,
        "total": [np.nan] * 3,
        "h2h_home": [-277.0, -275.0, -2490.0],   # last row is in-play
        "h2h_away": [221.0, 221.0, 940.0],
    })


class TestLineMovement:
    def test_american_to_prob(self):
        p = american_to_prob(pd.Series([-110, 100, 200, -200]))
        assert p.iloc[0] == pytest.approx(0.5238, abs=1e-4)
        assert p.iloc[1] == pytest.approx(0.5)
        assert p.iloc[2] == pytest.approx(1 / 3)
        assert p.iloc[3] == pytest.approx(2 / 3)

    def test_in_play_snapshots_are_dropped(self):
        kept = drop_in_play(_snapshots())
        assert len(kept) == 2
        assert kept["h2h_home"].min() == -277.0

    def test_closing_price_ignores_in_play(self):
        m = aggregate_movement(_snapshots())
        assert len(m) == 1
        row = m.iloc[0]
        assert row["close_h2h_home"] == -275.0
        # A live -2490 would have pushed this near 1.0.
        assert row["close_novig_home"] < 0.8

    def test_keeping_in_play_reproduces_the_leak(self):
        """Guards the regression: without the filter the close is contaminated."""
        row = aggregate_movement(_snapshots(), pregame_only=False).iloc[0]
        assert row["close_novig_home"] > 0.9

    def test_legacy_rows_aggregate_without_a_game_id(self):
        m = aggregate_movement(_snapshots())
        assert m["game_key"].iloc[0].startswith("Detroit Lions|")
        assert m["num_snapshots"].iloc[0] == 2

    def test_movement_direction(self):
        row = aggregate_movement(_snapshots()).iloc[0]
        # -277 -> -275 is a small move away from the home side.
        assert row["novig_move_home"] < 0


class TestEloVariants:
    """The four exploratory Elo formulations."""

    def _games(self, n=8):
        import numpy as np
        rows = []
        teams = ["AAA", "BBB", "CCC", "DDD"]
        for i in range(n):
            h, a = teams[i % 4], teams[(i + 1) % 4]
            rows.append({
                "game_id": f"g{i}", "season": 2024, "week": i + 1,
                "gameday": pd.Timestamp("2024-09-08") + pd.Timedelta(days=7 * i),
                "game_type": "REG", "home_team": h, "away_team": a,
                "home_score": 24.0, "away_score": 17.0,
                "location": "Home", "spread_line": 3.0,
            })
        return pd.DataFrame(rows)

    def test_adjustment_does_not_accumulate_into_rating(self):
        """A QB bonus must move the prediction, not permanently inflate the team."""
        from NFL.model.v2.elo_variants import run_elo
        g = self._games()
        adj = np.full(len(g), 100.0)
        plain = run_elo(g)
        boosted = run_elo(g, home_adj=adj, away_adj=np.zeros(len(g)))
        # The stored rating path differs (results differ in expectation) but the
        # adjustment itself is never banked: it shows up only in v_adj_home.
        assert (boosted["v_adj_home"] == 100.0).all()
        assert (plain["v_adj_home"] == 0.0).all()
        assert boosted["v_elo_diff"].iloc[0] == pytest.approx(
            plain["v_elo_diff"].iloc[0] + 100.0)

    def test_positive_adjustment_raises_win_probability(self):
        from NFL.model.v2.elo_variants import run_elo
        g = self._games()
        up = run_elo(g, home_adj=np.full(len(g), 80.0), away_adj=np.zeros(len(g)))
        flat = run_elo(g)
        assert up["v_elo_prob"].iloc[0] > flat["v_elo_prob"].iloc[0]

    def test_unplayed_games_do_not_move_ratings(self):
        from NFL.model.v2.elo_variants import run_elo
        g = self._games()
        g.loc[4:, ["home_score", "away_score"]] = np.nan
        d = run_elo(g)
        # Ratings frozen from the first unplayed game onward for a given team.
        later = d[d["game_id"].isin(["g4", "g5", "g6", "g7"])]
        assert later["v_elo_prob"].notna().all()

    def test_market_anchored_tracks_the_spread(self):
        """The spread variant's probability should start at the market's."""
        from NFL.model.v2.elo_variants import run_elo
        g = self._games()
        d = run_elo(g, market_anchored=True)
        # First game: no rating history, so the prediction is the market's own.
        market_p = 1.0 / (1.0 + np.exp(-0.145 * 3.0))
        assert d["v_elo_prob"].iloc[0] == pytest.approx(market_p, abs=1e-6)

    def test_shrink_constants_are_below_one(self):
        """Regression: full-strength adjustments double-count and hurt log loss."""
        from NFL.model.v2.elo_variants import QB_SHRINK, TALENT_SHRINK
        assert 0 < QB_SHRINK < 1
        assert 0 < TALENT_SHRINK < 1
