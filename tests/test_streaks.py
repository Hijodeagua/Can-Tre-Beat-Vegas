"""Tests for the streak / form analysis.

The module's job is to *not* find things that aren't there, so most of these
check that a null world produces a null answer.
"""

import numpy as np
import pandas as pd
import pytest

from NFL.model.v2.streaks import (
    BREAK_EVEN,
    bh_fdr,
    block_bootstrap,
    evaluate,
    season_t,
    team_games,
)


def _fake(n_seasons=10, cover_rate=0.5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(2010, 2010 + n_seasons):
        for w in range(1, 18):
            for i in range(16):
                rows.append({"season": s, "week": w,
                             "blk": f"{s}_{w}",
                             "covered": rng.random() < cover_rate})
    return pd.DataFrame(rows)


class TestBhFdr:
    def test_all_null_survives_nothing(self):
        assert not any(bh_fdr([0.4, 0.5, 0.6, 0.9, 0.95]))

    def test_one_strong_signal_survives(self):
        assert bh_fdr([1e-8, 0.4, 0.5, 0.6, 0.9])[0]

    def test_is_less_conservative_than_bonferroni(self):
        """The whole point of BH: it should keep things Bonferroni discards."""
        p = [0.001, 0.008, 0.02, 0.6, 0.7]
        bonf = [x < 0.05 / len(p) for x in p]
        assert sum(bh_fdr(p)) > sum(bonf)

    def test_borderline_case_at_the_boundary(self):
        # p = alpha * rank / m exactly: BH uses <=, so it survives.
        assert bh_fdr([0.05 * 1 / 2, 0.9])[0]

    def test_handles_nan(self):
        out = bh_fdr([1e-9, float("nan"), 0.8])
        assert out[0] and not out[1] and not out[2]


class TestEvaluate:
    def test_fair_market_looks_fair(self):
        r = evaluate(_fake(cover_rate=0.5, seed=1), "null")
        assert r["p_raw"] > 0.05
        assert r["ci_lo"] < 0.5 < r["ci_hi"]

    def test_real_edge_is_detected(self):
        r = evaluate(_fake(cover_rate=0.60, seed=2), "edge")
        assert r["p_raw"] < 0.001
        assert r["ci_lo"] > 0.5

    def test_small_samples_are_refused_not_guessed(self):
        r = evaluate(_fake(n_seasons=1, seed=3).head(20), "tiny")
        assert np.isnan(r["cover"])
        assert r["note"] == "too few games"

    def test_roi_matches_the_cover_rate(self):
        # Both fields are rounded to 4dp independently from the unrounded rate,
        # so reconstructing ROI from the rounded cover carries ~2e-4 of slop.
        r = evaluate(_fake(cover_rate=0.55, seed=4), "x")
        assert r["roi_at_110"] == pytest.approx(
            r["cover"] * (100/110) - (1 - r["cover"]), abs=5e-4)

    def test_break_even_cover_gives_zero_roi(self):
        d = _fake(seed=5)
        n = len(d)
        d["covered"] = [True] * round(BREAK_EVEN * n) + [False] * (n - round(BREAK_EVEN * n))
        assert evaluate(d, "be")["roi_at_110"] == pytest.approx(0.0, abs=1e-3)


class TestSeasonT:
    def test_consistent_edge_gives_large_t(self):
        t, ns = season_t(_fake(cover_rate=0.60, seed=6))
        assert t > 3 and ns == 10

    def test_fair_market_gives_small_t(self):
        t, _ = season_t(_fake(cover_rate=0.50, seed=7))
        assert abs(t) < 2

    def test_too_few_seasons_returns_nan(self):
        t, _ = season_t(_fake(n_seasons=2, seed=8))
        assert np.isnan(t)


class TestTeamGames:
    """The leakage-critical part: lags must not see their own game."""

    def test_lags_are_strictly_prior(self):
        t = team_games(2015)
        one = t[(t.team == t.team.iloc[0]) & (t.season == 2015)].sort_values("week")
        if len(one) < 4:
            pytest.skip("not enough games")
        assert one["pf_lag1"].iloc[1:].to_numpy() == pytest.approx(
            one["pf"].iloc[:-1].to_numpy())

    def test_first_game_of_a_season_has_no_history(self):
        t = team_games(2015)
        firsts = t.sort_values("week").groupby(["team", "season"]).head(1)
        assert firsts["pf_lag1"].isna().all()
        assert firsts["pf_roll3"].isna().all()

    def test_pushes_are_dropped(self):
        t = team_games(2015)
        assert (t["ats_margin"] != 0).all()

    def test_cover_rate_is_near_half_by_construction(self):
        t = team_games(2010)
        assert t["covered"].mean() == pytest.approx(0.5, abs=1e-9)

    def test_spread_is_team_oriented(self):
        t = team_games(2015)
        g = t.groupby("game_id")["team_spread"].sum()
        assert g.abs().max() < 1e-9   # the two sides must cancel


class TestTeamAliases:
    """The join that silently dropped 5% of team-games, pinned."""

    def test_relocated_franchises_collapse_to_one_code(self):
        from NFL.model.v2.streaks import normalize_team
        out = normalize_team(pd.Series(["OAK", "LV", "SD", "LAC",
                                        "STL", "LAR", "LA", "KC"]))
        assert list(out) == ["LV", "LV", "LAC", "LAC", "LA", "LA", "LA", "KC"]

    def test_no_legacy_codes_survive_in_the_spine(self):
        from NFL.model.v2.streaks import team_games
        t = team_games(2011)
        assert not ({"OAK", "SD", "STL", "LAR"} & set(t["team"].unique()))
        assert not ({"OAK", "SD", "STL", "LAR"} & set(t["opp"].unique()))

    def test_qb_stats_join_covers_nearly_every_team_game(self):
        """A silent join failure shows up here, not in any downstream p-value."""
        from NFL.model.v2.streaks import qb_weeks, team_games
        t, q = team_games(2011), qb_weeks(2011)
        if q.empty:
            pytest.skip("player stats not downloaded")
        merged = t.merge(q[["season", "week", "team"]].drop_duplicates(),
                         on=["season", "week", "team"], how="inner")
        assert len(merged) / len(t) > 0.97

    def test_every_stats_team_exists_in_the_schedule(self):
        from NFL.model.v2.streaks import qb_weeks, team_games
        t, q = team_games(2011), qb_weeks(2011)
        if q.empty:
            pytest.skip("player stats not downloaded")
        assert not (set(q["team"].unique()) - set(t["team"].unique()))


class TestTop100Flags:
    def test_flags_map_to_nflverse_ids(self):
        from NFL.model.v2.streaks import top100_qb_flags
        f = top100_qb_flags()
        if f.empty:
            pytest.skip("awards data not present")
        assert {"season", "rank", "player_id"} <= set(f.columns)
        assert f["season"].min() >= 2011
        assert f["rank"].between(1, 100).all()

    def test_one_row_per_player_season(self):
        from NFL.model.v2.streaks import top100_qb_flags
        f = top100_qb_flags()
        if f.empty:
            pytest.skip("awards data not present")
        assert not f.duplicated(["season", "player_id"]).any()
