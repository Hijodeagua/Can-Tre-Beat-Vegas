"""Tests for the early-season ATS analysis.

The claim this module makes is a betting claim, so the tests check that the
statistics would actually *reject* a null world — the failure mode that matters
is a test suite that passes on noise.
"""

import numpy as np
import pandas as pd
import pytest

from NFL.model.v2.early_season import (
    BREAK_EVEN,
    block_bootstrap,
    by_week,
    cutoff_sensitivity,
    prepare,
    roi_at_110,
    season_level,
)


def _synthetic(dog_rate_early=0.60, dog_rate_late=0.50, seasons=16, seed=0):
    """Games where the early-season dog cover rate is set by construction."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(2010, 2010 + seasons):
        for w in range(1, 19):
            p = dog_rate_early if w <= 4 else dog_rate_late
            for _ in range(16):
                dog_wins = rng.random() < p
                # home is the favourite; margin decides the cover
                rows.append({"season": s, "week": w, "game_type": "REG",
                             "spread_line": 3.0,
                             "ats_margin_home": -1.0 if dog_wins else 1.0,
                             "prob": 0.5})
    return pd.DataFrame(rows)


def _prep(df):
    fav_cov = np.where(df["spread_line"] > 0,
                       df["ats_margin_home"] > 0, df["ats_margin_home"] < 0)
    df = df.copy()
    df["dog_cov"] = ~fav_cov
    df["blk"] = df["season"].astype(str) + "_" + df["week"].astype(str)
    return df


class TestRoi:
    def test_break_even_returns_zero(self):
        assert roi_at_110(BREAK_EVEN) == pytest.approx(0.0, abs=1e-9)

    def test_known_values(self):
        assert roi_at_110(1.0) == pytest.approx(100 / 110)
        assert roi_at_110(0.0) == pytest.approx(-1.0)
        assert roi_at_110(0.55) == pytest.approx(0.55 * (100/110) - 0.45)


class TestBlockBootstrap:
    def test_detects_a_real_edge(self):
        d = _prep(_synthetic(dog_rate_early=0.60))
        lo, hi, p = block_bootstrap(d[d.week <= 4], reps=2000)
        assert lo > BREAK_EVEN
        assert p < 0.05

    def test_does_not_fire_on_a_fair_market(self):
        """The critical test: a 50% world must not look profitable."""
        d = _prep(_synthetic(dog_rate_early=0.50, dog_rate_late=0.50, seed=3))
        lo, hi, p = block_bootstrap(d[d.week <= 4], reps=2000)
        assert lo < BREAK_EVEN
        assert p > 0.20

    def test_interval_brackets_the_truth(self):
        d = _prep(_synthetic(dog_rate_early=0.58, seed=11))
        lo, hi, _ = block_bootstrap(d[d.week <= 4], reps=2000)
        assert lo <= 0.58 <= hi


class TestSeasonLevel:
    def test_consistent_edge_clears_the_bar(self):
        d = _prep(_synthetic(dog_rate_early=0.62, seed=5))
        s = season_level(d)
        assert s["t_stat"] > 1.753
        assert s["profitable_at_5pct"]

    def test_fair_market_does_not_clear_the_bar(self):
        """50% covers loses 4.5% a year to the vig. That is not an edge."""
        d = _prep(_synthetic(dog_rate_early=0.50, seed=9))
        s = season_level(d)
        assert not s["profitable_at_5pct"]
        assert s["mean_roi"] < 0

    def test_a_consistent_loser_is_not_called_significant(self):
        """The one-sided test must not dress up a reliable loss as a finding."""
        d = _prep(_synthetic(dog_rate_early=0.40, seed=4))
        s = season_level(d)
        assert s["t_stat"] < -2
        assert not s["profitable_at_5pct"]

    def test_counts_seasons_not_games(self):
        d = _prep(_synthetic(seasons=12))
        assert season_level(d)["seasons"] == 12

    def test_worst_season_is_reported(self):
        d = _prep(_synthetic(dog_rate_early=0.55, seed=2))
        s = season_level(d)
        assert s["worst_roi"] <= s["median_roi"]
        assert 2010 <= s["worst_season"] <= 2025


class TestShape:
    def test_by_week_covers_requested_range(self):
        t = by_week(_prep(_synthetic()), upto=6)
        assert list(t["week"]) == [1, 2, 3, 4, 5, 6]
        assert (t["games"] > 0).all()

    def test_cutoff_sensitivity_is_cumulative(self):
        t = cutoff_sensitivity(_prep(_synthetic(seasons=4)), upto=4)
        assert t["games"].is_monotonic_increasing

    def test_real_artifacts_load_if_present(self):
        from NFL.model.v2.early_season import ATS_PREDS
        if not ATS_PREDS.exists():
            pytest.skip("backtest artifacts not generated")
        d = prepare()
        assert {"dog_cov", "blk", "week", "season"} <= set(d.columns)
        assert d["dog_cov"].dtype == bool
