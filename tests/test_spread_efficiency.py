"""Tests for the spread efficiency suite.

The statistics here decide whether a bet is worth making, so each estimator is
checked against a case with a known answer rather than only for not crashing.
"""

import numpy as np
import pandas as pd
import pytest

from NFL.model.v2.spread_efficiency import (
    BREAK_EVEN,
    ClusterOLS,
    diebold_mariano,
    encompassing,
    implied_vs_actual,
    market_residual,
    mincer_zarnowitz,
    required_disagreement,
)


def _frame(n=600, noise=10.0, seed=0, ours_is=None):
    """A synthetic season set where the truth is known by construction."""
    rng = np.random.default_rng(seed)
    vegas = rng.normal(0, 5, n)
    margin = vegas + rng.normal(0, noise, n)
    ours = vegas.copy() if ours_is is None else ours_is(vegas, rng)
    return pd.DataFrame({
        "season": np.repeat(np.arange(2015, 2015 + n // 60), 60)[:n],
        "week": np.tile(np.arange(1, 61), n // 60 + 1)[:n],
        "margin": margin, "spread_line": vegas, "pred_margin": ours,
    })


class TestClusterOLS:
    def test_recovers_known_coefficients(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=2000)
        y = 3.0 + 2.5 * x + rng.normal(0, 0.5, 2000)
        m = ClusterOLS(y, np.column_stack([np.ones(2000), x]),
                       np.arange(2000) // 10, ["a", "b"])
        assert m.beta[0] == pytest.approx(3.0, abs=0.05)
        assert m.beta[1] == pytest.approx(2.5, abs=0.05)

    def test_clustering_widens_errors_when_shocks_are_shared(self):
        """Correlated residuals within a cluster must inflate the SE."""
        rng = np.random.default_rng(2)
        n, g = 2000, 100
        cl = np.repeat(np.arange(g), n // g)
        x = rng.normal(size=n)
        y = x + np.repeat(rng.normal(0, 3, g), n // g)   # shared shock per cluster
        X = np.column_stack([np.ones(n), x])
        clustered = ClusterOLS(y, X, cl, ["a", "b"]).se[0]
        naive = ClusterOLS(y, X, np.arange(n), ["a", "b"]).se[0]
        assert clustered > naive * 2

    def test_wald_rejects_a_false_joint_hypothesis(self):
        d = _frame(n=1200, ours_is=lambda v, r: 0.5 * v)
        m = mincer_zarnowitz(d, "pred_margin", "shrunk")
        assert m["b"] > 1.5          # a halved forecast needs doubling
        assert m["p_joint"] < 0.01
        assert m["efficient"] == "no"


class TestMincerZarnowitz:
    def test_a_perfect_forecast_passes(self):
        m = mincer_zarnowitz(_frame(), "spread_line", "vegas")
        assert m["b"] == pytest.approx(1.0, abs=0.15)
        assert m["efficient"] == "yes"

    def test_a_biased_forecast_is_caught(self):
        d = _frame()
        d["pred_margin"] = d["spread_line"] + 6.0    # 6 points too generous to home
        m = mincer_zarnowitz(d, "pred_margin", "biased")
        assert m["p_joint"] < 0.01
        assert m["efficient"] == "no"


class TestEncompassing:
    def test_a_redundant_forecast_gets_zero_weight(self):
        """Ours = vegas plus pure noise carries no independent information."""
        d = _frame(ours_is=lambda v, r: v + r.normal(0, 2, len(v)))
        enc, _ = encompassing(d)
        assert enc["p_ours"] > 0.05
        assert enc["b_vegas"] > 0.7

    def test_a_genuinely_informative_forecast_gets_weight(self):
        rng = np.random.default_rng(5)
        n = 1500
        vegas = rng.normal(0, 5, n)
        secret = rng.normal(0, 5, n)              # signal the market cannot see
        margin = vegas + secret + rng.normal(0, 6, n)
        d = pd.DataFrame({"season": np.repeat(np.arange(2015, 2040), 60)[:n],
                          "week": np.tile(np.arange(1, 61), 26)[:n],
                          "margin": margin, "spread_line": vegas,
                          "pred_margin": vegas + secret})
        enc, _ = encompassing(d)
        assert enc["p_ours"] < 0.01
        assert enc["b_ours"] > 0.5


class TestMarketResidual:
    def test_efficient_market_gives_zero_slope(self):
        # n large enough that the slope's SE (~sigma / (sd_disagree * sqrt(n)))
        # is small, so a 0.15 tolerance is a real check and not a coin flip.
        d = _frame(n=3000, ours_is=lambda v, r: v + r.normal(0, 4, len(v)))
        mr = market_residual(d)
        assert abs(mr["b"]) < 0.15
        assert mr["p"] > 0.05

    def test_exploitable_market_gives_positive_slope(self):
        rng = np.random.default_rng(7)
        n = 1500
        vegas = rng.normal(0, 5, n)
        secret = rng.normal(0, 4, n)
        margin = vegas + secret + rng.normal(0, 6, n)
        d = pd.DataFrame({"season": np.repeat(np.arange(2015, 2040), 60)[:n],
                          "week": np.tile(np.arange(1, 61), 26)[:n],
                          "margin": margin, "spread_line": vegas,
                          "pred_margin": vegas + secret})
        mr = market_residual(d)
        assert mr["b"] > 0.7 and mr["p"] < 0.01
        assert mr["ats_disagree_1pt"] > BREAK_EVEN


class TestDieboldMariano:
    def test_detects_the_better_forecast(self):
        d = _frame(noise=6.0, ours_is=lambda v, r: v + r.normal(0, 8, len(v)))
        dm = diebold_mariano(d)
        assert dm["dm_stat"] > 0          # positive favours vegas
        assert dm["verdict"] == "vegas better"

    def test_identical_forecasts_are_indistinguishable(self):
        d = _frame()
        assert diebold_mariano(d)["verdict"] == "indistinguishable"


class TestRequiredDisagreement:
    def test_bigger_edge_needs_less_disagreement(self):
        d = _frame()
        assert (required_disagreement(d, 0.5)["disagreement_needed"] <
                required_disagreement(d, 0.1)["disagreement_needed"])

    def test_no_edge_is_never_worth_betting(self):
        assert required_disagreement(_frame(), 0.0)["disagreement_needed"] == float("inf")

    def test_implied_cover_rises_with_disagreement(self):
        d = _frame(ours_is=lambda v, r: v + r.normal(0, 5, len(v)))
        iva = implied_vs_actual(d, 0.3)
        assert iva["implied_cover"].is_monotonic_increasing
        assert (iva["implied_cover"] >= 0.5).all()
