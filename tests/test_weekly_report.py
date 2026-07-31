"""Tests for the weekly report framework: recency weights, model registry,
book grading math, prediction-market parsers, and the pick ledger."""

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from NFL.model.v2.compare_models import (
    MODEL_KINDS,
    make_model,
    recency_weights,
    top_k_weekly,
)
from NFL.inventory.book_performance import leaderboard
from data_jobs.prediction_markets.fetch import _mid, fetch_kalshi, fetch_polymarket


# --------------------------------------------------------------------------
# recency weighting
# --------------------------------------------------------------------------

class TestRecencyWeights:
    def test_half_life_halves_weight(self):
        seasons = pd.Series([2024, 2021, 2018])
        w = recency_weights(seasons, test_season=2024, half_life=3.0)
        assert w[0] == pytest.approx(1.0)
        assert w[1] == pytest.approx(0.5)
        assert w[2] == pytest.approx(0.25)

    def test_zero_half_life_disables(self):
        w = recency_weights(pd.Series([2002, 2024]), 2025, half_life=0)
        assert (w == 1.0).all()

    def test_monotone_in_age(self):
        seasons = pd.Series(range(2002, 2025))
        w = recency_weights(seasons, 2025, half_life=6.0)
        assert (np.diff(w) > 0).all()


# --------------------------------------------------------------------------
# model registry
# --------------------------------------------------------------------------

class TestModelRegistry:
    @pytest.mark.parametrize("kind", MODEL_KINDS)
    def test_every_kind_fits_and_predicts_proba(self, kind):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(300, 5)), columns=list("abcde"))
        y = (X["a"] + rng.normal(scale=0.5, size=300) > 0).astype(int)
        m = make_model(kind)
        kw = {"clf__sample_weight" if hasattr(m, "steps") else "sample_weight":
              np.ones(len(y))}
        m.fit(X, y, **kw)
        p = m.predict_proba(X)[:, 1]
        assert p.shape == (300,)
        assert ((p >= 0) & (p <= 1)).all()
        # Signal is strong; any sane model should beat chance in-sample.
        assert ((p >= 0.5).astype(int) == y).mean() > 0.6

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            make_model("neural_vibes")

    def test_models_tolerate_nans(self):
        """Real features have missing temp/wind; imputation must handle it."""
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
        X.loc[::7, "b"] = np.nan
        y = (X["a"] > 0).astype(int)
        for kind in ["logistic", "rf", "xgb", "lgbm"]:
            m = make_model(kind)
            kw = {"clf__sample_weight" if hasattr(m, "steps") else "sample_weight":
                  np.ones(len(y))}
            m.fit(X, y, **kw)
            assert not np.isnan(m.predict_proba(X)[:, 1]).any()


# --------------------------------------------------------------------------
# top-k product metric
# --------------------------------------------------------------------------

class TestTopK:
    def _preds(self):
        return pd.DataFrame({
            "season": [2025] * 6, "week": [1, 1, 1, 1, 2, 2],
            "prob": [0.9, 0.8, 0.7, 0.51, 0.95, 0.2],
            "y":    [1,   1,   0,   0,    1,    0],
        })

    def test_takes_k_most_confident_per_week(self):
        out = top_k_weekly(self._preds(), k=3)
        # Week 1 top3 = 0.9(hit) 0.8(hit) 0.7(miss); week 2 has only 2 games,
        # both count: 0.95(hit), 0.2->pick away, y=0 -> hit.
        assert out["picks"] == 5
        assert out["hits"] == 4

    def test_confidence_is_distance_from_half(self):
        # prob 0.2 is more confident than 0.51 and must be selected first.
        p = self._preds()
        out = top_k_weekly(p, k=1)
        assert out["picks"] == 2  # one per week
        assert out["hits"] == 2   # 0.9 hit + 0.95 hit


# --------------------------------------------------------------------------
# bookmaker leaderboard
# --------------------------------------------------------------------------

class TestBookLeaderboard:
    def test_sharper_book_ranks_first(self):
        n = 40
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, n)
        sharp = np.clip(np.where(y == 1, 0.75, 0.25) + rng.normal(0, 0.03, n), 0.02, 0.98)
        square = np.clip(0.5 + rng.normal(0, 0.05, n), 0.02, 0.98)
        graded = pd.DataFrame({
            "book": ["Sharp"] * n + ["Square"] * n,
            "novig_home": np.concatenate([sharp, square]),
            "home_won": np.concatenate([y, y]),
            "hold": 0.04,
        })
        lb = leaderboard(graded, min_games=10)
        assert lb.iloc[0]["book"] == "Sharp"
        assert lb.iloc[0]["brier"] < lb.iloc[1]["brier"]

    def test_small_samples_excluded(self):
        graded = pd.DataFrame({
            "book": ["Tiny"] * 3,
            "novig_home": [0.6, 0.7, 0.4],
            "home_won": [1, 1, 0],
            "hold": 0.04,
        })
        assert leaderboard(graded, min_games=20).empty


# --------------------------------------------------------------------------
# prediction-market parsers (mocked HTTP)
# --------------------------------------------------------------------------

def _resp(payload):
    r = MagicMock()
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


class TestPredictionMarkets:
    def test_mid_price(self):
        assert _mid(45, 47) == pytest.approx(0.46)
        assert _mid(None, 50) is None
        assert _mid(0, 0) is None

    def test_kalshi_parser_pairs_sides(self):
        session = MagicMock()
        session.get.return_value = _resp({"markets": [
            {"event_ticker": "KXNFLGAME-25SEP04DALPHI", "ticker": "KXNFLGAME-25SEP04DALPHI-PHI",
             "yes_bid": 60, "yes_ask": 64, "volume": 1000, "liquidity": 500,
             "expected_expiration_time": "2025-09-05T03:00:00Z"},
            {"event_ticker": "KXNFLGAME-25SEP04DALPHI", "ticker": "KXNFLGAME-25SEP04DALPHI-DAL",
             "yes_bid": 36, "yes_ask": 40, "volume": 800, "liquidity": 400,
             "expected_expiration_time": "2025-09-05T03:00:00Z"},
        ], "cursor": None})
        df = fetch_kalshi(session)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["home_team"] == "PHI" and row["away_team"] == "DAL"
        assert row["home_prob"] == pytest.approx(0.62)
        assert row["away_prob"] == pytest.approx(0.38)
        assert row["volume"] == 1800

    def test_polymarket_parser_handles_json_strings(self):
        session = MagicMock()
        session.get.return_value = _resp([
            {"slug": "chiefs-vs-chargers", "question": "Chiefs vs. Chargers",
             "outcomes": json.dumps(["Chiefs", "Chargers"]),
             "outcomePrices": json.dumps(["0.41", "0.59"]),
             "gameStartTime": "2025-09-05T00:15:00Z",
             "volumeNum": 250000, "liquidityNum": 40000},
        ])
        df = fetch_polymarket(session)
        assert len(df) == 1
        assert df.iloc[0]["home_prob"] == pytest.approx(0.59)
        assert df.iloc[0]["away_team"] == "Chiefs"

    def test_polymarket_skips_non_binary_markets(self):
        session = MagicMock()
        session.get.return_value = _resp([
            {"slug": "mvp-field", "outcomes": json.dumps(["A", "B", "C"]),
             "outcomePrices": json.dumps(["0.2", "0.3", "0.5"])},
        ])
        assert fetch_polymarket(session).empty


# --------------------------------------------------------------------------
# pick ledger
# --------------------------------------------------------------------------

class TestPickLedger:
    def test_log_and_grade_round_trip(self, tmp_path, monkeypatch):
        import data_jobs.reports.weekly_nfl_report as wr
        monkeypatch.setattr(wr, "OUT_DIR", tmp_path)
        monkeypatch.setattr(wr, "PICKS_LOG", tmp_path / "picks_log.csv")

        scored = pd.DataFrame({
            "game_id": ["g1", "g2"],
            "gameday": pd.to_datetime(["2025-11-23", "2025-11-23"]),
            "pick_team": ["BAL", "DET"], "opp_team": ["NYJ", "NYG"],
            "confidence": [0.93, 0.90], "market_conf": [0.88, 0.88],
        })
        assert wr.log_picks(scored, 2025, 12, top_k=2) == 2
        # Re-logging the same games is a no-op (Friday rerun).
        assert wr.log_picks(scored, 2025, 12, top_k=2) == 0

        results = pd.DataFrame({
            "game_id": ["g1", "g2"],
            "home_team": ["BAL", "NYG"], "away_team": ["NYJ", "DET"],
            "home_score": [30.0, 10.0], "away_score": [10.0, 27.0],
        })
        graded = wr.grade_ledger(results)
        assert list(graded["result"]) == ["W", "W"]  # BAL won at home, DET on road

    def test_pending_games_stay_pending(self, tmp_path, monkeypatch):
        import data_jobs.reports.weekly_nfl_report as wr
        monkeypatch.setattr(wr, "OUT_DIR", tmp_path)
        monkeypatch.setattr(wr, "PICKS_LOG", tmp_path / "picks_log.csv")
        scored = pd.DataFrame({
            "game_id": ["g9"], "gameday": pd.to_datetime(["2026-09-13"]),
            "pick_team": ["KC"], "opp_team": ["DEN"],
            "confidence": [0.6], "market_conf": [0.58],
        })
        wr.log_picks(scored, 2026, 1, top_k=1)
        results = pd.DataFrame({
            "game_id": ["g9"], "home_team": ["KC"], "away_team": ["DEN"],
            "home_score": [np.nan], "away_score": [np.nan],
        })
        assert wr.grade_ledger(results)["result"].iloc[0] == "pending"
