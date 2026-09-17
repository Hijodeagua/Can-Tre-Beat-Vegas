"""Tests for the shared pregame/freshness/evaluation layer (common/)."""

import numpy as np
import pandas as pd
import pytest

from common import evaluate, freshness, pregame


def _team_games():
    rows = []
    for t, vals in (("A", [1.0, 2.0, 3.0, 4.0]), ("B", [10.0, 20.0, 30.0])):
        for i, v in enumerate(vals):
            rows.append({"team": t, "date": f"2025-09-{1 + i:02d}", "x": v})
    return pd.DataFrame(rows)


class TestShifting:
    def test_rolling_excludes_the_current_game(self):
        df = pregame.shifted_rolling(_team_games(), "team", ["x"], window=2,
                                     order=["team", "date"])
        a = df[df.team == "A"]["x_r2"].tolist()
        assert np.isnan(a[0])           # nothing before the first game
        assert a[1] == 1.0              # only game 1
        assert a[2] == pytest.approx(1.5)   # games 1-2, not 3
        assert a[3] == pytest.approx(2.5)   # games 2-3, not 4

    def test_ewm_excludes_the_current_game(self):
        df = pregame.shifted_ewm(_team_games(), "team", ["x"], half_life=1.0,
                                 order=["team", "date"])
        a = df[df.team == "A"]["x_ewm"].tolist()
        assert np.isnan(a[0]) and a[1] == 1.0
        # After games 1 and 2 with half-life 1: weights 0.5, 1 -> (0.5*1 + 2) / 1.5
        assert a[2] == pytest.approx((0.5 * 1.0 + 2.0) / 1.5)

    def test_leak_assertion_catches_an_unshifted_feature(self):
        df = _team_games().sort_values(["team", "date"])
        df["bad"] = df.groupby("team")["x"].transform(lambda s: s.rolling(2, min_periods=1).mean())
        with pytest.raises(AssertionError):
            pregame.assert_no_same_game_leak(df, "team", ["team", "date"], "x", "bad")
        good = pregame.shifted_rolling(df, "team", ["x"], 2, order=["team", "date"])
        pregame.assert_no_same_game_leak(good, "team", ["team", "date"], "x", "x_r2")

    def test_rolling_is_deterministic(self):
        a = pregame.shifted_rolling(_team_games(), "team", ["x"], 3, order=["team", "date"])
        b = pregame.shifted_rolling(_team_games(), "team", ["x"], 3, order=["team", "date"])
        pd.testing.assert_frame_equal(a, b)


class TestRestAndCongestion:
    def test_days_since_previous(self):
        df = pd.DataFrame({"team": ["A", "A", "A"],
                           "date": ["2025-09-01", "2025-09-04", "2025-09-11"]})
        rest = pregame.days_since_previous(df, "team", "date")
        assert rest.tolist()[0] != rest.tolist()[0]   # NaN first
        assert rest.tolist()[1:] == [3, 7]

    def test_count_in_previous_days(self):
        df = pd.DataFrame({"team": ["A"] * 4,
                           "date": ["2025-09-01", "2025-09-04", "2025-09-08", "2025-09-20"]})
        n = pregame.count_in_previous_days(df, "team", "date", days=14)
        assert n.tolist() == [0, 1, 2, 1]   # the last sees only 09-08

    def test_shrink_moves_small_samples_to_the_prior(self):
        v = pd.Series([1.0, 1.0, np.nan])
        n = pd.Series([1, 20, 0])
        out = pregame.shrink(v, n, prior=0.0, prior_games=4)
        assert out.iloc[0] == pytest.approx(0.2)
        assert out.iloc[1] == pytest.approx(20 / 24)
        assert out.iloc[2] == 0.0


class TestFreshness:
    def test_stale_and_fresh(self):
        df = pd.DataFrame({"date": ["2025-01-01", "2025-01-04"]})
        r = freshness.check("feed", df, "date", "2025-01-10", tolerance_days=7)
        assert r.fresh and r.age_days == 6 and r.rows == 2
        r = freshness.check("feed", df, "date", "2025-06-01", tolerance_days=7)
        assert not r.fresh
        assert not freshness.check("feed", df.iloc[0:0], "date", "2025-06-01", 7).fresh


class TestEvaluate:
    def test_walk_forward_is_out_of_sample_and_paired(self):
        rng = np.random.default_rng(0)
        n = 1200
        f = pd.DataFrame({
            "season": np.repeat(np.arange(2015, 2021), n // 6),
            "x": rng.normal(size=n),
        })
        f["y"] = (f["x"] + rng.normal(scale=1.5, size=n) > 0).astype(int)
        f["noise"] = rng.normal(size=n)
        base, per_season = evaluate.walk_forward(f, ["x"], "y", "season", [2018, 2019, 2020])
        cand, _ = evaluate.walk_forward(f, ["x", "noise"], "y", "season", [2018, 2019, 2020])
        assert base.n == 600 and len(per_season) == 3
        assert 0 < base.log_loss < 0.693
        # A pure-noise feature should not beat the baseline by anything
        # a paired test would call real.
        assert evaluate.paired_se(base, cand) < 2.0

    def test_multiclass_scores(self):
        y = np.array(["H", "A", "D"])
        p = np.array([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.2, 0.6, 0.2]])
        s = evaluate.score_multiclass(y, p, ["A", "D", "H"])
        assert s.accuracy == 1.0 and s.n == 3
        assert s.log_loss == pytest.approx(-np.mean(np.log([0.8, 0.8, 0.6])))
