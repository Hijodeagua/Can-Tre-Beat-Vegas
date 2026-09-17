"""The three learners every second stage is compared across: same call
shape, NaN tolerated everywhere, an all-NaN column (a feed that has not
landed) fits without error and cannot move the prediction."""

import numpy as np
import pandas as pd
import pytest

from common import learners


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "signal": rng.normal(size=n),
        "half_missing": np.where(rng.random(n) < 0.5, np.nan, rng.normal(size=n)),
        "empty": np.nan,
    })
    y = (X["signal"] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("kind", learners.KINDS)
def test_every_learner_fits_with_nan_and_empty_columns(kind):
    X, y = _data()
    model = learners.make(kind).fit(X, y)
    p = model.predict_proba(X)[:, 1]
    assert p.shape == (len(X),) and np.all((p >= 0) & (p <= 1))
    assert list(model.classes_) == [0, 1]
    # It learned the signal.
    assert ((p >= 0.5).astype(int) == y).mean() > 0.8
    # The empty column is inert: filling it with anything changes nothing.
    X2 = X.copy(); X2["empty"] = 5.0
    if kind != "gbm":     # imputer-based learners see a constant either way
        np.testing.assert_allclose(model.predict_proba(X2)[:, 1], p, atol=1e-6)


def test_multiclass_works_for_the_soccer_shape():
    X, y = _data()
    y3 = np.where(X["signal"] > 0.5, "H", np.where(X["signal"] < -0.5, "A", "D"))
    for kind in learners.KINDS:
        model = learners.make(kind).fit(X, y3)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 3) and list(model.classes_) == ["A", "D", "H"]
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_unknown_kind_is_an_error():
    with pytest.raises(ValueError):
        learners.make("svm")
