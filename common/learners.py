"""The learners every second stage is compared across, built the same way
for every sport so a head-to-head is a head-to-head.

Three kinds:

- `logistic`      — median-impute, standardise, L2 logistic. Linear in the
                    features, so it cannot see a threshold: the gap from
                    1600 to 1700 Elo is worth exactly what 1300 to 1400 is.
- `random_forest` — median-impute, then a forest of shallow-ish trees with
                    a large leaf minimum. Sees interactions and thresholds;
                    probabilities are leaf frequencies, so they are coarser
                    than a logistic's.
- `gbm`           — histogram gradient boosting (sklearn's, so no extra
                    dependency and NaN is handled natively). Also sees
                    thresholds and interactions; usually the best-calibrated
                    of the three on tabular data at this size.

Every hyperparameter is fixed here rather than tuned per sport, on
purpose: the comparison is between model families on the same features
and the same chronological split, not between tuning budgets. Regularise
by minimum leaf size, not depth, so a forest on 25k soccer rows and a
forest on 6k NFL rows are both allowed to grow until a leaf would be
thin. Change these constants in one place if the whole board should.

`make(kind)` returns an unfitted sklearn estimator with `fit` /
`predict_proba` / `classes_`; every evaluator and every daily state builds
its model through it, so what is measured is what ships.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FillEmpty(BaseEstimator, TransformerMixin):
    """Zero-fill columns that were entirely NaN at fit time (a feed that
    has not landed yet), leaving every other NaN in place for the
    boosting model's native handling. Histogram binning cannot bin a
    column with no values at all."""

    def fit(self, X, y=None):
        arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        self.empty_ = np.isnan(arr).all(axis=0)
        return self

    def transform(self, X):
        arr = (X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)).copy()
        arr[:, self.empty_] = np.where(np.isnan(arr[:, self.empty_]), 0.0, arr[:, self.empty_])
        return arr

KINDS = ("logistic", "random_forest", "gbm")
SEED = 0

RF_TREES = 500
RF_MIN_LEAF = 25
GBM_RATE = 0.03
GBM_ROUNDS = 400
GBM_LEAVES = 15
GBM_MIN_LEAF = 40
GBM_L2 = 1.0


def make(kind: str, C: float = 1.0, **overrides):
    """An unfitted estimator of the given kind. `C` is the logistic's
    inverse regularisation; the tree learners ignore it."""
    if kind == "logistic":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=5000)),
        ])
    if kind == "random_forest":
        params = dict(n_estimators=RF_TREES, min_samples_leaf=RF_MIN_LEAF,
                      max_features="sqrt", n_jobs=-1, random_state=SEED)
        params.update(overrides)
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("clf", RandomForestClassifier(**params)),
        ])
    if kind == "gbm":
        params = dict(learning_rate=GBM_RATE, max_iter=GBM_ROUNDS, max_leaf_nodes=GBM_LEAVES,
                      min_samples_leaf=GBM_MIN_LEAF, l2_regularization=GBM_L2,
                      early_stopping=False, random_state=SEED)
        params.update(overrides)
        # NaN goes straight in: histogram boosting learns a direction for
        # missing values per split, which is the right treatment for "no
        # feed yet" — better than pretending it is the median. Only a
        # column with no values at all is zero-filled, because it cannot
        # be binned.
        return Pipeline([
            ("fill_empty", FillEmpty()),
            ("clf", HistGradientBoostingClassifier(**params)),
        ])
    raise ValueError(f"unknown learner {kind!r}; one of {KINDS}")


def describe(kind: str) -> str:
    return {
        "logistic": "L2 logistic on median-imputed, standardised features",
        "random_forest": f"random forest, {RF_TREES} trees, min leaf {RF_MIN_LEAF}",
        "gbm": f"gradient boosting, {GBM_ROUNDS} rounds at {GBM_RATE}, {GBM_LEAVES} leaves, min leaf {GBM_MIN_LEAF}",
    }[kind]
