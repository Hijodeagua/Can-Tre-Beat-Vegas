"""Chronological evaluation for the second-stage models.

Two fixed rules:

- Splits are by time, never random. `walk_forward` retrains once per test
  season on everything before it; `fixed_split` fits once on the
  training seasons and scores the rest. Both report every prediction
  out of sample.
- Log loss and Brier are the metrics. Accuracy is reported because people
  ask, not because it decides anything.

Paired comparison: two candidate feature sets score the *same* rows, so
the honest test is on the per-row difference in log loss. `paired_se`
turns a candidate's gain over a baseline into standard errors — the
number a feature group has to clear.

The model is a regularised logistic regression on a standardised design
matrix with median imputation, all three fit on training rows only
(`Pipeline` guarantees that). A tree model is compared where the repo
already ships one (LightGBM in NFL), never as the production default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic(C: float = 1.0, max_iter: int = 5000) -> Pipeline:
    """Impute (train medians) -> scale (train stats) -> L2 logistic."""
    return Pipeline([
        # keep_empty_features: an all-NaN column (a feed that hasn't landed
        # yet) becomes a constant instead of silently vanishing, so the
        # design matrix keeps its shape and the scaler zeroes it out.
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=max_iter)),
    ])


@dataclass
class Scores:
    log_loss: float
    brier: float
    accuracy: float
    n: int
    per_row_loss: np.ndarray = field(repr=False, default=None)

    def row(self, label: str) -> dict:
        return {"model": label, "log_loss": round(self.log_loss, 5),
                "brier": round(self.brier, 5), "accuracy": round(self.accuracy, 4),
                "n": self.n}


def score_binary(y: np.ndarray, p: np.ndarray) -> Scores:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    per_row = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return Scores(
        log_loss=float(per_row.mean()),
        brier=float(brier_score_loss(y, p)),
        accuracy=float(accuracy_score(y, p >= 0.5)),
        n=int(len(y)),
        per_row_loss=per_row,
    )


def score_multiclass(y: np.ndarray, probs: np.ndarray, classes: list[str]) -> Scores:
    """Multinomial log loss, multiclass Brier (sum over classes of squared
    error, the standard definition), top-class accuracy."""
    idx = np.array([classes.index(v) for v in y])
    p = np.clip(probs, 1e-6, 1)
    per_row = -np.log(p[np.arange(len(y)), idx])
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), idx] = 1.0
    return Scores(
        log_loss=float(per_row.mean()),
        brier=float(((p - onehot) ** 2).sum(axis=1).mean()),
        accuracy=float((p.argmax(axis=1) == idx).mean()),
        n=int(len(y)),
        per_row_loss=per_row,
    )


def paired_se(baseline: Scores, candidate: Scores) -> float:
    """Candidate's mean gain in log loss over the baseline, in standard
    errors of the per-row differences. Positive = candidate better."""
    gain = baseline.per_row_loss - candidate.per_row_loss
    se = gain.std(ddof=1) / np.sqrt(len(gain))
    return float(gain.mean() / se) if se > 0 else float("nan")


def walk_forward(frame: pd.DataFrame, features: list[str], target: str,
                 season_col: str, test_seasons: list[int | str],
                 C: float = 1.0, min_train_seasons: int = 3,
                 fit=None) -> tuple[Scores, pd.DataFrame]:
    """Retrain once per test season on every earlier season; return the
    pooled out-of-sample scores plus a per-season table.

    `fit(train_X, train_y) -> object with predict_proba` overrides the
    model (used for the LightGBM comparison); default is the logistic
    pipeline.
    """
    ys, ps, per_season = [], [], []
    seasons = sorted(frame[season_col].unique())
    for s in test_seasons:
        train = frame[frame[season_col] < s]
        test = frame[frame[season_col] == s]
        if train[season_col].nunique() < min_train_seasons or test.empty:
            continue
        model = fit(train[features], train[target]) if fit else \
            make_logistic(C).fit(train[features], train[target])
        p = model.predict_proba(test[features])[:, 1]
        y = test[target].to_numpy()
        sc = score_binary(y, p)
        per_season.append({"season": s, **{k: v for k, v in sc.row("").items() if k != "model"}})
        ys.append(y); ps.append(p)
    if not ys:
        raise ValueError("no test season had enough training history")
    pooled = score_binary(np.concatenate(ys), np.concatenate(ps))
    return pooled, pd.DataFrame(per_season)


def fixed_split(frame: pd.DataFrame, features: list[str], target: str,
                season_col: str, train_until, test_from, test_until=None,
                C: float = 1.0, fit=None):
    """Fit once on seasons <= train_until, score seasons in
    [test_from, test_until]. Returns (Scores, fitted model)."""
    train = frame[frame[season_col] <= train_until]
    test = frame[frame[season_col] >= test_from]
    if test_until is not None:
        test = test[test[season_col] <= test_until]
    model = fit(train[features], train[target]) if fit else \
        make_logistic(C).fit(train[features], train[target])
    p = model.predict_proba(test[features])[:, 1]
    return score_binary(test[target].to_numpy(), p), model
