"""Train a LightGBM NFL win-probability model with temporal validation.

Two targets supported:
  - 'win': straight-up winner (team perspective, target column)
  - 'ats': covers the spread (drops pushes)

Compared against three baselines: pick-the-favorite, market-implied (from
spread, via a logistic fit on training data), and always-home (not
available here — no home/away column — so falls back to pick-favorite).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression

from features import build_feature_frame


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "2023-2025W3.csv"
MODEL_DIR = REPO_ROOT / "NFL" / "model" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["Date"] < "2024-09-01"]
    val = df[(df["Date"] >= "2024-09-01") & (df["Date"] < "2025-01-15")]
    test = df[df["Date"] >= "2025-01-15"]
    return train, val, test


def market_baseline_probs(train_spreads: pd.Series, train_y: pd.Series,
                          eval_spreads: pd.Series) -> np.ndarray:
    """Fit a 1-feature logistic on spread to get market-implied win prob."""
    lr = LogisticRegression()
    lr.fit(train_spreads.values.reshape(-1, 1), train_y.values)
    return lr.predict_proba(eval_spreads.values.reshape(-1, 1))[:, 1]


def evaluate(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "model": name,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
    }
    return out


def run(target: str = "win") -> None:
    print(f"Loading + building features ({DATA_PATH.name})...")
    df, feature_cols = build_feature_frame(str(DATA_PATH))

    if target == "win":
        y_col = "target"
    elif target == "ats":
        df = df.dropna(subset=["ats_cover"]).copy()
        y_col = "ats_cover"
    else:
        raise ValueError(f"unknown target: {target}")

    df[y_col] = df[y_col].astype(int)

    train, val, test = temporal_split(df)
    print(f"  rows  -> train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"  dates -> train {train['Date'].min().date()}..{train['Date'].max().date()}"
          f" | val {val['Date'].min().date()}..{val['Date'].max().date()}"
          f" | test {test['Date'].min().date()}..{test['Date'].max().date()}")

    X_train, y_train = train[feature_cols], train[y_col]
    X_val, y_val = val[feature_cols], val[y_col]
    X_test, y_test = test[feature_cols], test[y_col]

    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1,
    )

    print("\nTraining LightGBM...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"  best_iter={model.best_iteration_}")

    results = []
    for name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        prob = model.predict_proba(X)[:, 1]
        results.append(evaluate(f"lgbm_{name}", y.values, prob))

        fav_prob = (train.loc[X.index, "Spread_vg"] if False else None)
        spread = df.loc[X.index, "Spread_vg"].values
        fav_pred = (spread < 0).astype(float)
        results.append(evaluate(f"pick_favorite_{name}", y.values, fav_pred * 0.98 + 0.01))

        mkt_prob = market_baseline_probs(
            train["Spread_vg"], y_train, df.loc[X.index, "Spread_vg"]
        )
        results.append(evaluate(f"market_logistic_{name}", y.values, mkt_prob))

    print("\n=== Results ===")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    imp = pd.DataFrame({
        "feature": feature_cols,
        "gain": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False).head(20)
    print("\nTop 20 features by gain:")
    print(imp.to_string(index=False))

    out_path = MODEL_DIR / f"lgbm_{target}.txt"
    model.booster_.save_model(str(out_path))
    res_df.to_csv(MODEL_DIR / f"metrics_{target}.csv", index=False)
    imp.to_csv(MODEL_DIR / f"importance_{target}.csv", index=False)
    with open(MODEL_DIR / f"features_{target}.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"\nSaved model + metrics to {MODEL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["win", "ats"], default="win")
    args = parser.parse_args()
    run(args.target)
