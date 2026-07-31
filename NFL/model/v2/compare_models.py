"""Model bake-off: five model families, one honest harness.

Every candidate gets the same 45 features, the same walk-forward protocol
(train on everything before season S-1, calibrate on S-1, score S), and the
same exponential recency weighting. What differs is only the learner:

- ``logistic``  — scaled + median-imputed logistic regression. The floor any
  tree model has to justify itself against.
- ``rf``        — random forest.
- ``extratrees``— extremely randomized trees.
- ``xgb``       — XGBoost, hist tree method.
- ``lgbm``      — LightGBM (the incumbent from train.py).

Recency weighting: each training game gets weight 0.5 ** (age_seasons /
half_life), age measured from the test season. ``--half-life 0`` disables it.

Beyond the standard metrics, this reports the metric that matches the weekly
product: take the model's 3 most confident straight-up picks each week —
what fraction hit?

Usage
    python3 -m NFL.model.v2.compare_models --target win --start-season 2015
    python3 -m NFL.model.v2.compare_models --target win --half-life-sweep
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import build_dataset, feature_matrix
from .train import TARGETS, _apply_platt, _platt, market_prob

warnings.filterwarnings("ignore", category=FutureWarning)

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

DEFAULT_HALF_LIFE = 3.0  # seasons
SEED = 17


def recency_weights(train_seasons: pd.Series, test_season: int, half_life: float) -> np.ndarray:
    """0.5 ** (age / half_life); half_life <= 0 means unweighted."""
    if half_life <= 0:
        return np.ones(len(train_seasons))
    age = test_season - train_seasons.to_numpy(dtype=float)
    return 0.5 ** (age / half_life)


def make_model(kind: str):
    """Fresh estimator per season. All expose predict_proba."""
    if kind == "logistic":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=2000)),
        ])
    if kind == "rf":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=500, min_samples_leaf=20, max_features="sqrt",
                n_jobs=-1, random_state=SEED)),
        ])
    if kind == "extratrees":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(
                n_estimators=500, min_samples_leaf=20, max_features="sqrt",
                n_jobs=-1, random_state=SEED)),
        ])
    if kind == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=4,
            min_child_weight=30, subsample=0.8, colsample_bytree=0.7,
            reg_lambda=5.0, tree_method="hist", random_state=SEED,
            eval_metric="logloss", n_jobs=-1)
    if kind == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=15,
            min_child_samples=60, colsample_bytree=0.7, subsample=0.8,
            subsample_freq=1, reg_lambda=5.0, random_state=SEED,
            verbose=-1, n_jobs=-1)
    raise ValueError(f"unknown model kind: {kind}")


MODEL_KINDS = ["logistic", "rf", "extratrees", "xgb", "lgbm"]


def _sample_weight_param(model, kind: str) -> str:
    return "clf__sample_weight" if isinstance(model, Pipeline) else "sample_weight"


def walk_forward_model(
    df: pd.DataFrame, target: str, kind: str,
    start_season: int, half_life: float,
) -> pd.DataFrame:
    """Out-of-sample predictions for one model family."""
    y_col = TARGETS[target]
    played = df.dropna(subset=[y_col]).copy()
    seasons = sorted(s for s in played["season"].unique() if s >= start_season)

    out = []
    for season in seasons:
        fit = played[played["season"] < season - 1]
        cal = played[played["season"] == season - 1]
        test = played[played["season"] == season]
        if len(fit) < 500 or test.empty:
            continue

        model = make_model(kind)
        w = recency_weights(fit["season"], int(season), half_life)
        X_fit, y_fit = feature_matrix(fit), fit[y_col].astype(int)
        model.fit(X_fit, y_fit, **{_sample_weight_param(model, kind): w})

        raw_cal = model.predict_proba(feature_matrix(cal))[:, 1]
        platt = _platt(raw_cal, cal[y_col].astype(int).to_numpy())
        raw_test = model.predict_proba(feature_matrix(test))[:, 1]

        rec = test[["game_id", "season", "week", "gameday", "home_team", "away_team",
                    "spread_line", "total_line", "market_home_prob"]].copy()
        rec["y"] = test[y_col].astype(int).to_numpy()
        rec["prob_raw"] = raw_test
        rec["prob"] = _apply_platt(platt, raw_test)
        rec["market_prob"] = market_prob(test, target)
        rec["model"] = kind
        out.append(rec)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def top_k_weekly(preds: pd.DataFrame, k: int = 3) -> dict:
    """The product metric: k most confident picks per (season, week)."""
    p = preds.copy()
    p["confidence"] = np.abs(p["prob"] - 0.5)
    p["hit"] = np.where(p["prob"] >= 0.5, p["y"] == 1, p["y"] == 0)
    top = (p.sort_values("confidence", ascending=False)
             .groupby(["season", "week"], sort=False)
             .head(k))
    weekly = top.groupby(["season", "week"])["hit"].agg(["sum", "size"])
    return {
        "picks": int(weekly["size"].sum()),
        "hits": int(top["hit"].sum()),
        "hit_rate": float(top["hit"].mean()),
        "weeks_swept": float((weekly["sum"] == weekly["size"]).mean()),
        "avg_confidence": float(top["prob"].where(top["prob"] >= 0.5, 1 - top["prob"]).mean()),
    }


def summarize(preds: pd.DataFrame, label: str, k: int = 3) -> dict:
    y = preds["y"].to_numpy()
    p = np.clip(preds["prob"].to_numpy(), 1e-6, 1 - 1e-6)
    top = top_k_weekly(preds, k)
    return {
        "model": label,
        "n": len(y),
        "accuracy": round(float(accuracy_score(y, (p >= 0.5).astype(int))), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "auc": round(float(roc_auc_score(y, p)), 4),
        f"top{k}_hit": round(top["hit_rate"], 4),
        f"top{k}_conf": round(top["avg_confidence"], 4),
        "weeks_swept": round(top["weeks_swept"], 4),
    }


def market_summary(preds: pd.DataFrame, target: str, k: int = 3) -> dict:
    m = preds.copy()
    m["prob"] = m["market_prob"]
    row = summarize(m, "market (closing)", k)
    return row


def run_bakeoff(target: str, start_season: int, half_life: float,
                kinds: list[str], df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = build_dataset() if df is None else df
    rows, all_preds = [], {}
    for kind in kinds:
        preds = walk_forward_model(df, target, kind, start_season, half_life)
        if preds.empty:
            continue
        rows.append(summarize(preds, kind))
        all_preds[kind] = preds
        print(f"  {kind:<11} done ({len(preds)} games)")

    # Market baseline uses any model's frame (same games).
    first = next(iter(all_preds.values()))
    rows.append(market_summary(first, target))

    table = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    return table, all_preds


def half_life_sweep(target: str, start_season: int, kind: str = "lgbm",
                    half_lives: tuple = (0, 1.5, 3.0, 6.0, 10.0)) -> pd.DataFrame:
    df = build_dataset()
    rows = []
    for hl in half_lives:
        preds = walk_forward_model(df, target, kind, start_season, hl)
        row = summarize(preds, f"{kind} hl={hl or 'none'}")
        row["half_life"] = hl
        rows.append(row)
        print(f"  half-life {hl or 'none'}: log_loss={row['log_loss']}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="win", choices=list(TARGETS))
    ap.add_argument("--start-season", type=int, default=2015)
    ap.add_argument("--half-life", type=float, default=DEFAULT_HALF_LIFE)
    ap.add_argument("--models", default=",".join(MODEL_KINDS))
    ap.add_argument("--half-life-sweep", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    if args.half_life_sweep:
        print(f"=== half-life sweep ({args.target}, lgbm) ===")
        tbl = half_life_sweep(args.target, args.start_season)
        print(tbl.drop(columns=["half_life"]).to_string(index=False))
        if args.save:
            tbl.to_csv(ARTIFACTS / f"half_life_sweep_{args.target}.csv", index=False)
        return

    kinds = args.models.split(",")
    print(f"=== bake-off: target={args.target}, seasons>={args.start_season}, "
          f"half-life={args.half_life} ===")
    table, all_preds = run_bakeoff(args.target, args.start_season, args.half_life, kinds)
    print()
    print(table.to_string(index=False))
    if args.save:
        table.to_csv(ARTIFACTS / f"bakeoff_{args.target}.csv", index=False)
        for kind, preds in all_preds.items():
            preds.to_csv(ARTIFACTS / f"bakeoff_preds_{args.target}_{kind}.csv", index=False)
        print(f"\nsaved -> {ARTIFACTS}")


if __name__ == "__main__":
    main()
