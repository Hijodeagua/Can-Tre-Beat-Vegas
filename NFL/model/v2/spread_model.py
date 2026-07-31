"""A spread built from scratch: bake-off, calibration, and KPIs.

`margin.py` established that a market-blind line lands close to Vegas. This is
the full treatment of that question — the same protocol the classifier got, but
for a point prediction:

- six model families on an identical harness
- walk-forward, one retrain per season, recency-weighted
- a calibration step fitted on the prior season
- KPIs framed around the only question that matters here: **is our number
  closer to the final margin than the bookmaker's?**

Every model is **blind to the market**: `spread_line`, `total_line`,
`market_home_prob`, `market_vig` and `elo_vs_spread` are all removed, so the
line is constructed from Elo, form, rest, travel, weather and roster quality
alone. Nothing here can copy the answer.

## Why calibrate a regression

A point forecast is calibrated when regressing actual on predicted gives slope
1 and intercept 0. Tree ensembles systematically shrink toward the mean —
predicting +3 for games that actually average +5 — which a slope below 1
exposes. The fix is a linear map fitted on a held-out season, the direct
analogue of the Platt step the classifier uses.

## Reading the KPIs

MAE and RMSE say how wrong the line is. **R² is meaningful here**, unlike on the
binary win target. But the decisive column is `closer_than_vegas`: the share of
games where our line missed by less than the closing spread did. 50% means we
have matched the bookmaker; below 50% means they are better, and no amount of
favourable MAE rounding changes that.

Usage
    python3 -m NFL.model.v2.spread_model --evaluate
    python3 -m NFL.model.v2.spread_model --evaluate --kind lgbm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .compare_models import recency_weights
from .dataset import build_dataset, feature_matrix
from .margin import MARKET_FEATURES, blind_features

ARTIFACTS = Path(__file__).resolve().parent / "artifacts" / "spread"

EVAL_FROM = 2015
HALF_LIFE = 6.0
SEED = 17

MODEL_KINDS = ["ridge", "huber", "rf", "extratrees", "xgb", "lgbm"]


def make_regressor(kind: str):
    """Fresh estimator per season. Linear models get imputation + scaling."""
    if kind == "ridge":
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("reg", Ridge(alpha=10.0))])
    if kind == "huber":
        # NFL margins are heavy-tailed; a 45-point blowout should not drag the
        # fit the way squared error lets it.
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("reg", HuberRegressor(alpha=1e-3, epsilon=1.35,
                                                max_iter=500))])
    if kind == "rf":
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("reg", RandomForestRegressor(
                             n_estimators=400, min_samples_leaf=20,
                             max_features="sqrt", n_jobs=-1, random_state=SEED))])
    if kind == "extratrees":
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("reg", ExtraTreesRegressor(
                             n_estimators=400, min_samples_leaf=20,
                             max_features="sqrt", n_jobs=-1, random_state=SEED))])
    if kind == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4,
                            min_child_weight=30, subsample=0.8,
                            colsample_bytree=0.7, reg_lambda=5.0,
                            tree_method="hist", random_state=SEED, n_jobs=-1)
    if kind == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                             min_child_samples=60, colsample_bytree=0.7,
                             subsample=0.8, subsample_freq=1, reg_lambda=5.0,
                             random_state=SEED, verbose=-1, n_jobs=-1)
    raise ValueError(f"unknown model kind: {kind}")


def _weight_key(model) -> str:
    return "reg__sample_weight" if isinstance(model, Pipeline) else "sample_weight"


def walk_forward(df: pd.DataFrame, kind: str, features: list[str],
                 start_season: int = EVAL_FROM,
                 calibrate: bool = True) -> pd.DataFrame:
    """Out-of-sample margin predictions, one retrain per season.

    Fits on everything before S-1, calibrates on S-1, predicts S — the same
    three-way split the classifier uses, so the two are directly comparable.
    """
    played = df.dropna(subset=["margin"]).copy()
    out = []
    for season in sorted(s for s in played["season"].unique() if s >= start_season):
        fit = played[played["season"] < season - 1]
        cal = played[played["season"] == season - 1]
        test = played[played["season"] == season]
        if len(fit) < 500 or test.empty:
            continue

        model = make_regressor(kind)
        w = recency_weights(fit["season"], int(season), HALF_LIFE)
        model.fit(feature_matrix(fit, features), fit["margin"],
                  **{_weight_key(model): w})

        raw_test = model.predict(feature_matrix(test, features))
        pred = raw_test
        slope, intercept = 1.0, 0.0
        if calibrate and len(cal) >= 100:
            raw_cal = model.predict(feature_matrix(cal, features))
            lin = LinearRegression().fit(raw_cal.reshape(-1, 1), cal["margin"])
            slope, intercept = float(lin.coef_[0]), float(lin.intercept_)
            pred = lin.predict(raw_test.reshape(-1, 1))

        rec = test[["game_id", "season", "week", "home_team", "away_team",
                    "spread_line", "margin"]].copy()
        rec["pred_raw"] = raw_test
        rec["pred_margin"] = pred
        rec["cal_slope_fitted"] = slope
        rec["cal_intercept_fitted"] = intercept
        rec["model"] = kind
        out.append(rec)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# --------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------

def kpis(preds: pd.DataFrame, label: str, col: str = "pred_margin") -> dict:
    d = preds.dropna(subset=["spread_line"])
    y = d["margin"].to_numpy()
    p = d[col].to_numpy()
    vegas = d["spread_line"].to_numpy()

    # Calibration of the shipped prediction: regress actual on predicted.
    lin = LinearRegression().fit(p.reshape(-1, 1), y)

    err_model = np.abs(y - p)
    err_vegas = np.abs(y - vegas)
    decided = err_model != err_vegas   # exact ties are pushes for this metric

    return {
        "model": label,
        "n": int(len(y)),
        "mae": round(float(mean_absolute_error(y, p)), 3),
        "rmse": round(float(np.sqrt(np.mean((y - p) ** 2))), 3),
        "r2": round(float(r2_score(y, p)), 4),
        "bias": round(float(np.mean(y - p)), 3),
        "cal_slope": round(float(lin.coef_[0]), 3),
        "cal_intercept": round(float(lin.intercept_), 3),
        "corr_with_vegas": round(float(np.corrcoef(p, vegas)[0, 1]), 4),
        "mean_abs_disagreement": round(float(np.mean(np.abs(p - vegas))), 2),
        "closer_than_vegas": round(float((err_model < err_vegas)[decided].mean()), 4),
    }


def vegas_row(preds: pd.DataFrame) -> dict:
    """The bookmaker's own line, scored the same way."""
    d = preds.dropna(subset=["spread_line"])
    y, v = d["margin"].to_numpy(), d["spread_line"].to_numpy()
    lin = LinearRegression().fit(v.reshape(-1, 1), y)
    return {
        "model": "closing spread (Vegas)", "n": int(len(y)),
        "mae": round(float(mean_absolute_error(y, v)), 3),
        "rmse": round(float(np.sqrt(np.mean((y - v) ** 2))), 3),
        "r2": round(float(r2_score(y, v)), 4),
        "bias": round(float(np.mean(y - v)), 3),
        "cal_slope": round(float(lin.coef_[0]), 3),
        "cal_intercept": round(float(lin.intercept_), 3),
        "corr_with_vegas": 1.0, "mean_abs_disagreement": 0.0,
        "closer_than_vegas": np.nan,
    }


def head_to_head_by_disagreement(preds: pd.DataFrame,
                                 bins=(0, 1, 2, 3, 5, 40)) -> pd.DataFrame:
    """When we disagree with the book by N points, who ends up closer?

    This is where a real edge would show itself: if our line is only as good as
    Vegas on average but *better* when we disagree strongly, that is a tradable
    signal. If it gets worse as disagreement grows, our outliers are just noise.
    """
    d = preds.dropna(subset=["spread_line"]).copy()
    d["disagree"] = (d["pred_margin"] - d["spread_line"]).abs()
    d["err_model"] = (d["margin"] - d["pred_margin"]).abs()
    d["err_vegas"] = (d["margin"] - d["spread_line"]).abs()
    d["ats"] = d["margin"] - d["spread_line"]

    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        g = d[(d["disagree"] >= lo) & (d["disagree"] < hi)]
        if len(g) < 60:
            continue
        graded = g[g["ats"] != 0]
        side = np.where(g.loc[graded.index, "pred_margin"] >
                        g.loc[graded.index, "spread_line"],
                        graded["ats"] > 0, graded["ats"] < 0)
        decided = g["err_model"] != g["err_vegas"]
        rows.append({
            "disagreement_pts": f"{lo}-{hi}",
            "games": len(g),
            "model_mae": round(float(g["err_model"].mean()), 2),
            "vegas_mae": round(float(g["err_vegas"].mean()), 2),
            "closer_than_vegas": round(float(
                (g["err_model"] < g["err_vegas"])[decided].mean()), 4),
            "ats_betting_our_side": round(float(side.mean()), 4),
            "n_ats": int(len(graded)),
        })
    return pd.DataFrame(rows)


def run(kinds: list[str], save: bool = False) -> tuple[pd.DataFrame, dict]:
    df = build_dataset(with_squad=True)
    feats = blind_features(df)
    print(f"blind feature set: {len(feats)} features "
          f"(removed {', '.join(sorted(MARKET_FEATURES))})\n", flush=True)

    table, store = [], {}
    for kind in kinds:
        p = walk_forward(df, kind, feats)
        if p.empty:
            continue
        store[kind] = p
        table.append(kpis(p, kind))
        # Uncalibrated companion, to show what the calibration step bought.
        table.append(kpis(p, f"{kind} (uncalibrated)", col="pred_raw"))
        print(f"  {kind:<11} mae={table[-2]['mae']:.3f} "
              f"r2={table[-2]['r2']:.4f} "
              f"closer={table[-2]['closer_than_vegas']:.3f}", flush=True)

    table.append(vegas_row(next(iter(store.values()))))
    out = pd.DataFrame(table).sort_values("mae").reset_index(drop=True)
    return out, store


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kinds", default=",".join(MODEL_KINDS))
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()

    table, store = run(args.kinds.split(","))
    print("\n=== Spread bake-off, walk-forward %d-2025 ===" % EVAL_FROM)
    print(table.to_string(index=False), flush=True)

    best = table[~table["model"].str.contains("uncalibrated|Vegas")].iloc[0]["model"]
    print(f"\nbest by MAE: {best}")
    print("\n=== Head to head with the book, by size of disagreement ===")
    print(head_to_head_by_disagreement(store[best]).to_string(index=False), flush=True)

    if args.evaluate:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        table.to_csv(ARTIFACTS / "bakeoff.csv", index=False)
        head_to_head_by_disagreement(store[best]).to_csv(
            ARTIFACTS / "head_to_head.csv", index=False)
        for k, v in store.items():
            v.to_csv(ARTIFACTS / f"preds_{k}.csv", index=False)
        print(f"\nsaved -> {ARTIFACTS}")


if __name__ == "__main__":
    main()
