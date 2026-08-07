"""Can we make a better spread than Vegas?

Everything else in this repo predicts *whether* the home team wins. This
predicts *by how much* — the same question a bookmaker answers when it opens a
number. That makes it directly comparable to the spread, and it is the only
framing in which "we made a better line" is a testable claim.

Two variants, and the distinction is the whole point:

``blind``
    Trained with **every market feature removed** — no spread, no total, no
    moneyline, no vig. It has to build a number from Elo, form, rest, travel,
    weather and roster quality alone. This is our independent spread, and the
    only honest answer to "could I make this line myself".

``market``
    Keeps the market features. It will mostly learn to copy the spread, so its
    error is a floor on what is achievable rather than evidence of skill. Its
    real use is the residual: where it disagrees with the closing number after
    seeing it, that disagreement is the model's actual opinion.

## Why a bookmaker's line might be beatable at all

The common belief is that a spread is set to split the money evenly. That is
the textbook story, not the practice — books knowingly run unbalanced positions
because bettors reliably overbet favourites, home teams and popular franchises,
and shading the line into that flow earns more than balancing it would. If that
is true here, the shading should be *visible*: the closing spread should be
biased in predictable directions, and a model that ignores the market should
systematically disagree with it on those games. `bias_report` looks for exactly
that.

Metrics: MAE and RMSE against the actual margin, plus R2 — which *is*
meaningful here, unlike on the binary win target. The closing spread's own MAE
is the number to beat (~9.7 points in 2025 - the spread is a good estimator of
a very noisy quantity).

Usage
    python3 -m NFL.model.v2.margin --evaluate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .compare_models import recency_weights
from .dataset import FEATURE_COLS, build_dataset, feature_matrix

ARTIFACTS = Path(__file__).resolve().parent / "artifacts" / "margin"

EVAL_FROM = 2015
HALF_LIFE = 6.0
SEED = 17

# Anything that leaks the bookmaker's opinion into a "blind" line.
MARKET_FEATURES = {"spread_line", "total_line", "market_home_prob", "market_vig",
                   "elo_vs_spread"}


def blind_features() -> list[str]:
    return [c for c in FEATURE_COLS if c not in MARKET_FEATURES]


def make_regressor(kind: str = "ridge"):
    if kind == "ridge":
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("reg", Ridge(alpha=10.0))])
    if kind == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                             min_child_samples=60, colsample_bytree=0.7,
                             subsample=0.8, subsample_freq=1, reg_lambda=5.0,
                             random_state=SEED, verbose=-1, n_jobs=-1)
    raise ValueError(kind)


def walk_forward_margin(df: pd.DataFrame, features: list[str], kind: str = "ridge",
                        start_season: int = EVAL_FROM) -> pd.DataFrame:
    """One retrain per season; predicts the home margin."""
    played = df.dropna(subset=["margin"]).copy()
    out = []
    for season in sorted(s for s in played["season"].unique() if s >= start_season):
        fit = played[played["season"] < season]
        test = played[played["season"] == season]
        if len(fit) < 500 or test.empty:
            continue
        model = make_regressor(kind)
        w = recency_weights(fit["season"], int(season), HALF_LIFE)
        key = "reg__sample_weight" if isinstance(model, Pipeline) else "sample_weight"
        model.fit(feature_matrix(fit, features), fit["margin"], **{key: w})

        rec = test[["game_id", "season", "week", "home_team", "away_team",
                    "spread_line", "margin"]].copy()
        rec["pred_margin"] = model.predict(feature_matrix(test, features))
        out.append(rec)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def accuracy_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """How close is each line to the actual margin?"""
    rows = []
    ref = next(iter(preds.values()))
    y = ref["margin"].to_numpy()

    for label, p in preds.items():
        yy, pm = p["margin"].to_numpy(), p["pred_margin"].to_numpy()
        rows.append({"line": label, "n": len(yy),
                     "mae": round(float(mean_absolute_error(yy, pm)), 3),
                     "rmse": round(float(np.sqrt(np.mean((yy - pm) ** 2))), 3),
                     "r2": round(float(r2_score(yy, pm)), 4),
                     "bias": round(float(np.mean(yy - pm)), 3)})
    # The number to beat.
    sp = ref["spread_line"].to_numpy()
    rows.append({"line": "closing spread (Vegas)", "n": len(y),
                 "mae": round(float(mean_absolute_error(y, sp)), 3),
                 "rmse": round(float(np.sqrt(np.mean((y - sp) ** 2))), 3),
                 "r2": round(float(r2_score(y, sp)), 4),
                 "bias": round(float(np.mean(y - sp)), 3)})
    rows.append({"line": "always 0 (pick'em)", "n": len(y),
                 "mae": round(float(mean_absolute_error(y, np.zeros_like(y))), 3),
                 "rmse": round(float(np.sqrt(np.mean(y ** 2))), 3),
                 "r2": round(float(r2_score(y, np.zeros_like(y))), 4),
                 "bias": round(float(np.mean(y)), 3)})
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def disagreement_ats(p: pd.DataFrame, thresholds=(0, 1, 2, 3, 4, 6)) -> pd.DataFrame:
    """Bet the side our number likes; how often does it cover?"""
    d = p.dropna(subset=["spread_line"]).copy()
    d["disagree"] = d["pred_margin"] - d["spread_line"]
    d["ats"] = d["margin"] - d["spread_line"]
    d = d[d["ats"] != 0]
    rows = []
    for t in thresholds:
        sel = d[d["disagree"].abs() >= t]
        if len(sel) < 60:
            continue
        win = np.where(sel["disagree"] > 0, sel["ats"] > 0, sel["ats"] < 0)
        n = len(sel)
        se = float(np.sqrt(0.25 / n))
        rows.append({"min_disagreement_pts": t, "bets": n,
                     "ats_rate": round(float(win.mean()), 4),
                     "se": round(se, 4),
                     "vs_breakeven": round(float(win.mean()) - 0.5238, 4),
                     "z_vs_breakeven": round((float(win.mean()) - 0.5238) / se, 2)})
    return pd.DataFrame(rows)


def bias_report(p: pd.DataFrame) -> pd.DataFrame:
    """Is the closing spread systematically off in predictable buckets?

    If books shade toward what the public likes, the shading should show up as
    a non-zero average of (actual margin - spread) in those buckets.
    """
    d = p.dropna(subset=["spread_line"]).copy()
    d["ats"] = d["margin"] - d["spread_line"]
    d["fav_size"] = pd.cut(d["spread_line"].abs(), [-0.01, 3, 7, 10, 40],
                           labels=["0-3", "3.5-7", "7.5-10", "10.5+"])
    d["side"] = np.where(d["spread_line"] > 0, "home favourite", "away favourite")

    rows = []
    for keys, g in d.groupby(["side", "fav_size"], observed=True):
        n = len(g)
        if n < 60:
            continue
        mean_ats = float(g["ats"].mean())
        se = float(g["ats"].std(ddof=1) / np.sqrt(n))
        rows.append({"side": keys[0], "spread_size": str(keys[1]), "games": n,
                     "avg_margin_vs_spread": round(mean_ats, 3),
                     "se": round(se, 3), "z": round(mean_ats / se, 2),
                     "fav_cover_pct": round(float(
                         np.where(g["spread_line"] > 0, g["ats"] > 0, g["ats"] < 0).mean()), 4)})
    return pd.DataFrame(rows).sort_values("z")


def run(kind: str = "ridge", save: bool = False) -> None:
    df = build_dataset(with_squad=True)
    blind = blind_features()
    print(f"blind model uses {len(blind)} features "
          f"(dropped: {', '.join(sorted(MARKET_FEATURES))})\n", flush=True)

    preds = {
        "model (blind to market)": walk_forward_margin(df, blind, kind),
        "model (sees market)": walk_forward_margin(df, list(FEATURE_COLS), kind),
    }

    print("=== How close is each line to the actual margin? (%d-2025) ===" % EVAL_FROM)
    print(accuracy_table(preds).to_string(index=False), flush=True)

    print("\n=== Our blind line vs the closing spread ===")
    b = preds["model (blind to market)"]
    diff = b["pred_margin"] - b["spread_line"]
    print(f"  correlation with the closing spread : {np.corrcoef(b.pred_margin, b.spread_line)[0,1]:.4f}")
    print(f"  mean absolute disagreement          : {diff.abs().mean():.2f} points")
    print(f"  disagree by 3+ points on            : {(diff.abs() >= 3).mean():.1%} of games")

    print("\n=== Betting the disagreement (blind model) ===")
    print(disagreement_ats(b).to_string(index=False), flush=True)

    print("\n=== Is the closing spread shaded? (all games) ===")
    print("avg_margin_vs_spread > 0 means favourites underperform the number.")
    print(bias_report(b).to_string(index=False), flush=True)

    if save:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        accuracy_table(preds).to_csv(ARTIFACTS / "accuracy.csv", index=False)
        disagreement_ats(b).to_csv(ARTIFACTS / "disagreement_ats.csv", index=False)
        bias_report(b).to_csv(ARTIFACTS / "bias_report.csv", index=False)
        for k, v in preds.items():
            v.to_csv(ARTIFACTS / f"preds_{k.split('(')[1].strip(') ').replace(' ', '_')}.csv",
                     index=False)
        print(f"\nsaved -> {ARTIFACTS}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", default="ridge", choices=["ridge", "lgbm"])
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()
    run(args.kind, save=args.evaluate)


if __name__ == "__main__":
    main()
