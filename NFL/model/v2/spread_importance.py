"""Feature importance for the market-blind spread models.

The classifier's importance charts are dominated by `spread_line` and
`market_home_prob` — they carry roughly 80% of the mass and everything else is
rounding. These models cannot see the market at all, so this is the first
ranking in the project that shows what actually predicts a football result when
the bookmaker's opinion is unavailable.

Same protocol as `feature_importance.py`: fit on 2002-2023, measure on held-out
2024-25, permutation importance (degradation in held-out MAE when a column is
shuffled) plus SHAP, then a uniform ranking averaging both across all six model
families.

Usage
    python3 -m NFL.model.v2.spread_importance
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .compare_models import recency_weights
from .dataset import build_dataset, feature_matrix
from .margin import blind_features
from .spread_model import MODEL_KINDS, make_regressor, _weight_key

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent / "artifacts" / "spread" / "importance"
EVAL_SEASONS = (2024, 2025)
HALF_LIFE = 6.0
SEED = 17

BAR = "#2a78d6"
INK = "#374151"
MUTED = "#9ca3af"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.xaxis.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.set_axisbelow(True)


def fit_for_importance(df: pd.DataFrame, kind: str, features: list[str]):
    played = df.dropna(subset=["margin"])
    fit = played[played["season"] < EVAL_SEASONS[0]]
    ev = played[played["season"].isin(EVAL_SEASONS)]

    model = make_regressor(kind)
    w = recency_weights(fit["season"], EVAL_SEASONS[0], HALF_LIFE)
    model.fit(feature_matrix(fit, features), fit["margin"], **{_weight_key(model): w})
    return (model, feature_matrix(fit, features), feature_matrix(ev, features),
            ev["margin"].to_numpy())


def perm_importance(model, X_ev, y_ev) -> pd.DataFrame:
    r = permutation_importance(model, X_ev, y_ev,
                               scoring="neg_mean_absolute_error",
                               n_repeats=10, random_state=SEED, n_jobs=-1)
    return pd.DataFrame({
        "feature": X_ev.columns,
        "perm_importance": r.importances_mean,
        "perm_std": r.importances_std,
    }).sort_values("perm_importance", ascending=False).reset_index(drop=True)


def shap_values(model, kind: str, X_fit: pd.DataFrame, X_ev: pd.DataFrame):
    if kind in ("ridge", "huber"):
        imp, sc = model.named_steps["impute"], model.named_steps["scale"]
        reg = model.named_steps["reg"]
        bg = sc.transform(imp.transform(X_fit.sample(500, random_state=SEED)))
        Xe = sc.transform(imp.transform(X_ev))
        sv = shap.LinearExplainer(reg, bg).shap_values(Xe)
        display_X = X_ev
    elif kind in ("rf", "extratrees"):
        reg = model.named_steps["reg"]
        Xe = model.named_steps["impute"].transform(X_ev)
        sv = shap.TreeExplainer(reg).shap_values(Xe, check_additivity=False)
        display_X = pd.DataFrame(Xe, columns=X_ev.columns)
    else:
        sv = shap.TreeExplainer(model).shap_values(X_ev)
        display_X = X_ev
    mean_abs = pd.DataFrame({
        "feature": X_ev.columns,
        "shap_mean_abs": np.abs(sv).mean(axis=0),
    }).sort_values("shap_mean_abs", ascending=False).reset_index(drop=True)
    return mean_abs, sv, display_X


def plot_perm(perm: pd.DataFrame, kind: str, top: int = 20) -> Path:
    d = perm.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6.5), dpi=150)
    ax.barh(d["feature"], d["perm_importance"], xerr=d["perm_std"],
            color=BAR, height=0.62, error_kw={"ecolor": MUTED, "lw": 0.8})
    _style(ax)
    ax.set_xlabel("points of held-out MAE lost when shuffled (2024-25)",
                  fontsize=8, color=INK)
    ax.set_title(f"Spread model — permutation importance ({kind})",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    p = OUT_DIR / f"perm_{kind}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_shap(sv, display_X, kind: str) -> Path:
    plt.figure(figsize=(7, 6.5), dpi=150)
    shap.summary_plot(sv, display_X, max_display=20, show=False, plot_size=None)
    plt.title(f"Spread model — SHAP ({kind})", fontsize=11, loc="left")
    plt.tight_layout()
    p = OUT_DIR / f"shap_{kind}.png"
    plt.savefig(p)
    plt.close("all")
    return p


def uniform_ranking(per_model: dict[str, pd.DataFrame], features: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({"feature": features})
    scores = []
    for kind, tbl in per_model.items():
        t = tbl.copy()
        t["rank_perm"] = t["perm_importance"].rank(ascending=False)
        t["rank_shap"] = t["shap_mean_abs"].rank(ascending=False)
        t[f"rank_{kind}"] = (t["rank_perm"] + t["rank_shap"]) / 2
        for col in ("perm_importance", "shap_mean_abs"):
            mx = t[col].clip(lower=0).max()
            t[f"n_{col}"] = t[col].clip(lower=0) / mx if mx > 0 else 0.0
        t[f"score_{kind}"] = (t["n_perm_importance"] + t["n_shap_mean_abs"]) / 2
        scores.append(t.set_index("feature")[[f"score_{kind}"]])
        out = out.merge(t[["feature", f"rank_{kind}"]], on="feature")

    rank_cols = [c for c in out.columns if c.startswith("rank_")]
    out["avg_rank"] = out[rank_cols].mean(axis=1)
    out["avg_norm_score"] = out["feature"].map(pd.concat(scores, axis=1).mean(axis=1))
    out = out.sort_values(["avg_rank", "avg_norm_score"],
                          ascending=[True, False]).reset_index(drop=True)
    out.insert(0, "uniform_rank", out.index + 1)
    return out


def plot_uniform(rank: pd.DataFrame) -> Path:
    d = rank.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 9.5), dpi=150)
    ax.barh(d["feature"], d["avg_norm_score"], color=BAR, height=0.62)
    _style(ax)
    ax.set_xlabel("mean normalized importance across 6 models", fontsize=8, color=INK)
    ax.set_title("Market-blind spread model — uniform feature ranking",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    p = OUT_DIR / "uniform_ranking.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kinds", default=",".join(MODEL_KINDS))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset(with_squad=True)
    feats = blind_features(df)
    per_model = {}
    for kind in args.kinds.split(","):
        print(f"[{kind}] fitting...", flush=True)
        model, X_fit, X_ev, y_ev = fit_for_importance(df, kind, feats)
        perm = perm_importance(model, X_ev, y_ev)
        sh, sv, display_X = shap_values(model, kind, X_fit, X_ev)
        merged = perm.merge(sh, on="feature")
        merged.to_csv(OUT_DIR / f"perm_{kind}.csv", index=False)
        plot_perm(perm, kind)
        plot_shap(sv, display_X, kind)
        per_model[kind] = merged
        print(f"[{kind}] top5: {', '.join(perm['feature'].head(5))}", flush=True)

    rank = uniform_ranking(per_model, feats)
    rank.to_csv(OUT_DIR / "uniform_ranking.csv", index=False)
    plot_uniform(rank)
    print("\n=== uniform ranking, market-blind spread model ===")
    print(rank.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
