"""Permutation + SHAP importance for every bake-off model family.

Setup mirrors live use: each model fits on 2002-2023 with the production
recency weighting (half-life 6, aimed at the eval window), then importance is
measured on a held-out 2024-2025 window (~570 games) the model never saw.

- **Permutation importance** — degradation in held-out log loss when one
  feature's values are shuffled (n_repeats=10). Measures what the model
  actually leans on out of sample.
- **SHAP** — mean |SHAP value| over the held-out window. LinearExplainer for
  the logistic pipeline, TreeExplainer for the tree families.

The uniform ranking averages, per model, the feature's permutation rank and
SHAP rank, then averages that across the five models. Ties broken by mean
normalized importance. Outputs:

    artifacts/importance/perm_<model>.csv / .png
    artifacts/importance/shap_<model>.csv / .png   (beeswarm)
    artifacts/importance/uniform_ranking.csv / .png

Usage
    python3 -m NFL.model.v2.feature_importance --target win
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

from .compare_models import MODEL_KINDS, make_model, recency_weights
from .dataset import FEATURE_COLS, build_dataset, feature_matrix
from .train import TARGETS

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent / "artifacts" / "importance"

EVAL_SEASONS = (2024, 2025)
HALF_LIFE = 6.0
SEED = 17

BAR = "#4269d0"      # single sequential hue — magnitude only, no identity
INK = "#374151"
MUTED = "#9ca3af"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.xaxis.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.set_axisbelow(True)


def fit_for_importance(df: pd.DataFrame, target: str, kind: str):
    y_col = TARGETS[target]
    played = df.dropna(subset=[y_col])
    fit = played[played["season"] < EVAL_SEASONS[0]]
    ev = played[played["season"].isin(EVAL_SEASONS)]

    model = make_model(kind)
    w = recency_weights(fit["season"], EVAL_SEASONS[0], HALF_LIFE)
    wkey = "clf__sample_weight" if isinstance(model, Pipeline) else "sample_weight"
    model.fit(feature_matrix(fit), fit[y_col].astype(int), **{wkey: w})
    return model, feature_matrix(fit), feature_matrix(ev), ev[y_col].astype(int).to_numpy()


def perm_importance(model, X_ev, y_ev) -> pd.DataFrame:
    r = permutation_importance(
        model, X_ev, y_ev, scoring="neg_log_loss",
        n_repeats=10, random_state=SEED, n_jobs=-1)
    return pd.DataFrame({
        "feature": X_ev.columns,
        "perm_importance": r.importances_mean,
        "perm_std": r.importances_std,
    }).sort_values("perm_importance", ascending=False).reset_index(drop=True)


def shap_values(model, kind: str, X_fit: pd.DataFrame, X_ev: pd.DataFrame):
    """Mean |SHAP| per feature + the raw matrix for the beeswarm."""
    if kind == "logistic":
        imp = model.named_steps["impute"]
        sc = model.named_steps["scale"]
        lr = model.named_steps["clf"]
        bg = sc.transform(imp.transform(X_fit.sample(500, random_state=SEED)))
        Xe = sc.transform(imp.transform(X_ev))
        sv = shap.LinearExplainer(lr, bg).shap_values(Xe)
        display_X = X_ev  # show raw feature values in the beeswarm
    elif kind in ("rf", "extratrees"):
        clf = model.named_steps["clf"]
        Xe_raw = model.named_steps["impute"].transform(X_ev)
        sv = shap.TreeExplainer(clf).shap_values(Xe_raw, check_additivity=False)
        if isinstance(sv, list):          # sklearn returns per-class
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
        display_X = pd.DataFrame(Xe_raw, columns=X_ev.columns)
    else:  # xgb, lgbm accept NaNs natively
        sv = shap.TreeExplainer(model).shap_values(X_ev)
        if isinstance(sv, list):
            sv = sv[1]
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
    ax.set_xlabel("Δ held-out log loss when shuffled (2024-25)", fontsize=8, color=INK)
    ax.set_title(f"Permutation importance — {kind}", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    p = OUT_DIR / f"perm_{kind}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_shap(sv: np.ndarray, display_X: pd.DataFrame, kind: str) -> Path:
    plt.figure(figsize=(7, 6.5), dpi=150)
    shap.summary_plot(sv, display_X, max_display=20, show=False, plot_size=None)
    plt.title(f"SHAP summary — {kind}", fontsize=11, loc="left")
    plt.tight_layout()
    p = OUT_DIR / f"shap_{kind}.png"
    plt.savefig(p)
    plt.close("all")
    return p


def uniform_ranking(per_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average of (perm rank + shap rank)/2 across models; lower = better."""
    out = pd.DataFrame({"feature": FEATURE_COLS})
    norm_scores = []
    for kind, tbl in per_model.items():
        t = tbl.copy()
        t["rank_perm"] = t["perm_importance"].rank(ascending=False)
        t["rank_shap"] = t["shap_mean_abs"].rank(ascending=False)
        t[f"rank_{kind}"] = (t["rank_perm"] + t["rank_shap"]) / 2
        for col in ("perm_importance", "shap_mean_abs"):
            mx = t[col].clip(lower=0).max()
            t[f"n_{col}"] = t[col].clip(lower=0) / mx if mx > 0 else 0.0
        t[f"score_{kind}"] = (t["n_perm_importance"] + t["n_shap_mean_abs"]) / 2
        norm_scores.append(t[["feature", f"rank_{kind}", f"score_{kind}"]])
        out = out.merge(t[["feature", f"rank_{kind}"]], on="feature")
    rank_cols = [c for c in out.columns if c.startswith("rank_")]
    out["avg_rank"] = out[rank_cols].mean(axis=1)
    score = pd.concat([s.set_index("feature").filter(like="score_") for s in norm_scores],
                      axis=1).mean(axis=1)
    out["avg_norm_score"] = out["feature"].map(score)
    out = out.sort_values(["avg_rank", "avg_norm_score"],
                          ascending=[True, False]).reset_index(drop=True)
    out.insert(0, "uniform_rank", out.index + 1)
    return out


def plot_uniform(rank: pd.DataFrame) -> Path:
    d = rank.head(45).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 10), dpi=150)
    ax.barh(d["feature"], d["avg_norm_score"], color=BAR, height=0.62)
    _style(ax)
    ax.set_xlabel("mean normalized importance across 5 models", fontsize=8, color=INK)
    ax.set_title("Uniform feature ranking — all 45 features", fontsize=11,
                 color=INK, loc="left")
    fig.tight_layout()
    p = OUT_DIR / "uniform_ranking.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="win", choices=list(TARGETS))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    per_model = {}
    for kind in MODEL_KINDS:
        print(f"[{kind}] fitting...")
        model, X_fit, X_ev, y_ev = fit_for_importance(df, args.target, kind)
        perm = perm_importance(model, X_ev, y_ev)
        sh, sv, display_X = shap_values(model, kind, X_fit, X_ev)
        merged = perm.merge(sh, on="feature")
        merged.to_csv(OUT_DIR / f"perm_{kind}.csv", index=False)
        plot_perm(perm, kind)
        plot_shap(sv, display_X, kind)
        per_model[kind] = merged
        print(f"[{kind}] top5 perm: {', '.join(perm['feature'].head(5))}")

    rank = uniform_ranking(per_model)
    rank.to_csv(OUT_DIR / "uniform_ranking.csv", index=False)
    plot_uniform(rank)
    print("\n=== uniform ranking (all 45) ===")
    print(rank.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
