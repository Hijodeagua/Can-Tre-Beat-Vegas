"""Feature-ablation study over the top-15 features from the uniform ranking.

## Why not the full 2^15 sweep

One honest evaluation of one subset = a full walk-forward (one retrain per
test season). Measured on this machine that costs ~1.5s per subset for
logistic and ~8-40s for the tree families. The exhaustive 2^15 = 32,767
non-empty subsets therefore runs ~14 hours for logistic alone and multiple
*weeks* for the tree models. Anything cheaper (single split instead of
walk-forward) would answer a different, easier question and its subset
rankings would not transfer.

## What runs instead (~1,600 evaluations)

- **Tier A — winner check.** All five families on a common grid: top-5/8/10/
  12/15 prefixes + the full 45, full walk-forward 2015-2025. Answers "does
  trimming change which model type wins".
- **Tier B — exhaustive core.** Every subset of the top-8 (255) for logistic,
  LightGBM, and Extra Trees (the best tree family), test seasons 2019-2025.
  Genuinely exhaustive where the importance mass actually lives.
- **Tier C — the 15-space, sampled.** Greedy forward selection and backward
  elimination over all 15 (logistic + LightGBM), plus 100 random subsets
  (logistic). Covers sizes 9-15 that Tier B can't reach.

Tradeoff vs exhaustive: Tiers B/C use 2019-2025 test seasons (not 2015-2025)
and a 200-tree LightGBM/250-tree ET, so their absolute numbers sit slightly
above Tier A's; comparisons *within* a tier are apples-to-apples. Random
sampling covers ~0.4% of the 2^15 space — but with importance this
concentrated (top-2 features carry ~80% of the mass), subsets differing only
in tail features are near-duplicates, which is exactly why the full sweep is
low-value.

Usage
    python3 -m NFL.model.v2.ablation --tier all
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .compare_models import make_model, summarize, walk_forward_model
from .dataset import FEATURE_COLS, build_dataset

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
IMPORTANCE = ARTIFACTS / "importance" / "uniform_ranking.csv"
OUT = ARTIFACTS / "ablation"

HALF_LIFE = 6.0
TIER_A_START = 2015
TIER_BC_START = 2019
RANDOM_SUBSETS = 100
SEED = 17


def fast_factory(kind: str):
    """Smaller tree ensembles for the sweep tiers — subset *ranking* is what
    matters there, and 200-tree LightGBM orders subsets the same way 400 does."""
    m = make_model(kind)
    if kind == "lgbm":
        m.set_params(n_estimators=200)
    elif kind in ("rf", "extratrees"):
        m.named_steps["clf"].set_params(n_estimators=250)
    return m


def top_features(n: int = 15) -> list[str]:
    rank = pd.read_csv(IMPORTANCE)
    return rank["feature"].head(n).tolist()


def evaluate(df, kind, features, start, label, fast=False):
    preds = walk_forward_model(
        df, "win", kind, start, HALF_LIFE, features=features,
        model_factory=fast_factory if fast else None)
    row = summarize(preds, label)
    row.update(model_kind=kind, n_features=len(features),
               features="|".join(features))
    return row


def tier_a(df: pd.DataFrame, top15: list[str]) -> pd.DataFrame:
    """All five families × prefix sets × full 45. Full-fidelity walk-forward."""
    sets = {f"top{k}": top15[:k] for k in (5, 8, 10, 12, 15)}
    sets["full45"] = list(FEATURE_COLS)
    rows = []
    for kind in ("logistic", "rf", "extratrees", "xgb", "lgbm"):
        for name, feats in sets.items():
            rows.append(evaluate(df, kind, feats, TIER_A_START, f"{kind}/{name}"))
            print(f"[A] {kind}/{name}: log_loss={rows[-1]['log_loss']}", flush=True)
    return pd.DataFrame(rows)


def tier_b(df: pd.DataFrame, top15: list[str]) -> pd.DataFrame:
    """Exhaustive over the top 8 — every non-empty subset, three families."""
    core = top15[:8]
    subsets = [list(c) for r in range(1, 9) for c in itertools.combinations(core, r)]
    rows = []
    for kind in ("logistic", "lgbm", "extratrees"):
        for i, feats in enumerate(subsets):
            rows.append(evaluate(df, kind, feats, TIER_BC_START,
                                 f"{kind}/exh8", fast=True))
            if (i + 1) % 64 == 0:
                print(f"[B] {kind}: {i + 1}/255", flush=True)
    return pd.DataFrame(rows)


def _stepwise(df, kind, pool, forward: bool):
    selected = [] if forward else list(pool)
    rows, best_ll = [], np.inf
    steps = len(pool) if forward else len(pool) - 1
    for _ in range(steps):
        candidates = ([f for f in pool if f not in selected] if forward
                      else list(selected))
        results = []
        for f in candidates:
            feats = selected + [f] if forward else [x for x in selected if x != f]
            if not feats:
                continue
            r = evaluate(df, kind, feats, TIER_BC_START,
                         f"{kind}/{'fwd' if forward else 'bwd'}", fast=True)
            results.append((r["log_loss"], f, r))
        results.sort(key=lambda t: t[0])
        ll, f, r = results[0]
        selected = selected + [f] if forward else [x for x in selected if x != f]
        r["step_feature"] = ("+" if forward else "-") + f
        rows.append(r)
        best_ll = min(best_ll, ll)
        print(f"[C] {kind} {'fwd' if forward else 'bwd'} "
              f"{'+' if forward else '-'}{f}: {ll}", flush=True)
    return pd.DataFrame(rows)


def tier_c(df: pd.DataFrame, top15: list[str]) -> pd.DataFrame:
    frames = []
    for kind in ("logistic", "lgbm"):
        frames.append(_stepwise(df, kind, top15, forward=True))
        frames.append(_stepwise(df, kind, top15, forward=False))
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(RANDOM_SUBSETS):
        size = int(rng.integers(3, 16))
        feats = list(rng.choice(top15, size=size, replace=False))
        rows.append(evaluate(df, "logistic", feats, TIER_BC_START,
                             "logistic/rand", fast=True))
        if (i + 1) % 25 == 0:
            print(f"[C] random: {i + 1}/{RANDOM_SUBSETS}", flush=True)
    frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="all", choices=["all", "a", "b", "c"])
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    top15 = top_features(15)
    print(f"top-15: {top15}", flush=True)

    if args.tier in ("all", "a"):
        a = tier_a(df, top15)
        a.to_csv(OUT / "tier_a_prefix_grid.csv", index=False)
    if args.tier in ("all", "b"):
        b = tier_b(df, top15)
        b.to_csv(OUT / "tier_b_exhaustive8.csv", index=False)
    if args.tier in ("all", "c"):
        c = tier_c(df, top15)
        c.to_csv(OUT / "tier_c_stepwise_random.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
