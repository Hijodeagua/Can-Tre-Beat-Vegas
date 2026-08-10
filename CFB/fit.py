"""Grid-search the CFB Elo constants (DATA_PULL_PLAN.md §8).

Every config runs the same walk-forward pass as ``CFB.elo`` over all
seasons, but is *scored* only on 2005+ (2000-2004 are burn-in). The metric
is log loss of the pregame home-win probability — the same yardstick the
NFL model reports, so the numbers are comparable across sports.

Also fits ``ELO_PER_POINT`` by regressing the actual margin on the pregame
Elo difference under the best config (through the origin: margin ≈
elo_diff / epp).

Usage
    python3 -m CFB.fit             # coarse grid, ~256 configs
    python3 -m CFB.fit --quick     # 16-config smoke test
    python3 -m CFB.fit --refine    # coarse pass, then a local refine pass

Output: ranked grid to ``data/college_football/agg/fit_grid.csv`` and the
best constants block printed ready to paste into ``CFB/elo.py``.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from CFB import elo

AGG_DIR = elo.AGG_DIR

COARSE = {
    "k": [20.0, 26.0, 32.0, 38.0],
    "hfa": [55.0, 65.0, 75.0, 85.0],
    "regression": [0.25, 0.35, 0.45, 0.55],
    "margin_cap": [21.0, 28.0, 35.0, 999.0],  # 999 = effectively uncapped
}
QUICK = {
    "k": [26.0, 32.0],
    "hfa": [65.0, 75.0],
    "regression": [0.35, 0.45],
    "margin_cap": [28.0, 35.0],
}


def run_grid(games: pd.DataFrame, grid: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    combos = list(itertools.product(*grid.values()))
    for i, combo in enumerate(combos, 1):
        params = dict(zip(grid.keys(), combo))
        res = elo.evaluate(games, **params)
        rows.append({**params, **res})
        if i % 25 == 0 or i == len(combos):
            print(f"  {i}/{len(combos)} configs done")
    return (
        pd.DataFrame(rows)
        .sort_values("log_loss")
        .reset_index(drop=True)
    )


def refine_grid(best: pd.Series) -> dict[str, list[float]]:
    """A tighter grid bracketing the coarse winner."""
    return {
        "k": sorted({best["k"] - 3.0, best["k"], best["k"] + 3.0}),
        "hfa": sorted({best["hfa"] - 5.0, best["hfa"], best["hfa"] + 5.0}),
        "regression": sorted(
            {max(0.0, round(best["regression"] + d, 2)) for d in (-0.05, 0.0, 0.05)}
        ),
        "margin_cap": sorted({best["margin_cap"] - 4.0, best["margin_cap"], best["margin_cap"] + 4.0})
        if best["margin_cap"] < 900
        else [best["margin_cap"]],
    }


def fit_elo_per_point(games: pd.DataFrame, params: dict[str, float]) -> float:
    """margin ≈ elo_diff / epp, least squares through the origin, 2005+."""
    df, _ = elo.walk_forward(games, **params)
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[df["season"] >= elo.EVAL_FROM]
    margin = df["home_score"] - df["away_score"]
    denom = float((df["elo_diff"] * margin).sum())
    if denom == 0:
        return float("nan")
    return round(float((df["elo_diff"] ** 2).sum()) / denom, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--refine", action="store_true")
    args = ap.parse_args()

    games = elo.load_games(pool=True)
    print(f"{len(games)} games loaded (FCS pooled); scoring on {elo.EVAL_FROM}+")

    grid = QUICK if args.quick else COARSE
    results = run_grid(games, grid)

    if args.refine and not args.quick:
        # Keep bracketing the winner until it sits in a grid interior (or
        # three passes) — a best config on a refine boundary means the
        # optimum is still outside the searched box.
        for i in range(3):
            print(f"refine pass {i + 1} around current best…")
            prev_best = tuple(results.iloc[0][["k", "hfa", "regression", "margin_cap"]])
            fine = run_grid(games, refine_grid(results.iloc[0]))
            results = (
                pd.concat([results, fine], ignore_index=True)
                .drop_duplicates(subset=["k", "hfa", "regression", "margin_cap"])
                .sort_values("log_loss")
                .reset_index(drop=True)
            )
            if tuple(results.iloc[0][["k", "hfa", "regression", "margin_cap"]]) == prev_best:
                break

    out = AGG_DIR / "fit_grid.csv"
    results.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\ntop 15 configs by walk-forward log loss (2005+):")
    print(results.head(15).to_string(index=False))

    best = results.iloc[0]
    params = {
        "k": best["k"],
        "hfa": best["hfa"],
        "regression": best["regression"],
        "margin_cap": best["margin_cap"],
    }
    epp = fit_elo_per_point(games, params)
    print("\nbest constants — paste into CFB/elo.py:")
    print(f"K_FACTOR = {best['k']}")
    print(f"HFA_ELO = {best['hfa']}")
    print(f"SEASON_REGRESSION = {best['regression']}")
    print(f"MARGIN_CAP = {best['margin_cap']}")
    print(f"ELO_PER_POINT = {epp}  # fitted margin regression")


if __name__ == "__main__":
    main()
