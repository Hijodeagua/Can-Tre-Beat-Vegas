"""
Grid-search the betting-blind MLB Elo hyperparameters on game-outcome
log-loss (evaluated on 2012+ so the fresh-start burn-in years don't
dominate). Saves the winning parameters and the full run artifacts:

    data/mlb/elo_params.json        tuned hyperparameters + eval metrics
    data/mlb/elo_game_history.csv   per-game pre-ratings & win prob
    data/mlb/elo_team_seasons.csv   per-franchise-season start/end/delta
"""

import itertools
import json
from pathlib import Path

import numpy as np

from elo import REPO, run_history

OUT_DIR = REPO / "data" / "mlb"
EVAL_START = 2012


def evaluate(hist) -> dict:
    ev = hist[hist.season >= EVAL_START]
    p = ev.p_home.clip(1e-9, 1 - 1e-9)
    y = ev.home_win
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    acc = ((p > 0.5) == (y == 1)).mean()
    brier = ((p - y) ** 2).mean()
    # baseline: constant home win rate fitted on the same window
    hr = y.mean()
    ll_base = -(y * np.log(hr) + (1 - y) * np.log(1 - hr)).mean()
    return {
        "log_loss": round(float(ll), 5),
        "accuracy": round(float(acc), 4),
        "brier": round(float(brier), 5),
        "baseline_log_loss": round(float(ll_base), 5),
        "home_win_rate": round(float(hr), 4),
        "n_games": int(len(ev)),
    }


def main():
    grid = {
        "k": [2, 3, 4, 5, 6, 8],
        "home_advantage": [15, 24, 33],
        "carryover": [0.5, 0.6, 0.7, 0.8, 1.0],
        "use_mov": [True, False],
    }
    results = []
    best = None
    for k, ha, co, mov in itertools.product(*grid.values()):
        _, hist, _ = run_history(k, ha, co, mov)
        m = evaluate(hist)
        rec = {"k": k, "home_advantage": ha, "carryover": co,
               "use_mov": mov, **m}
        results.append(rec)
        if best is None or m["log_loss"] < best["log_loss"]:
            best = rec
            print("new best:", rec)

    engine, hist, seasons = run_history(
        best["k"], best["home_advantage"], best["carryover"], best["use_mov"]
    )
    hist.to_csv(OUT_DIR / "elo_game_history.csv", index=False)
    seasons.to_csv(OUT_DIR / "elo_team_seasons.csv", index=False)
    with open(OUT_DIR / "elo_params.json", "w") as fh:
        json.dump({"best": best, "grid_results": results}, fh, indent=2)
    print("\nfinal best:", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
