"""Tune the NFL Elo parameters by staged grid search on one-step-ahead log
loss — the NFL sibling of `CFB/model/tune.py`, `mlb/tune_elo.py` and
`soccer/clubs/model/tune.py`.

Objective: mean log loss of the pre-game home-win probability over every
game (regular season and playoffs — both are on the slate and both get
graded) with SCORE_FROM <= season < HOLDOUT_FROM. 1999-2004 are burn-in
from the flat 1500 start; HOLDOUT_FROM onward is never touched here and is
reported separately as the honest out-of-sample number. The split matches
the college tuner's (score 2005-23, hold out 2024-25) so the two holdout
numbers are read the same way.

Six parameters, coordinate descent over the grid: sweep one parameter at a
time holding the rest at their current best, repeat until a full pass
changes nothing. Each replay of ~7,000 games is ~30 ms.

Writes `artifacts/tuned_params.json`, which `NflEloEngine.tuned()` reads.

Usage:
    python -m NFL.elo.tune
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from NFL.elo.engine import (
    DEFAULTS, PARAMS_FILE, REST_BONUS_DAYS, fast_replay, load_games,
)

SCORE_FROM = 2005
HOLDOUT_FROM = 2024
MAX_PASSES = 6

GRID = {
    "k": [12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 30.0],
    "home_advantage": [30.0, 40.0, 48.0, 55.0, 62.0, 70.0, 80.0],
    "season_regression": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
    "playoff_k_mult": [1.0, 1.2, 1.5],
    # 60 is effectively uncapped: one NFL game since 1999 has a 59-point margin.
    "margin_cap": [14.0, 21.0, 28.0, 35.0, 45.0, 60.0],
    "rest_bonus": [0.0, 10.0, 20.0, 25.0, 35.0, 50.0],
}


def replay_rows(games) -> list[tuple]:
    played = games[games["completed"].astype(bool)].sort_values(["date", "gametime", "game_id"])
    return list(zip(
        played["season"].astype(int),
        played["home_team"], played["away_team"],
        played["neutral"].astype(bool), played["playoff"].astype(bool),
        played["home_rest"].fillna(7).ge(REST_BONUS_DAYS),
        played["away_rest"].fillna(7).ge(REST_BONUS_DAYS),
        played["home_score"].astype(float), played["away_score"].astype(float),
    ))


def tune(rows, grid=GRID, start=None, verbose=True) -> dict:
    params = dict(start or DEFAULTS)
    best_ll, n = fast_replay(rows, params, SCORE_FROM, HOLDOUT_FROM)
    if verbose:
        print(f"start: {best_ll:.5f} over {n} games  {params}")
    for p in range(MAX_PASSES):
        changed = False
        for name, values in grid.items():
            for v in values:
                if v == params[name]:
                    continue
                trial = {**params, name: v}
                ll, _ = fast_replay(rows, trial, SCORE_FROM, HOLDOUT_FROM)
                if ll < best_ll - 1e-7:
                    best_ll, params, changed = ll, trial, True
                    if verbose:
                        print(f"  pass {p + 1}: {name}={v:g} -> {ll:.5f}")
        if not changed:
            break
    return {"params": params, "log_loss": best_ll, "games_scored": n}


def main() -> None:
    games = load_games()
    rows = replay_rows(games)
    result = tune(rows)
    params = result["params"]

    holdout_ll, holdout_n = fast_replay(rows, params, HOLDOUT_FROM, 9999)
    default_ll, _ = fast_replay(rows, DEFAULTS, SCORE_FROM, HOLDOUT_FROM)
    default_holdout, _ = fast_replay(rows, DEFAULTS, HOLDOUT_FROM, 9999)
    print(f"\ntuned:    train {result['log_loss']:.5f}  holdout {holdout_ll:.5f} ({holdout_n} games)")
    print(f"defaults: train {default_ll:.5f}  holdout {default_holdout:.5f}")

    payload = {
        "tunedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "objective": ("one-step-ahead log loss on all games incl. playoffs, "
                      f"seasons {SCORE_FROM}-{HOLDOUT_FROM - 1}; coordinate descent over GRID"),
        "scoreFrom": SCORE_FROM,
        "holdoutFrom": HOLDOUT_FROM,
        "grid": GRID,
        "params": params,
        "trainLogLoss": round(result["log_loss"], 5),
        "trainGames": result["games_scored"],
        "holdoutLogLoss": round(holdout_ll, 5),
        "holdoutGames": holdout_n,
        "defaultsTrainLogLoss": round(default_ll, 5),
        "defaultsHoldoutLogLoss": round(default_holdout, 5),
    }
    PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_FILE.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {PARAMS_FILE}")


if __name__ == "__main__":
    main()
