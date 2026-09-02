"""Tune the CFB Elo parameters by staged grid search on one-step-ahead log
loss — the college sibling of `mlb/tune_elo.py` and
`soccer/clubs/model/tune.py`.

Objective: mean log loss of the pre-game home-win probability over every
FBS-involved game (FCS opponents included — they are on the slate and get
graded, so the pooled rating has to earn its keep) with
SCORE_FROM <= season < HOLDOUT_FROM. Seasons before SCORE_FROM are burn-in
(the 2001 fresh start needs a few years to mean anything — plan §2.1);
HOLDOUT_FROM onward is never touched here and is reported separately as
the honest out-of-sample number.

Seven parameters is too many for one full grid over a 20k-game replay, so
this runs coordinate descent over the grid: sweep one parameter at a time
holding the rest at their current best, repeat until a full pass changes
nothing. Each replay is ~0.1s, so a pass is a few seconds.

Writes `artifacts/tuned_params.json`, which `CfbEloEngine.tuned()` reads.

Usage:
    python -m CFB.model.tune
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from CFB.data.teams import FBS
from CFB.model.elo import (
    DEFAULTS, PARAMS_FILE, fast_replay, load_games, season_conferences,
)

SCORE_FROM = 2005
HOLDOUT_FROM = 2024
MAX_PASSES = 6

GRID = {
    "k": [20.0, 25.0, 30.0, 35.0, 40.0, 50.0],
    "home_advantage": [40.0, 50.0, 60.0, 70.0, 80.0, 95.0],
    "season_regression": [0.20, 0.30, 0.40, 0.50, 0.60],
    "conf_weight": [0.0, 0.25, 0.5, 0.75, 1.0],
    "fcs_rating": [850.0, 950.0, 1000.0, 1050.0, 1100.0, 1150.0, 1250.0],
    "entry_rating": [1100.0, 1150.0, 1200.0, 1250.0, 1300.0, 1350.0, 1400.0, 1500.0],
    # 100 is effectively uncapped: no FBS game has a 100-point margin.
    "margin_cap": [21.0, 28.0, 35.0, 45.0, 60.0, 80.0, 100.0],
}


def replay_rows(games) -> list[tuple]:
    played = games[games["completed"].astype(bool)].sort_values(["start_utc", "game_id"])
    return list(zip(
        played["season"].astype(int),
        played["home_team"], played["away_team"],
        played["home_division"].eq(FBS), played["away_division"].eq(FBS),
        played["neutral_site"].astype(bool),
        played["home_points"].astype(float), played["away_points"].astype(float),
    ))


def tune(rows, conferences, grid=GRID, start=None, verbose=True) -> dict:
    params = dict(start or DEFAULTS)
    best_ll, n = fast_replay(rows, conferences, params, SCORE_FROM, HOLDOUT_FROM)
    if verbose:
        print(f"start: {best_ll:.5f} over {n} games  {params}")
    for p in range(MAX_PASSES):
        changed = False
        for name, values in grid.items():
            for v in values:
                if v == params[name]:
                    continue
                trial = {**params, name: v}
                ll, _ = fast_replay(rows, conferences, trial, SCORE_FROM, HOLDOUT_FROM)
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
    conferences = season_conferences(games)
    result = tune(rows, conferences)
    params = result["params"]

    holdout_ll, holdout_n = fast_replay(rows, conferences, params, HOLDOUT_FROM, 9999)
    default_ll, _ = fast_replay(rows, conferences, DEFAULTS, SCORE_FROM, HOLDOUT_FROM)
    default_holdout, _ = fast_replay(rows, conferences, DEFAULTS, HOLDOUT_FROM, 9999)
    print(f"\ntuned:    train {result['log_loss']:.5f}  holdout {holdout_ll:.5f} ({holdout_n} games)")
    print(f"defaults: train {default_ll:.5f}  holdout {default_holdout:.5f}")

    payload = {
        "tunedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "objective": ("one-step-ahead log loss on all FBS-involved games, "
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
