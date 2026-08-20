"""
Tune each league's Elo parameters (K, home advantage, season regression,
promoted-club entry rating) by grid search.

Objective: mean squared error between the pre-match Elo expectation and the
actual score (1 / 0.5 / 0) — a one-step-ahead Brier score, so every scored
match is predicted with only earlier matches' information. The first two
seasons of each league are burn-in (ratings still finding their level) and
the last two seasons (>= HOLDOUT_SEASON) are excluded entirely; they stay
untouched for `train.py`'s temporal validation.

Writes `artifacts/tuned_params.json`, which `ClubEloEngine.for_league()`
picks up automatically.

Usage:
    python -m soccer.clubs.model.tune
"""

import itertools
import json
from datetime import datetime, timezone

import pandas as pd

from soccer.clubs.data.leagues import LEAGUES, next_season
from soccer.clubs.model.elo import (
    BASE_RATING,
    PARAMS_FILE,
    expected_score,
    load_results,
    mov_multiplier,
)

HOLDOUT_SEASON = "2024-25"  # this season onward is train.py's holdout
BURN_IN_SEASONS = 2

GRID = {
    "k": [6.0, 8.0, 10.0, 12.0, 14.0, 18.0, 22.0, 26.0],
    "home_advantage": [30.0, 45.0, 60.0, 75.0, 90.0],
    "season_regression": [0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
    "entry_rating": [1340.0, 1380.0, 1420.0, 1460.0],
}


def replay_brier(
    matches: list[tuple[str, str, str, int, int]],
    k: float,
    home_advantage: float,
    season_regression: float,
    entry_rating: float,
    score_from_season: str,
) -> tuple[float, int]:
    """Replay one league; return (mean one-step-ahead Brier, matches scored).

    `matches` is (season, home, away, home_score, away_score) in date order.
    A stripped-down twin of ClubEloEngine.update() — kept dumb and local so
    the grid search stays fast; the engine remains the source of truth for
    anything exported.
    """
    ratings: dict[str, float] = {}
    current_season = None
    sse, n = 0.0, 0
    for season, home, away, hs, aw in matches:
        if season != current_season:
            if current_season is not None:
                for t in ratings:
                    ratings[t] += season_regression * (BASE_RATING - ratings[t])
            current_season = season
        r_home = ratings.get(home, entry_rating)
        r_away = ratings.get(away, entry_rating)
        exp_home = expected_score(r_home + home_advantage, r_away)
        goal_diff = hs - aw
        actual = 1.0 if goal_diff > 0 else (0.0 if goal_diff < 0 else 0.5)
        if season >= score_from_season and season < HOLDOUT_SEASON:
            sse += (exp_home - actual) ** 2
            n += 1
        delta = k * mov_multiplier(goal_diff) * (actual - exp_home)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta
    return sse / n, n


def league_matches(df: pd.DataFrame, league: str) -> list[tuple]:
    sub = df[df["league"] == league]
    return list(
        zip(
            sub["season"],
            sub["home_team"],
            sub["away_team"],
            sub["home_score"].astype(int),
            sub["away_score"].astype(int),
        )
    )


def tune_league(df: pd.DataFrame, league: str) -> dict:
    matches = league_matches(df, league)
    score_from = LEAGUES[league].first_season
    for _ in range(BURN_IN_SEASONS):
        score_from = next_season(score_from)

    best = None
    for k, ha, reg, entry in itertools.product(*GRID.values()):
        brier, n = replay_brier(matches, k, ha, reg, entry, score_from)
        if best is None or brier < best["brier"]:
            best = {
                "k": k,
                "home_advantage": ha,
                "season_regression": reg,
                "entry_rating": entry,
                "brier": round(brier, 5),
                "matches_scored": n,
            }
    return best


def main() -> None:
    df = load_results()
    params = {}
    for league in LEAGUES:
        best = tune_league(df, league)
        params[league] = best
        print(
            f"{league:>10}: K={best['k']:.0f}  home={best['home_advantage']:.0f}  "
            f"regression={best['season_regression']:.2f}  entry={best['entry_rating']:.0f}  "
            f"brier={best['brier']:.5f}  ({best['matches_scored']} matches)"
        )

    payload = {
        "tunedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "objective": "one-step-ahead Brier on the home score (draw = 0.5)",
        "burnInSeasons": BURN_IN_SEASONS,
        "holdoutFrom": HOLDOUT_SEASON,
        "grid": GRID,
        "params": params,
    }
    PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_FILE.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {PARAMS_FILE}")


if __name__ == "__main__":
    main()
