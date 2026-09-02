"""
Tune each country pool's Elo parameters (K, home advantage, season
regression, and the two tier entry ratings) by grid search over a
two-division replay.

Objective: mean squared error between the pre-match Elo expectation and the
actual score (1 / 0.5 / 0) on **top-flight matches only** — the second
division is replayed (that's where promoted clubs' ratings come from) but
scored only for the record. K and home advantage are shared across the two
divisions; what the second division contributes to the fit is carried
ratings, which is the point.

As before: the first two seasons of each pool are burn-in and everything
from HOLDOUT_SEASON on is excluded — those seasons stay untouched for
`train.py`'s temporal validation.

Writes `artifacts/tuned_params.json`, keyed by pool, which
`ClubEloEngine.for_league()` resolves through `pool_of()`.

Usage:
    python -m soccer.clubs.model.tune
"""

import itertools
import json
from datetime import datetime, timezone

import pandas as pd

from soccer.clubs.data.leagues import LEAGUES, POOLS, TIER1, next_season
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
    "k": [10.0, 12.0, 14.0, 18.0],
    "home_advantage": [45.0, 60.0, 75.0],
    "season_regression": [0.05, 0.10, 0.15, 0.20],
    "entry_rating": [1380.0, 1420.0, 1460.0],
    "entry_rating_t2": [1150.0, 1250.0, 1350.0],
    # Pinned, not swept: the objective scores top-flight matches only, so
    # it cannot see the damage a partial carry does to relegated clubs in
    # the second division (a 0.25 carry cut relegated Premier League sides
    # ~200 Elo overnight, inverting the Championship table vs. ClubElo).
    # Sweeping it again requires an objective that scores both divisions.
    "division_carry": [1.0],
}


def replay_brier(
    matches: list[tuple[str, int, str, str, int, int]],
    k: float,
    home_advantage: float,
    season_regression: float,
    entry_rating: float,
    entry_rating_t2: float,
    division_carry: float,
    score_from_season: str,
) -> tuple[float, int]:
    """Replay one pool; return (mean one-step-ahead Brier on tier-1 matches,
    matches scored).

    `matches` is (season, tier, home, away, home_score, away_score) in date
    order. A stripped-down twin of ClubEloEngine.update() — kept dumb and
    local so the grid search stays fast; the engine remains the source of
    truth for anything exported.
    """
    ratings: dict[str, float] = {}
    last_tier: dict[str, int] = {}
    current_season = None
    sse, n = 0.0, 0
    for season, tier, home, away, hs, aw in matches:
        if season != current_season:
            if current_season is not None:
                for t in ratings:
                    ratings[t] += season_regression * (BASE_RATING - ratings[t])
            current_season = season
        entry = entry_rating if tier == 1 else entry_rating_t2
        for t in (home, away):
            if t in ratings and last_tier.get(t, tier) != tier:
                ratings[t] = entry + division_carry * (ratings[t] - entry)
            last_tier[t] = tier
        r_home = ratings.get(home, entry)
        r_away = ratings.get(away, entry)
        exp_home = expected_score(r_home + home_advantage, r_away)
        goal_diff = hs - aw
        actual = 1.0 if goal_diff > 0 else (0.0 if goal_diff < 0 else 0.5)
        if tier == 1 and score_from_season <= season < HOLDOUT_SEASON:
            sse += (exp_home - actual) ** 2
            n += 1
        delta = k * mov_multiplier(goal_diff) * (actual - exp_home)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta
    return sse / n, n


def pool_matches(df: pd.DataFrame, pool: str) -> list[tuple]:
    sub = df[df["league"].isin(POOLS[pool])].sort_values("date", kind="stable")
    tiers = sub["league"].map(lambda lg: LEAGUES[lg].tier)
    return list(
        zip(
            sub["season"],
            tiers,
            sub["home_team"],
            sub["away_team"],
            sub["home_score"].astype(int),
            sub["away_score"].astype(int),
        )
    )


def tune_pool(df: pd.DataFrame, pool: str) -> dict:
    matches = pool_matches(df, pool)
    score_from = TIER1[pool].first_season
    for _ in range(BURN_IN_SEASONS):
        score_from = next_season(score_from)

    best = None
    for k, ha, reg, e1, e2, carry in itertools.product(*GRID.values()):
        brier, n = replay_brier(matches, k, ha, reg, e1, e2, carry, score_from)
        if best is None or brier < best["brier"]:
            best = {
                "k": k,
                "home_advantage": ha,
                "season_regression": reg,
                "entry_rating": e1,
                "entry_rating_t2": e2,
                "division_carry": carry,
                "brier": round(brier, 5),
                "matches_scored": n,
            }
    return best


def main() -> None:
    df = load_results()
    params = {}
    for pool in POOLS:
        best = tune_pool(df, pool)
        params[pool] = best
        print(
            f"{pool:>10}: K={best['k']:.0f}  home={best['home_advantage']:.0f}  "
            f"regression={best['season_regression']:.2f}  "
            f"entry={best['entry_rating']:.0f}/{best['entry_rating_t2']:.0f}  "
            f"brier={best['brier']:.5f}  ({best['matches_scored']} T1 matches)"
        )

    payload = {
        "tunedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "objective": (
            "one-step-ahead Brier on top-flight matches (draw = 0.5), "
            "two-division pool replay"
        ),
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
