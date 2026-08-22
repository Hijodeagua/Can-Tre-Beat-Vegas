"""
Rest-of-season Monte Carlo per league: title, top-4 and relegation odds
plus expected points, from N replays of the remaining fixtures.

Each sim carries its own copy of the league's ratings and updates them
live with the engine's tuned K / MOV rules as sampled results come in, so
a club that starts hot in a sim keeps mattering in that sim (same posture
as the MLB futures sim). Match scores are sampled from the independent-
Poisson grid parameterized by the live Elo expectation, which bakes W/D/L
and margin into one draw.

A league with no published fixtures for the current season (Ligue 1 until
its upstream repo catches up) is skipped and reported as such.
"""

import numpy as np
import pandas as pd

from soccer.clubs.daily import scoring
from soccer.clubs.daily.config import (
    MAX_GOALS,
    RELEGATION_SPOTS,
    SEASON_SIMS,
    UCL_SPOTS,
)
from soccer.clubs.daily.state import DailyState
from soccer.clubs.model.elo import expected_score, mov_multiplier


def _current_table(results: pd.DataFrame, league: str, season: str) -> dict[str, int]:
    """Points already on the board this season."""
    played = results.dropna(subset=["home_score", "away_score"])
    sub = played[(played["league"] == league) & (played["season"] == season)]
    pts: dict[str, int] = {}
    for r in sub.itertuples():
        hs, as_ = int(r.home_score), int(r.away_score)
        pts[r.home_team] = pts.get(r.home_team, 0) + (3 if hs > as_ else 1 if hs == as_ else 0)
        pts[r.away_team] = pts.get(r.away_team, 0) + (3 if as_ > hs else 1 if hs == as_ else 0)
    return pts


def simulate_league(state: DailyState, league: str, season: str,
                    n_sims: int = SEASON_SIMS, seed: int | None = 0) -> dict | None:
    results = state.results
    remaining = results[
        results["home_score"].isna()
        & (results["league"] == league)
        & (results["season"] == season)
    ]
    if remaining.empty:
        return None

    engine = state.engines[league]
    base_pts = _current_table(results, league, season)
    clubs = sorted(
        set(remaining["home_team"]) | set(remaining["away_team"]) | set(base_pts)
    )
    fixtures = [(r.home_team, r.away_team) for r in remaining.itertuples()]

    # Ratings enter the season through the engine's own accessor, so a
    # promoted club gets the entry rating and the season rollover has
    # already been applied by the replay if any 2026-27 match was played.
    start_ratings = {c: engine.get(c) for c in clubs}
    if engine.current_season != season:
        start_ratings = {
            c: r + engine.season_regression * (engine.base - r)
            for c, r in start_ratings.items()
        }

    rng = np.random.default_rng(seed)
    titles = {c: 0 for c in clubs}
    top4 = {c: 0 for c in clubs}
    releg = {c: 0 for c in clubs}
    pts_sum = {c: 0.0 for c in clubs}

    for _ in range(n_sims):
        ratings = dict(start_ratings)
        pts = dict.fromkeys(clubs, 0)
        for c, p in base_pts.items():
            pts[c] = p
        for home, away in fixtures:
            exp = expected_score(ratings[home] + engine.home_advantage, ratings[away])
            lam_h, lam_a = state.score_params.lambdas(league, exp)
            hs = min(int(rng.poisson(lam_h)), MAX_GOALS)
            as_ = min(int(rng.poisson(lam_a)), MAX_GOALS)
            if hs > as_:
                pts[home] += 3
            elif hs < as_:
                pts[away] += 3
            else:
                pts[home] += 1
                pts[away] += 1
            actual = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
            delta = engine.k * mov_multiplier(hs - as_) * (actual - exp)
            ratings[home] += delta
            ratings[away] -= delta

        # Ties broken uniformly at random (goal-difference tiebreaks are
        # not modeled): jitter far below 1 point.
        order = sorted(clubs, key=lambda c: pts[c] + rng.random() * 1e-6, reverse=True)
        titles[order[0]] += 1
        for c in order[:UCL_SPOTS]:
            top4[c] += 1
        for c in order[-RELEGATION_SPOTS:]:
            releg[c] += 1
        for c in clubs:
            pts_sum[c] += pts[c]

    table = sorted(clubs, key=lambda c: pts_sum[c], reverse=True)
    return {
        "season": season,
        "sims": n_sims,
        "remaining_matches": len(fixtures),
        "clubs": [
            {
                "team": c,
                "elo": round(start_ratings[c], 1),
                "points": base_pts.get(c, 0),
                "exp_points": round(pts_sum[c] / n_sims, 1),
                "p_title": round(titles[c] / n_sims, 4),
                "p_top4": round(top4[c] / n_sims, 4),
                "p_relegation": round(releg[c] / n_sims, 4),
            }
            for c in table
        ],
    }
