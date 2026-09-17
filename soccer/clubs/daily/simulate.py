"""
Rest-of-season Monte Carlo per league: title, UCL (top-4), Europa League
(5th-6th) and relegation odds plus expected points and expected final
position, from N replays of the remaining fixtures.

Each sim carries its own copy of the league's ratings and updates them
live with the engine's tuned K / MOV rules as sampled results come in, so
a club that starts hot in a sim keeps mattering in that sim (same posture
as the MLB futures sim). Match scores are sampled from the independent-
Poisson grid parameterized by the live Elo expectation, which bakes W/D/L
and margin into one draw.

Alongside the odds the sim reports a `projection` block: the mean and
10th/90th-percentile Elo of every club at a handful of checkpoint dates
across the remaining fixtures, read straight off the live in-sim ratings,
plus a few individual simulated seasons.

Those individual seasons are the point of exporting anything beyond the
mean. An Elo update is K * (actual - expected) and the sim draws results
at its own expected rate, so every club's *expected* rating change is
about zero: the mean across 30,000 seasons is flat by construction no
matter how far single seasons swing. The percentiles and the sample paths
are what carry the movement, and a chart that draws only the mean would
say the table never changes — which the odds in this same payload
flatly contradict.

A league with no published fixtures for the current season (Ligue 1 until
its upstream repo catches up) is skipped and reported as such.
"""

import numpy as np
import pandas as pd

from soccer.clubs.daily import scoring
from soccer.clubs.daily.config import (
    MAX_GOALS,
    PROJECTION_POINTS,
    PROJECTION_SAMPLES,
    RELEGATION_SPOTS,
    SEASON_SIMS,
    UCL_SPOTS,
    UEL_SPOTS,
)
from soccer.clubs.daily.state import DailyState
from soccer.clubs.data.leagues import pool_of
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


def _checkpoints(dates: list[str], n: int = PROJECTION_POINTS) -> list[str]:
    """Up to `n` evenly spaced dates out of the remaining fixture dates,
    the last one always included — the x positions the projected Elo line
    is drawn through. Fewer than `n` distinct dates left means every one
    of them is a checkpoint."""
    uniq = sorted(set(dates))
    if len(uniq) <= n:
        return uniq
    last = len(uniq) - 1
    return [uniq[i] for i in sorted({round(i * last / (n - 1)) for i in range(n)})]


def simulate_league(state: DailyState, league: str, season: str,
                    n_sims: int = SEASON_SIMS, seed: int | None = 0,
                    as_of: str | None = None) -> dict | None:
    results = state.results
    remaining = results[
        results["home_score"].isna()
        & (results["league"] == league)
        & (results["season"] == season)
    ]
    if remaining.empty:
        return None

    engine = state.engines[pool_of(league)]
    base_pts = _current_table(results, league, season)
    clubs = sorted(
        set(remaining["home_team"]) | set(remaining["away_team"]) | set(base_pts)
    )
    # Date order, so the in-sim ratings evolve in the order the season
    # actually plays out — and so a checkpoint date can be read off a
    # fixture index for the projection snapshots below.
    remaining = remaining.sort_values("date", kind="stable")
    fixtures = [(r.date, r.home_team, r.away_team) for r in remaining.itertuples()]

    # rating_for applies the division-switch blend for clubs promoted or
    # relegated into this league who haven't played in it yet; the season
    # rollover has already been applied by the replay if any current-season
    # match was played.
    start_ratings = {c: engine.rating_for(c, league) for c in clubs}
    if engine.current_season != season:
        start_ratings = {
            c: r + engine.season_regression * (engine.base - r)
            for c, r in start_ratings.items()
        }

    # Projection checkpoints: the fixture index after which each snapshot
    # date's ratings are complete, so one pass over the fixtures fills
    # `snaps[slot, sim, club]` with that sim's live ratings at that date.
    club_idx = {c: i for i, c in enumerate(clubs)}
    # Only dates after the run date are projection checkpoints: a fixture
    # still unplayed on an earlier date (a postponement, or a result that
    # hasn't landed upstream yet) is simulated, but its date is not the
    # future and would draw the projected line backwards. Those fixtures
    # fold into the first real checkpoint instead.
    checkpoints = _checkpoints(
        [d for d, _, _ in fixtures if as_of is None or d > as_of])
    snap_at = {}
    for slot, d in enumerate(checkpoints):
        last = max(i for i, (fd, _, _) in enumerate(fixtures) if fd <= d)
        snap_at[last] = slot
    snaps = np.zeros((len(checkpoints), n_sims, len(clubs)), dtype=np.float32)

    rng = np.random.default_rng(seed)
    titles = {c: 0 for c in clubs}
    top4 = {c: 0 for c in clubs}
    uel = {c: 0 for c in clubs}
    releg = {c: 0 for c in clubs}
    pts_sum = {c: 0.0 for c in clubs}
    pos_sum = {c: 0.0 for c in clubs}

    for sim in range(n_sims):
        ratings = dict(start_ratings)
        pts = dict.fromkeys(clubs, 0)
        for c, p in base_pts.items():
            pts[c] = p
        for i, (_, home, away) in enumerate(fixtures):
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
            slot = snap_at.get(i)
            if slot is not None:
                # One row assignment per checkpoint, not one per club:
                # 30k sims x 8 checkpoints x 20 clubs of scalar numpy
                # writes would cost more than the rest of the sim.
                snaps[slot, sim] = [ratings[c] for c in clubs]

        # Ties broken uniformly at random (goal-difference tiebreaks are
        # not modeled): jitter far below 1 point.
        order = sorted(clubs, key=lambda c: pts[c] + rng.random() * 1e-6, reverse=True)
        titles[order[0]] += 1
        for c in order[:UCL_SPOTS]:
            top4[c] += 1
        for c in order[UCL_SPOTS:UCL_SPOTS + UEL_SPOTS]:
            uel[c] += 1
        for c in order[-RELEGATION_SPOTS:]:
            releg[c] += 1
        for c in clubs:
            pts_sum[c] += pts[c]
        for i, c in enumerate(order):
            pos_sum[c] += i + 1

    table = sorted(clubs, key=lambda c: pos_sum[c])
    # Mean and 10th/90th-percentile Elo per club per checkpoint — the
    # projected line and the band around it.
    mean = snaps.mean(axis=1)
    lo, hi = np.percentile(snaps, [10, 90], axis=1)
    # The first PROJECTION_SAMPLES sims, whole: sims[p][club] is one
    # club's rating through one simulated season, and the same p across
    # clubs is the same season.
    n_paths = min(PROJECTION_SAMPLES, n_sims)
    return {
        "season": season,
        "sims": n_sims,
        "remaining_matches": len(fixtures),
        "projection": {
            "dates": checkpoints,
            "clubs": {
                c: [[round(float(mean[s, j]), 1), round(float(lo[s, j]), 1),
                     round(float(hi[s, j]), 1)] for s in range(len(checkpoints))]
                for c, j in club_idx.items()
            },
            "samples": {
                c: [[round(float(snaps[s, p, j]), 1) for s in range(len(checkpoints))]
                    for p in range(n_paths)]
                for c, j in club_idx.items()
            },
        },
        "clubs": [
            {
                "team": c,
                "elo": round(start_ratings[c], 1),
                "points": base_pts.get(c, 0),
                "exp_points": round(pts_sum[c] / n_sims, 1),
                "exp_position": round(pos_sum[c] / n_sims, 1),
                "p_title": round(titles[c] / n_sims, 4),
                "p_top4": round(top4[c] / n_sims, 4),
                "p_uel": round(uel[c] / n_sims, 4),
                "p_relegation": round(releg[c] / n_sims, 4),
            }
            for c in table
        ],
    }
