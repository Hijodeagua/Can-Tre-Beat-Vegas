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

Alongside the odds the sim reports a `projection` block: every club's
live in-sim Elo at a handful of checkpoint dates across the remaining
fixtures, read three ways — a median simulated season, the 10th/90th
percentile band, and a few whole simulated seasons (`_projection_block`).

What is deliberately absent is the mean. An Elo update is
K * (actual - expected) and the sim draws results at its own expected
rate, so every club's *expected* rating change is about zero: averaging
30,000 seasons returns today's rating for everybody, however far single
seasons swing. A chart led by that average says the table never changes,
which is the one thing the odds in this same payload rule out.

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


def _projection_block(snaps: np.ndarray, checkpoints: list[str],
                      index: dict[str, int], n_samples: int) -> dict:
    """The projection payload, out of a (checkpoints, sims, sides) array of
    live in-sim ratings. Sibling of the same helper in `NFL/daily/` and
    `CFB/daily/simulate.py`.

    Three readings of the same simulations, because none of them tells the
    whole truth alone:

    - `median` — for each side, the one simulated season whose final
      rating is that side's median. A real season, so it moves the way a
      season moves; picked per side, so two median lines are *not* the
      same simulated season.
    - `band` — the 10th/90th percentile at each checkpoint. Where a side
      could plausibly be, and by the end of a season it is wider than the
      gaps between the sides.
    - `samples` — the first `n_samples` sims, whole. Path p of every side
      comes from the same simulated season, so these crossings are a
      coherent league rather than unrelated draws.

    The mean is deliberately not here. A fair game's expected Elo change
    is about zero, so averaging the sims returns today's rating for
    everyone — the one season in which nothing happens, which is the one
    thing the simulation does not predict.
    """
    if not checkpoints:
        # Nothing left to project (every remaining fixture is already
        # past the run date); the export drops the block on this.
        return {"dates": [], "median": {}, "band": {}, "samples": {}}
    n_sims = snaps.shape[1]
    mid = n_sims // 2
    # Median *run*, not the pointwise median: rank the sims by where each
    # side ends up and keep the whole season of the one in the middle.
    median_sim = np.argsort(snaps[-1], axis=0)[mid]
    lo, hi = np.percentile(snaps, [10, 90], axis=1)
    n_paths = min(n_samples, n_sims)
    r1 = lambda x: round(float(x), 1)
    return {
        "dates": checkpoints,
        "median": {
            side: [r1(snaps[s, median_sim[j], j]) for s in range(len(checkpoints))]
            for side, j in index.items()
        },
        "band": {
            side: [[r1(lo[s, j]), r1(hi[s, j])] for s in range(len(checkpoints))]
            for side, j in index.items()
        },
        "samples": {
            side: [[r1(snaps[s, p, j]) for s in range(len(checkpoints))]
                   for p in range(n_paths)]
            for side, j in index.items()
        },
    }


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
    return {
        "season": season,
        "sims": n_sims,
        "remaining_matches": len(fixtures),
        "projection": _projection_block(snaps, checkpoints, club_idx,
                                        PROJECTION_SAMPLES),
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
