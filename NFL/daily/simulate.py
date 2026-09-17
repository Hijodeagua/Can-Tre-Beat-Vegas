"""Rest-of-season Monte Carlo: expected wins, the division, a playoff
berth, the #1 seed (and its bye), the conference title and the Super
Bowl, from N vectorized replays of the remaining schedule and the full
seven-team-per-conference bracket with live in-sim Elo — the NFL sibling
of `CFB/daily/simulate.py`.

The college sim stops before its playoff because a committee picks the
field. The NFL's is a rulebook, so it is played out here:

- Every remaining regular-season game on the spine is played with the
  Elo win probability from the sim's own current ratings (home edge,
  neutral sites and the bye-week rest bonus from the spine's rest
  columns), vectorized across sims. Ratings move by K times the fitted
  mean margin multiplier, since scores are not simulated. Ties are not
  simulated (one a season).
- Standings: win percentage (ties half). Division titles break ties on
  division record, wild cards and seeding on conference record, and
  anything still level is uniform random — the real ladder (head-to-head,
  common games, strength of victory...) is not modeled.
- Seeds 1-4 are the division winners by record, 5-7 the best three
  others. Wild Card: 2v7, 3v6, 4v5 at the higher seed; Divisional: the
  #1 seed (off its bye, so it carries the rest bonus) meets the lowest
  survivor; Championship at the higher seed; Super Bowl neutral.
- A playoff game already on the spine as final is honoured: whenever the
  simulated bracket produces that matchup, the real result is used.

Alongside the odds the sim reports a `projection` block: every team's
live in-sim Elo at a handful of checkpoint dates across the remaining
regular season, read three ways — a median simulated season, the
10th/90th percentile band, and a few whole simulated seasons
(`_projection_block`).

What is deliberately absent is the mean. An Elo update is
K * (actual - expected) and the sim draws results at its own expected
rate, so every team's *expected* rating change is about zero: averaging
the sims returns today's rating for everybody, however far single seasons
swing. A chart led by that average says the board never changes, which is
the one thing the odds in this same payload rule out. The bracket is left
out of the projection: the postseason is not every team's season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from NFL.daily.config import PROJECTION_POINTS, PROJECTION_SAMPLES, SEASON_SIMS
from NFL.daily.state import DailyState
from NFL.elo.engine import REST_BONUS_DAYS
from NFL.elo.teams import (
    CONFERENCE_OF, CONFERENCES, DIVISION_OF, DIVISIONS, TEAMS, name,
)

PLAYOFF_TEAMS = 7


def current_records(games: pd.DataFrame, season: int,
                    teams: list[str] | tuple[str, ...] = TEAMS) -> pd.DataFrame:
    """Season-to-date W-L-T, division and conference W-L-T and point
    differential per team (regular season only)."""
    played = games[games["completed"].astype(bool) & (games["season"] == season)
                   & (games["game_type"] == "REG")]
    keys = ("wins", "losses", "ties", "div_wins", "div_losses", "div_ties",
            "conf_wins", "conf_losses", "conf_ties", "pts_diff")
    rec = {t: {k: 0 for k in keys} for t in teams}
    for r in played.itertuples(index=False):
        margin = int(r.home_score - r.away_score)
        div = DIVISION_OF.get(r.home_team) == DIVISION_OF.get(r.away_team)
        conf = CONFERENCE_OF.get(r.home_team) == CONFERENCE_OF.get(r.away_team)
        for team, m in ((r.home_team, margin), (r.away_team, -margin)):
            if team not in rec:
                continue
            res = "wins" if m > 0 else ("losses" if m < 0 else "ties")
            rec[team][res] += 1
            rec[team]["pts_diff"] += m
            if div:
                rec[team]["div_" + res] += 1
            if conf:
                rec[team]["conf_" + res] += 1
    out = pd.DataFrame.from_dict(rec, orient="index")
    out.index.name = "team"
    return out.reset_index()


def _checkpoints(dates: list[str], n: int = PROJECTION_POINTS) -> list[str]:
    """Up to `n` evenly spaced dates out of the remaining game dates, the
    last one always included — the x positions the projected Elo line is
    drawn through. Fewer than `n` distinct dates left means every one of
    them is a checkpoint."""
    uniq = sorted(set(dates))
    if len(uniq) <= n:
        return uniq
    last = len(uniq) - 1
    return [uniq[i] for i in sorted({round(i * last / (n - 1)) for i in range(n)})]


def _projection_block(snaps: np.ndarray, checkpoints: list[str],
                      index: dict[str, int], n_samples: int) -> dict:
    """The projection payload, out of a (checkpoints, sims, sides) array of
    live in-sim ratings. Sibling of the same helper in the other two
    daily pipelines.

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
      coherent board rather than unrelated draws.

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


def _pct(w: np.ndarray, l: np.ndarray, t: np.ndarray) -> np.ndarray:
    g = w + l + t
    return np.where(g > 0, (w + 0.5 * t) / np.maximum(g, 1), 0.0)


def simulate_season(state: DailyState, season: int | None = None,
                    n_sims: int = SEASON_SIMS, seed: int | None = 0,
                    as_of: str | None = None) -> dict | None:
    season = season or state.season
    games = state.games
    teams = list(TEAMS)
    engine = state.engine
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    season_games = games[games["season"] == season]
    if season_games.empty:
        return None
    remaining = season_games[~season_games["completed"].astype(bool)
                             & (season_games["game_type"] == "REG")]
    records = current_records(games, season, teams).set_index("team").loc[teams]

    rng = np.random.default_rng(seed)
    R = np.tile(np.array([engine.rating_for(t) for t in teams], dtype=float), (n_sims, 1))

    def tile(col):
        return np.tile(records[col].to_numpy(dtype=float), (n_sims, 1))

    wins, losses, ties = tile("wins"), tile("losses"), tile("ties")
    dwins, dloss, dties = tile("div_wins"), tile("div_losses"), tile("div_ties")
    cwins, closs, cties = tile("conf_wins"), tile("conf_losses"), tile("conf_ties")

    hfa = engine.home_advantage
    k_sim = engine.k * state.score_params.mov_mean
    bye = engine.rest_bonus
    games_left = np.zeros(n)

    # Projection checkpoints. Each snapshot date's ratings are complete
    # once the last remaining game on or before it has been played, so the
    # snapshot is taken on the way into the *next* game (and after the loop
    # for the final one) — a position the `continue` below can't skip.
    ordered = remaining.sort_values(["date", "gametime", "game_id"])
    dates = [str(d) for d in ordered["date"]]
    # Only dates after the run date are checkpoints: today's own slate is
    # unplayed, and a game still unplayed on an earlier date would draw
    # the projected line backwards. Both fold into the first real
    # checkpoint instead.
    checkpoints = _checkpoints([d for d in dates if as_of is None or d > as_of])
    snap_before = {max(i for i, d in enumerate(dates) if d <= cp) + 1: slot
                   for slot, cp in enumerate(checkpoints)}
    snaps = np.zeros((len(checkpoints), n_sims, n), dtype=np.float32)

    for i, r in enumerate(ordered.itertuples(index=False)):
        if i in snap_before:
            snaps[snap_before[i]] = R
        h, a = idx.get(r.home_team), idx.get(r.away_team)
        if h is None or a is None:
            continue
        h_adj = 0.0 if bool(r.neutral) else hfa
        if r.home_rest == r.home_rest and r.home_rest >= REST_BONUS_DAYS:
            h_adj += bye
        a_adj = bye if (r.away_rest == r.away_rest and r.away_rest >= REST_BONUS_DAYS) else 0.0
        p = 1.0 / (1.0 + 10 ** (-((R[:, h] + h_adj) - (R[:, a] + a_adj)) / 400.0))
        home_won = rng.random(n_sims) < p
        delta = k_sim * (home_won - p)
        R[:, h] += delta
        R[:, a] -= delta
        wins[:, h] += home_won; losses[:, h] += ~home_won
        wins[:, a] += ~home_won; losses[:, a] += home_won
        games_left[h] += 1; games_left[a] += 1
        if DIVISION_OF.get(r.home_team) == DIVISION_OF.get(r.away_team):
            dwins[:, h] += home_won; dloss[:, h] += ~home_won
            dwins[:, a] += ~home_won; dloss[:, a] += home_won
        if CONFERENCE_OF.get(r.home_team) == CONFERENCE_OF.get(r.away_team):
            cwins[:, h] += home_won; closs[:, h] += ~home_won
            cwins[:, a] += ~home_won; closs[:, a] += home_won
    if len(dates) in snap_before:
        snaps[snap_before[len(dates)]] = R

    # --- standings -> seeds ---------------------------------------------------
    pct = _pct(wins, losses, ties)
    dpct = _pct(dwins, dloss, dties)
    cpct = _pct(cwins, closs, cties)
    rows = np.arange(n_sims)

    div_winner = np.zeros((n_sims, n), dtype=bool)
    for div, members in DIVISIONS.items():
        m = np.array([idx[t] for t in members])
        key = pct[:, m] + 1e-3 * dpct[:, m] + 1e-6 * rng.random((n_sims, len(m)))
        div_winner[rows, m[np.argmax(key, axis=1)]] = True

    seed_of = np.zeros((n_sims, n), dtype=int)            # 1-7, 0 = out
    seed_team: dict[str, np.ndarray] = {}                 # conf -> (n_sims, 7) team idx
    for conf in CONFERENCES:
        m = np.array([idx[t] for t in teams if CONFERENCE_OF[t] == conf])
        key = pct[:, m] + 1e-3 * cpct[:, m] + 1e-6 * rng.random((n_sims, len(m)))
        # Division winners first (by record among themselves), then the rest.
        key = key + 10.0 * div_winner[:, m]
        order = np.argsort(-key, axis=1)[:, :PLAYOFF_TEAMS]
        seeded = m[order]                                 # (n_sims, 7)
        seed_team[conf] = seeded
        for s in range(PLAYOFF_TEAMS):
            seed_of[rows, seeded[:, s]] = s + 1

    # --- the bracket ----------------------------------------------------------
    forced = np.full((n, n), -1, dtype=int)   # forced[i, j] = winner idx, if played
    for r in season_games[season_games["completed"].astype(bool)
                          & (season_games["game_type"] != "REG")].itertuples(index=False):
        h, a = idx.get(r.home_team), idx.get(r.away_team)
        if h is None or a is None or r.home_score == r.away_score:
            continue
        w = h if r.home_score > r.away_score else a
        forced[h, a] = forced[a, h] = w

    def play(home: np.ndarray, away: np.ndarray, neutral: bool = False,
             home_bye: bool = False) -> np.ndarray:
        """One playoff game per sim; returns the winner's team index."""
        h_adj = (0.0 if neutral else hfa) + (bye if home_bye else 0.0)
        p = 1.0 / (1.0 + 10 ** (-((R[rows, home] + h_adj) - R[rows, away]) / 400.0))
        home_won = rng.random(n_sims) < p
        f = forced[home, away]
        known = f >= 0
        home_won = np.where(known, f == home, home_won)
        delta = k_sim * engine.playoff_k_mult * (home_won - p)
        R[rows, home] += delta
        R[rows, away] -= delta
        return np.where(home_won, home, away)

    conf_champ: dict[str, np.ndarray] = {}
    for conf in CONFERENCES:
        st = seed_team[conf]
        s = {i + 1: st[:, i] for i in range(PLAYOFF_TEAMS)}
        # Wild Card round, higher seed hosts.
        w27 = play(s[2], s[7]); w36 = play(s[3], s[6]); w45 = play(s[4], s[5])
        # Divisional: #1 (off the bye) hosts the lowest surviving seed; the
        # other two survivors meet at the better seed.
        survivors = np.stack([w27, w36, w45], axis=1)             # (n_sims, 3)
        surv_seed = seed_of[rows[:, None], survivors]              # their seeds
        order = np.argsort(surv_seed, axis=1)                      # best seed first
        best = survivors[rows, order[:, 0]]
        mid = survivors[rows, order[:, 1]]
        low = survivors[rows, order[:, 2]]
        d1 = play(s[1], low, home_bye=True)
        d2 = play(best, mid)
        # Championship at the better seed.
        d1_seed = seed_of[rows, d1]; d2_seed = seed_of[rows, d2]
        host = np.where(d1_seed <= d2_seed, d1, d2)
        guest = np.where(d1_seed <= d2_seed, d2, d1)
        conf_champ[conf] = play(host, guest)

    sb_winner = play(conf_champ["AFC"], conf_champ["NFC"], neutral=True)

    made = seed_of > 0
    top_seed = seed_of == 1
    champ = np.zeros((n_sims, n), dtype=bool)
    for conf in CONFERENCES:
        champ[rows, conf_champ[conf]] = True
    sb = np.zeros((n_sims, n), dtype=bool)
    sb[rows, sb_winner] = True

    rows_out = []
    for t in teams:
        i = idx[t]
        rows_out.append({
            "team": t,
            "name": name(t),
            "conference": CONFERENCE_OF[t],
            "division": DIVISION_OF[t],
            "elo": round(float(engine.rating_for(t)), 1),
            "wins": int(records.loc[t, "wins"]),
            "losses": int(records.loc[t, "losses"]),
            "ties": int(records.loc[t, "ties"]),
            "pts_diff": int(records.loc[t, "pts_diff"]),
            "games_left": int(games_left[i]),
            "exp_wins": round(float(wins[:, i].mean()), 1),
            "exp_losses": round(float(losses[:, i].mean()), 1),
            "p_division": round(float(div_winner[:, i].mean()), 4),
            "p_playoffs": round(float(made[:, i].mean()), 4),
            "p_top_seed": round(float(top_seed[:, i].mean()), 4),
            "p_conf": round(float(champ[:, i].mean()), 4),
            "p_sb": round(float(sb[:, i].mean()), 4),
        })
    table = sorted(rows_out, key=lambda r: (-r["p_sb"], -r["exp_wins"], -r["elo"]))
    return {
        "season": season,
        "sims": n_sims,
        "remaining_games": int(len(remaining)),
        "teams": table,
        "projection": _projection_block(snaps, checkpoints, idx,
                                        PROJECTION_SAMPLES) if checkpoints else None,
    }
