"""Rest-of-season Monte Carlo: expected wins, bowl eligibility, an
undefeated regular season, a conference-championship-game berth and the
conference title, from N vectorized replays of the remaining schedule
with live in-sim Elo — the CFB sibling of `mlb/daily/simulate.py`'s
season sim and `soccer/clubs/daily/simulate.py`.

Mechanics
---------
- Every remaining regular-season game on the spine is played with the
  Elo win probability from the sim's own current ratings (K updates, no
  margin term since scores are not simulated), vectorized across sims.
  FCS opponents are the pooled fixed rating and never update.
- Conference standings are conference-game win percentage, ties broken
  uniformly at random (the real multi-team tiebreakers are not modeled).
- The conference championship game is the top two by that standing at a
  neutral site — unless the spine already carries a scheduled CCG for the
  conference (they appear in December), in which case that fixture is
  used, and once it has been played the result is final. Independents
  have no CCG or title.
- Bowl eligibility is BOWL_ELIGIBLE_WINS total wins (the one-FCS-win rule
  and APR exceptions are not modeled).

What is deliberately NOT here: the 12-team playoff field. Selection is a
committee ranking, and inventing one would put a made-up number next to
the honest ones. Elo rank and the conference title odds are the inputs a
reader can combine themselves.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from CFB.daily.config import BOWL_ELIGIBLE_WINS, SEASON_SIMS
from CFB.daily.state import DailyState
from CFB.data.teams import FBS
from CFB.model.elo import INDEPENDENT

CCG_RE = re.compile(r"championship", re.IGNORECASE)


def _is_ccg(row) -> bool:
    return isinstance(row.notes, str) and bool(CCG_RE.search(row.notes)) \
        and row.season_type == "regular"


def current_records(games: pd.DataFrame, season: int, teams: list[str]) -> pd.DataFrame:
    """Season-to-date W-L, conference W-L and point differential per FBS
    program (regular season, CCG excluded from the conference record)."""
    played = games[games["completed"].astype(bool) & (games["season"] == season)]
    rec = {t: {"wins": 0, "losses": 0, "conf_wins": 0, "conf_losses": 0, "pts_diff": 0}
           for t in teams}
    for r in played.itertuples(index=False):
        if r.season_type != "regular":
            continue
        home_won = r.home_points > r.away_points
        margin = int(r.home_points - r.away_points)
        conf = bool(r.conference_game) and not _is_ccg(r)
        for team, won, m in ((r.home_team, home_won, margin),
                             (r.away_team, not home_won, -margin)):
            if team not in rec:
                continue
            rec[team]["wins" if won else "losses"] += 1
            rec[team]["pts_diff"] += m
            if conf:
                rec[team]["conf_wins" if won else "conf_losses"] += 1
    out = pd.DataFrame.from_dict(rec, orient="index")
    out.index.name = "team"
    return out.reset_index()


def simulate_season(state: DailyState, season: int | None = None,
                    n_sims: int = SEASON_SIMS, seed: int | None = 0) -> dict | None:
    season = season or state.season
    games = state.games
    teams = state.fbs_teams()
    if not teams:
        return None
    engine = state.engine
    idx = {t: i for i, t in enumerate(teams)}
    conf_of = dict(engine.conference)

    season_games = games[games["season"] == season]
    remaining = season_games[~season_games["completed"].astype(bool)
                             & (season_games["season_type"] == "regular")]
    records = current_records(games, season, teams).set_index("team")

    rng = np.random.default_rng(seed)
    n = len(teams)
    R = np.tile(np.array([engine.rating_for(t) for t in teams], dtype=float), (n_sims, 1))
    wins = np.tile(records.loc[teams, "wins"].to_numpy(dtype=float), (n_sims, 1))
    losses = np.tile(records.loc[teams, "losses"].to_numpy(dtype=float), (n_sims, 1))
    cwins = np.tile(records.loc[teams, "conf_wins"].to_numpy(dtype=float), (n_sims, 1))
    closs = np.tile(records.loc[teams, "conf_losses"].to_numpy(dtype=float), (n_sims, 1))

    # Scheduled (unplayed) CCGs are simulated after the regular season, with
    # their real participants; played ones are final.
    scheduled_ccg: dict[str, tuple[str, str]] = {}
    played_ccg: dict[str, tuple[str, str]] = {}   # conf -> (winner, loser)
    for r in season_games.itertuples(index=False):
        if not _is_ccg(r):
            continue
        conf = conf_of.get(r.home_team) or conf_of.get(r.away_team)
        if not conf:
            continue
        if bool(r.completed):
            w, l = ((r.home_team, r.away_team) if r.home_points > r.away_points
                    else (r.away_team, r.home_team))
            played_ccg[conf] = (w, l)
        else:
            scheduled_ccg[conf] = (r.home_team, r.away_team)

    hfa = engine.home_advantage
    k = engine.k
    fcs = engine.fcs_rating
    games_left = np.zeros(n)
    for r in remaining.sort_values(["start_utc", "game_id"]).itertuples(index=False):
        if _is_ccg(r):
            continue
        h = idx.get(r.home_team) if r.home_division == FBS else None
        a = idx.get(r.away_team) if r.away_division == FBS else None
        if h is None and a is None:
            continue
        adv = 0.0 if bool(r.neutral_site) else hfa
        r_h = R[:, h] if h is not None else fcs
        r_a = R[:, a] if a is not None else fcs
        p = 1.0 / (1.0 + 10 ** (-((r_h + adv) - r_a) / 400.0))
        home_won = rng.random(n_sims) < p
        conf_game = bool(r.conference_game)
        delta = k * (home_won - p)
        if h is not None:
            wins[:, h] += home_won
            losses[:, h] += ~home_won
            R[:, h] += delta
            games_left[h] += 1
            if conf_game:
                cwins[:, h] += home_won
                closs[:, h] += ~home_won
        if a is not None:
            wins[:, a] += ~home_won
            losses[:, a] += home_won
            R[:, a] -= delta
            games_left[a] += 1
            if conf_game:
                cwins[:, a] += ~home_won
                closs[:, a] += home_won

    # Conference standings -> CCG -> champion.
    ccg = np.zeros((n_sims, n), dtype=bool)
    champ = np.zeros((n_sims, n), dtype=bool)
    conferences = sorted({c for c in conf_of.values() if c != INDEPENDENT})
    for conf in conferences:
        members = np.array([idx[t] for t in teams if conf_of.get(t) == conf])
        if len(members) < 2:
            continue
        if conf in played_ccg:
            w, l = played_ccg[conf]
            for t in (w, l):
                if t in idx:
                    ccg[:, idx[t]] = True
            if w in idx:
                champ[:, idx[w]] = True
            continue
        if conf in scheduled_ccg:
            t1, t2 = scheduled_ccg[conf]
            if t1 not in idx or t2 not in idx:
                continue
            i1, i2 = idx[t1], idx[t2]
            ccg[:, i1] = ccg[:, i2] = True
            p1 = 1.0 / (1.0 + 10 ** (-(R[:, i1] - R[:, i2]) / 400.0))
            won = rng.random(n_sims) < p1
            champ[:, i1] = won
            champ[:, i2] = ~won
            continue
        cg = cwins[:, members] + closs[:, members]
        pct = np.where(cg > 0, cwins[:, members] / np.maximum(cg, 1), 0.0)
        noisy = pct + rng.random(pct.shape) * 1e-6
        order = np.argsort(-noisy, axis=1)
        top1 = members[order[:, 0]]
        top2 = members[order[:, 1]]
        rows = np.arange(n_sims)
        ccg[rows, top1] = True
        ccg[rows, top2] = True
        p1 = 1.0 / (1.0 + 10 ** (-(R[rows, top1] - R[rows, top2]) / 400.0))
        won = rng.random(n_sims) < p1
        champ[rows, np.where(won, top1, top2)] = True

    rows_out = []
    for t in teams:
        i = idx[t]
        conf = conf_of.get(t)
        indep = conf in (None, INDEPENDENT)
        rows_out.append({
            "team": t,
            "conference": conf,
            "elo": round(float(engine.rating_for(t)), 1),
            "wins": int(records.loc[t, "wins"]),
            "losses": int(records.loc[t, "losses"]),
            "conf_wins": int(records.loc[t, "conf_wins"]),
            "conf_losses": int(records.loc[t, "conf_losses"]),
            "pts_diff": int(records.loc[t, "pts_diff"]),
            "games_left": int(games_left[i]),
            "exp_wins": round(float(wins[:, i].mean()), 1),
            "exp_losses": round(float(losses[:, i].mean()), 1),
            "p_bowl": round(float((wins[:, i] >= BOWL_ELIGIBLE_WINS).mean()), 4),
            "p_undefeated": round(float((losses[:, i] == 0).mean()), 4),
            "p_ccg": None if indep else round(float(ccg[:, i].mean()), 4),
            "p_conf_title": None if indep else round(float(champ[:, i].mean()), 4),
        })
    table = sorted(rows_out, key=lambda r: (-r["exp_wins"], -r["elo"]))
    return {
        "season": season,
        "sims": n_sims,
        "remaining_games": int(len(remaining)),
        "teams": table,
    }
