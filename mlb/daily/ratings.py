"""Current Elo ratings and season-to-date standings.

Ratings are produced by replaying the full 2009-present game file through the
tuned Elo engine (mlb/elo.py) on every run. The replay covers ~41k games and
takes a couple of seconds, which buys determinism: there is no incremental
rating state on disk that can drift or double-count a game.

Standings (wins, losses, run differential) are *rolling season-to-date* —
computed only from games completed so far, never from a full-season total.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlb.elo import GAMES_CSV, run_history
from mlb.daily.config import ALL_TEAMS, CURRENT_SEASON


@dataclass
class ModelState:
    ratings: dict[str, float]          # franchise -> current Elo
    standings: pd.DataFrame            # team, wins, losses, run_diff (season-to-date)
    history: pd.DataFrame              # per-game Elo history (p_home, run_diff, ...)
    games: pd.DataFrame                # full games file


def season_standings(games: pd.DataFrame, season: int = CURRENT_SEASON) -> pd.DataFrame:
    g = games[games.season == season]
    rows = []
    for t in ALL_TEAMS:
        home = g[g.home_fr == t]
        away = g[g.away_fr == t]
        wins = int((home.home_score > home.away_score).sum()
                   + (away.away_score > away.home_score).sum())
        losses = len(home) + len(away) - wins
        run_diff = int((home.home_score - home.away_score).sum()
                       + (away.away_score - away.home_score).sum())
        rows.append({"team": t, "wins": wins, "losses": losses, "run_diff": run_diff})
    return pd.DataFrame(rows)


def build_state() -> ModelState:
    engine, history, _ = run_history()
    games = pd.read_csv(GAMES_CSV)
    return ModelState(
        ratings={t: engine.get(t) for t in ALL_TEAMS},
        standings=season_standings(games),
        history=history,
        games=games,
    )
