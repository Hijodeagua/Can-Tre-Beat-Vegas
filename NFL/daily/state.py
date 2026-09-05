"""Shared per-run state for the NFL daily pipeline: one Elo replay over
the full 1999-present nflverse spine plus the in-run score-model fits,
built once and passed to every step — the NFL twin of
`CFB/daily/state.py`.

Deterministic by construction: no incremental rating state on disk that
can drift or double-count a game; every run replays ~7,000 games in
well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from NFL.daily import scoring
from NFL.elo.engine import NflEloEngine, load_games, replay
from NFL.elo.teams import TEAMS


@dataclass
class DailyState:
    engine: NflEloEngine
    history: pd.DataFrame        # per-game pre-rating records (played games)
    games: pd.DataFrame          # the full spine incl. unplayed fixtures
    score_params: scoring.ScoreParams
    rates: scoring.TeamRates
    season: int                  # the season the engine has rolled into

    def feature_row(self, home: str, away: str, neutral: bool = False,
                    home_rest: float | None = None,
                    away_rest: float | None = None) -> dict:
        """Pre-game features for one fixture from current ratings."""
        r_home, r_away, p_home = self.engine.pregame(
            home, away, neutral, home_rest, away_rest)
        h_adj, a_adj = self.engine.edges(neutral, home_rest, away_rest)
        elo_diff = (r_home + h_adj) - (r_away + a_adj)
        total = self.rates.matchup_total(home, away)
        pred_home, pred_away = self.score_params.expected_score(elo_diff, total)
        return {
            "home_team": home, "away_team": away,
            "elo_home_pre": r_home, "elo_away_pre": r_away,
            "elo_diff": elo_diff, "p_home": p_home,
            "pred_total": round(total, 1),
            "pred_home_score": pred_home, "pred_away_score": pred_away,
            # Model's own line, nflverse sign: positive = home favoured.
            "elo_spread": round(self.score_params.expected_margin(elo_diff), 1),
        }

    def teams(self) -> list[str]:
        return list(TEAMS)


def as_of(games: pd.DataFrame, run_date: str) -> pd.DataFrame:
    """The spine as it looked on the morning of `run_date`: every game
    dated on or after it is unplayed. Live this is a no-op (a 10:00 UTC
    run never has a final for today's date); backdated, it makes `--date`
    reproduce exactly what that morning's run would have produced."""
    g = games.copy()
    future = g["date"] >= run_date
    g.loc[future, "completed"] = False
    g.loc[future, ["home_score", "away_score"]] = float("nan")
    g["completed"] = g["completed"].astype(bool)
    return g


def build_state(games: pd.DataFrame | None = None,
                run_date: str | None = None) -> DailyState:
    games = load_games() if games is None else games
    if run_date:
        games = as_of(games, run_date)
    engine, history = replay(games)
    season = int(games["season"].max())
    if engine.current_season != season:
        # No game of the new season has been played yet: roll the ratings
        # into it now so the preseason board and slate use regressed ratings.
        engine.roll_season(season)
    return DailyState(
        engine=engine,
        history=history,
        games=games,
        score_params=scoring.fit(history, engine.margin_cap),
        rates=scoring.rates_from_games(games),
        season=season,
    )
