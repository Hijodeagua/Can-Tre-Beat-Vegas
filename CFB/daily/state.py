"""Shared per-run state for the daily pipeline: one Elo replay over the
full 2001-present spine plus the in-run score-model fits, built once and
passed to every step — the CFB twin of `soccer/clubs/daily/state.py` and
`mlb/daily/ratings.py`.

Deterministic by construction: there is no incremental rating state on
disk that can drift or double-count a game; every run replays ~20k games
in well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from CFB.daily import scoring
from CFB.data.teams import FBS
from CFB.model.elo import CfbEloEngine, expected_score, load_games, replay


@dataclass
class DailyState:
    engine: CfbEloEngine
    history: pd.DataFrame        # per-game pre-rating records (played games)
    games: pd.DataFrame          # the full spine incl. unplayed fixtures
    score_params: scoring.ScoreParams
    rates: scoring.TeamRates
    season: int                  # the season the engine has rolled into

    def feature_row(self, home: str, away: str, home_division: str = FBS,
                    away_division: str = FBS, neutral: bool = False) -> dict:
        """Pre-game features for one fixture from current ratings."""
        r_home, r_away, p_home = self.engine.pregame(
            home, away, home_division, away_division, neutral)
        adv = 0.0 if neutral else self.engine.home_advantage
        elo_diff = (r_home + adv) - r_away
        total = self.rates.matchup_total(
            home if home_division == FBS else "FCS",
            away if away_division == FBS else "FCS")
        pred_home, pred_away = self.score_params.expected_score(elo_diff, total)
        return {
            "home_team": home, "away_team": away,
            "elo_home_pre": r_home, "elo_away_pre": r_away,
            "elo_diff": elo_diff, "p_home": p_home,
            "pred_total": round(total, 1),
            "pred_home_score": pred_home, "pred_away_score": pred_away,
        }

    def fbs_teams(self) -> list[str]:
        """Every FBS program in the current season's conference map."""
        return sorted(self.engine.conference)


def as_of(games: pd.DataFrame, run_date: str) -> pd.DataFrame:
    """The spine as it looked on the morning of `run_date`: every game
    dated on or after it is unplayed. A live run at 10:00 UTC never has a
    final for today's date, so this is a no-op live and makes a backdated
    run (`--date` in the past) reproduce exactly what that morning's run
    would have predicted, graded and simulated — no hindsight anywhere."""
    g = games.copy()
    future = g["date"] >= run_date
    g.loc[future, "completed"] = False
    g.loc[future, ["home_points", "away_points"]] = pd.NA
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
        # into it now so the preseason board and slate use regressed
        # ratings and the new season's conference map.
        from CFB.model.elo import season_conferences
        engine.roll_season(season, season_conferences(games).get(season, {}))
    return DailyState(
        engine=engine,
        history=history,
        games=games,
        score_params=scoring.fit(history),
        rates=scoring.rates_from_games(games),
        season=season,
    )


def win_probability(state: DailyState, home: str, away: str,
                    neutral: bool = False) -> float:
    e = state.engine
    adv = 0.0 if neutral else e.home_advantage
    return expected_score(e.rating_for(home) + adv, e.rating_for(away))
