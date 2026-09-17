"""Shared per-run state for the NFL daily pipeline: one Elo replay over
the full 1999-present nflverse spine plus the in-run score-model fits,
built once and passed to every step — the NFL twin of
`CFB/daily/state.py`.

Deterministic by construction: no incremental rating state on disk that
can drift or double-count a game; every run replays ~7,000 games in
well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from common.freshness import FreshnessReport, gate
from NFL.daily import scoring
from NFL.elo.engine import NflEloEngine, load_games, replay
from NFL.elo.teams import TEAMS
from NFL.model import advanced


@dataclass
class DailyState:
    engine: NflEloEngine
    history: pd.DataFrame        # per-game pre-rating records (played games)
    games: pd.DataFrame          # the full spine incl. unplayed fixtures
    score_params: scoring.ScoreParams
    rates: scoring.TeamRates
    season: int                  # the season the engine has rolled into
    # Elo + adjusted success rate (NFL/model/advanced.py). None when the
    # aggregates are missing or stale: every forecast is then Elo alone.
    second_stage: "advanced.SecondStage | None" = None
    feeds: dict[str, FreshnessReport] = field(default_factory=dict)

    def feature_row(self, home: str, away: str, neutral: bool = False,
                    home_rest: float | None = None,
                    away_rest: float | None = None,
                    season: int | None = None, week: int | None = None) -> dict:
        """Pre-game features for one fixture from current ratings.
        `p_home` is the shipped probability: the second stage when it is
        on and both sides have a rating for (season, week), else Elo.
        `p_home_elo` is always the Elo number; `model` says which one
        `p_home` is."""
        r_home, r_away, p_elo = self.engine.pregame(
            home, away, neutral, home_rest, away_rest)
        p_home, model = p_elo, "elo"
        if self.second_stage is not None and season is not None and week is not None:
            p2 = self.second_stage.p_home(home, away, p_elo, int(season), int(week))
            if p2 is not None:
                p_home, model = p2, "elo+success"
        h_adj, a_adj = self.engine.edges(neutral, home_rest, away_rest)
        elo_diff = (r_home + h_adj) - (r_away + a_adj)
        total = self.rates.matchup_total(home, away)
        pred_home, pred_away = self.score_params.expected_score(elo_diff, total)
        return {
            "home_team": home, "away_team": away,
            "elo_home_pre": r_home, "elo_away_pre": r_away,
            "elo_diff": elo_diff, "p_home": p_home,
            "p_home_elo": p_elo, "model": model,
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
    stage, feeds = build_second_stage(games, history, run_date)
    return DailyState(
        engine=engine,
        history=history,
        games=games,
        score_params=scoring.fit(history, engine.margin_cap),
        rates=scoring.rates_from_games(games),
        season=season,
        second_stage=stage,
        feeds=feeds,
    )


def build_second_stage(games: pd.DataFrame, history: pd.DataFrame,
                       run_date: str | None = None):
    """The Elo + adjusted-success stage, or None with the reason logged.
    Off when the aggregates file is missing or has fallen behind the
    spine's completed games (a stale feed is worse than none). Backdated
    runs only see aggregates dated before the run date, same as the
    spine."""
    tg = advanced.load_team_games()
    if run_date and len(tg):
        tg = tg[tg["date"] < run_date]
    report = advanced.aggregates_freshness(tg, games)
    feeds = {"pbp_aggregates": report}
    if not gate(report):
        print("   second stage OFF: forecasts are Elo only")
        return None, feeds
    try:
        stage = advanced.SecondStage(tg, history)
    except Exception as exc:          # a broken feed must not kill the run
        print(f"   second stage OFF ({exc!r}): forecasts are Elo only")
        return None, feeds
    print(f"   second stage ON: Elo + adjusted success, fit on {stage.n_train} games "
          f"{stage.seasons[0]}-{stage.seasons[1]}")
    return stage, feeds
