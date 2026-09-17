"""Shared per-run state for the daily pipeline: one Elo replay over the
full 2001-present spine plus the in-run score-model fits, built once and
passed to every step — the CFB twin of `soccer/clubs/daily/state.py` and
`mlb/daily/ratings.py`.

Deterministic by construction: there is no incremental rating state on
disk that can drift or double-count a game; every run replays ~20k games
in well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from CFB.daily import scoring
from CFB.data.teams import FBS
from CFB.model import advanced
from CFB.model.elo import CfbEloEngine, expected_score, load_games, replay
from common.freshness import FreshnessReport, gate


@dataclass
class DailyState:
    engine: CfbEloEngine
    history: pd.DataFrame        # per-game pre-rating records (played games)
    games: pd.DataFrame          # the full spine incl. unplayed fixtures
    score_params: scoring.ScoreParams
    rates: scoring.TeamRates
    season: int                  # the season the engine has rolled into
    # Elo + SportsDataverse efficiency (CFB/model/advanced.py). None when
    # the weekly table is missing or stale: every forecast is then Elo.
    second_stage: "advanced.SecondStage | None" = None
    feeds: dict[str, FreshnessReport] = field(default_factory=dict)

    def feature_row(self, home: str, away: str, home_division: str = FBS,
                    away_division: str = FBS, neutral: bool = False,
                    season: int | None = None, week: int | None = None,
                    season_type: str = "regular",
                    home_id=None, away_id=None) -> dict:
        """Pre-game features for one fixture from current ratings.
        `p_home` is the shipped probability: the second stage when it is
        on and both sides have a strength row for (season, week), else
        Elo. `p_home_elo` is always the Elo number; `model` names which
        one `p_home` is."""
        r_home, r_away, p_elo = self.engine.pregame(
            home, away, home_division, away_division, neutral)
        p_home, model = p_elo, "elo"
        if self.second_stage is not None and season is not None and week is not None:
            p2 = self.second_stage.p_home(home_id, away_id, p_elo, int(season), int(week),
                                          postseason=(season_type == "postseason"))
            if p2 is not None:
                p_home, model = p2, "elo+adj_epa"
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
            "p_home_elo": p_elo, "model": model,
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
    stage, feeds = build_second_stage(games, history, run_date)
    return DailyState(
        engine=engine,
        history=history,
        games=games,
        score_params=scoring.fit(history),
        rates=scoring.rates_from_games(games),
        season=season,
        second_stage=stage,
        feeds=feeds,
    )


def build_second_stage(games: pd.DataFrame, history: pd.DataFrame,
                       run_date: str | None = None):
    """The Elo + efficiency stage, or None with the reason logged. Off
    when `data/college_football/team_weeks.csv` is missing or its last
    real snapshot is more than TOLERANCE_WEEKS behind the spine's last
    completed regular-season week. A backdated run only sees snapshots
    through the last week completed before the run date, same as the
    spine (`as_of` has already masked later results)."""
    tw = advanced.load_team_weeks()
    if run_date and len(tw):
        played = games[games["completed"].astype(bool) & games["season_type"].eq("regular")]
        if len(played):
            season = int(played["season"].max())
            last_week = int(played.loc[played["season"] == season, "week"].max())
            tw = tw[(tw["season"] < season) | (tw["through_week"] <= last_week)]
    report = advanced.coverage_report(tw, games)
    feeds = {"weekly_summaries": report}
    if not gate(report):
        print("   second stage OFF: forecasts are Elo only")
        return None, feeds
    try:
        hist = history.merge(games[["game_id", "home_id", "away_id"]], on="game_id", how="left") \
            if "home_id" not in history.columns else history
        stage = advanced.SecondStage(tw, hist[hist["season"] >= 2004])
    except Exception as exc:          # a broken feed must not kill the run
        print(f"   second stage OFF ({exc!r}): forecasts are Elo only")
        return None, feeds
    print(f"   second stage ON: Elo + {', '.join(stage.features[1:])}, fit on "
          f"{stage.n_train} games {stage.seasons[0]}-{stage.seasons[1]}")
    return stage, feeds


def win_probability(state: DailyState, home: str, away: str,
                    neutral: bool = False) -> float:
    e = state.engine
    adv = 0.0 if neutral else e.home_advantage
    return expected_score(e.rating_for(home) + adv, e.rating_for(away))
