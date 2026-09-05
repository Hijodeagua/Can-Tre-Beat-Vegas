"""Predict the slate: every unplayed game of the next NFL week, with the
Elo win probability, the pick, the model's own spread and an expected
score.

The slate is persisted to data/nfl/predictions/slate_{D}.csv the day it is
predicted — that file, written before the games were played, is what
`grade.py` later grades. Re-running a date overwrites its slate file, but
grading is idempotent through the ledger (keyed by game id, earliest slate
first), so a regenerated slate can't rewrite the record.
"""

from __future__ import annotations

import pandas as pd

from NFL.daily.config import PREDICTIONS_DIR
from NFL.daily.state import DailyState
from NFL.elo.teams import DIVISION_OF, name, week_label

SLATE_COLUMNS = [
    "game_id", "date", "weekday", "gametime", "season", "week", "game_type", "week_label",
    "home_team", "away_team", "home_name", "away_name",
    "home_division", "away_division", "div_game", "neutral",
    "home_rest", "away_rest",
    "elo_home_pre", "elo_away_pre", "p_home", "pick", "pick_prob",
    "elo_spread", "pred_home_score", "pred_away_score", "pred_total",
]


def next_week(state: DailyState, run_date: str) -> tuple[int, int] | None:
    """(season, week) of the earliest unplayed game dated on or after the
    run date — the week the slate covers."""
    g = state.games
    pending = g[~g["completed"].astype(bool) & (g["date"] >= run_date)]
    if pending.empty:
        return None
    first = pending.sort_values(["season", "week", "date"]).iloc[0]
    return int(first["season"]), int(first["week"])


def upcoming_games(state: DailyState, run_date: str,
                   weeks: int = 1) -> pd.DataFrame:
    """Unplayed games dated on or after the run date in the next `weeks`
    NFL weeks (the current week counts as one)."""
    nw = next_week(state, run_date)
    if nw is None:
        return state.games.iloc[0:0]
    season, week = nw
    g = state.games
    return g[
        ~g["completed"].astype(bool)
        & (g["date"] >= run_date)
        & (g["season"] == season)
        & (g["week"] >= week) & (g["week"] < week + weeks)
    ].copy()


def build_slate(state: DailyState, run_date: str, weeks: int = 1) -> pd.DataFrame:
    fixtures = upcoming_games(state, run_date, weeks)
    if fixtures.empty:
        return pd.DataFrame(columns=SLATE_COLUMNS)

    rows = []
    for f in fixtures.itertuples(index=False):
        neutral = bool(f.neutral)
        h_rest = None if f.home_rest != f.home_rest else float(f.home_rest)
        a_rest = None if f.away_rest != f.away_rest else float(f.away_rest)
        feat = state.feature_row(f.home_team, f.away_team, neutral, h_rest, a_rest)
        pick_home = feat["p_home"] >= 0.5
        rows.append({
            "game_id": f.game_id,
            "date": f.date, "weekday": f.weekday, "gametime": f.gametime,
            "season": int(f.season), "week": int(f.week), "game_type": f.game_type,
            "week_label": week_label(int(f.week), f.game_type),
            "home_team": f.home_team, "away_team": f.away_team,
            "home_name": name(f.home_team), "away_name": name(f.away_team),
            "home_division": DIVISION_OF.get(f.home_team),
            "away_division": DIVISION_OF.get(f.away_team),
            "div_game": bool(f.div_game) if f.div_game == f.div_game else False,
            "neutral": neutral,
            "home_rest": h_rest, "away_rest": a_rest,
            "elo_home_pre": round(feat["elo_home_pre"], 1),
            "elo_away_pre": round(feat["elo_away_pre"], 1),
            "p_home": round(feat["p_home"], 4),
            "pick": f.home_team if pick_home else f.away_team,
            "pick_prob": round(feat["p_home"] if pick_home else 1 - feat["p_home"], 4),
            "elo_spread": feat["elo_spread"],
            "pred_home_score": feat["pred_home_score"],
            "pred_away_score": feat["pred_away_score"],
            "pred_total": feat["pred_total"],
        })
    slate = pd.DataFrame(rows, columns=SLATE_COLUMNS)
    return slate.sort_values(["date", "gametime", "home_team"]).reset_index(drop=True)


def persist_slate(slate: pd.DataFrame, run_date: str) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    slate.to_csv(PREDICTIONS_DIR / f"slate_{run_date}.csv", index=False)
