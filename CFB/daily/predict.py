"""Predict the slate: every FBS-involved game whose US/Eastern kickoff date
falls inside the run's window, with the Elo win probability, the pick and
an expected score.

The slate is persisted to data/college_football/predictions/slate_{D}.csv
the day it is predicted — that file, written before the games were played,
is what `grade.py` later grades. Re-running a date overwrites its slate
file, but grading is idempotent through the ledger (keyed by game id), so
a regenerated slate can't rewrite the record.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from CFB.daily.config import PREDICTIONS_DIR, SLATE_WINDOW_DAYS
from CFB.daily.state import DailyState
from CFB.data.teams import FBS

SLATE_COLUMNS = [
    "game_id", "date", "season", "week", "season_type",
    "home_team", "away_team", "home_conference", "away_conference",
    "neutral", "home_fcs", "away_fcs",
    "elo_home_pre", "elo_away_pre", "p_home", "pick", "pick_prob",
    "pred_home_score", "pred_away_score", "pred_total", "notes",
]


def upcoming_games(state: DailyState, run_date: str,
                   window_days: int = SLATE_WINDOW_DAYS) -> pd.DataFrame:
    end = (date.fromisoformat(run_date) + timedelta(days=window_days)).isoformat()
    g = state.games
    # `completed` is already masked for dates >= run_date by state.as_of,
    # so the window filter alone is the slate; the mask is belt and braces.
    return g[
        ~g["completed"].astype(bool)
        & (g["date"] >= run_date)
        & (g["date"] < end)
    ].copy()


def build_slate(state: DailyState, run_date: str,
                window_days: int = SLATE_WINDOW_DAYS) -> pd.DataFrame:
    fixtures = upcoming_games(state, run_date, window_days)
    if fixtures.empty:
        return pd.DataFrame(columns=SLATE_COLUMNS)

    rows = []
    for f in fixtures.itertuples(index=False):
        h_div = f.home_division if isinstance(f.home_division, str) else ""
        a_div = f.away_division if isinstance(f.away_division, str) else ""
        neutral = bool(f.neutral_site)
        feat = state.feature_row(f.home_team, f.away_team, h_div, a_div, neutral)
        pick_home = feat["p_home"] >= 0.5
        rows.append({
            "game_id": int(f.game_id),
            "date": f.date, "season": int(f.season), "week": int(f.week),
            "season_type": f.season_type,
            "home_team": f.home_team, "away_team": f.away_team,
            "home_conference": f.home_conference, "away_conference": f.away_conference,
            "neutral": neutral,
            "home_fcs": h_div != FBS, "away_fcs": a_div != FBS,
            "elo_home_pre": round(feat["elo_home_pre"], 1),
            "elo_away_pre": round(feat["elo_away_pre"], 1),
            "p_home": round(feat["p_home"], 4),
            "pick": f.home_team if pick_home else f.away_team,
            "pick_prob": round(feat["p_home"] if pick_home else 1 - feat["p_home"], 4),
            "pred_home_score": feat["pred_home_score"],
            "pred_away_score": feat["pred_away_score"],
            "pred_total": feat["pred_total"],
            "notes": f.notes if isinstance(f.notes, str) else "",
        })
    slate = pd.DataFrame(rows, columns=SLATE_COLUMNS)
    return slate.sort_values(["date", "week", "home_team"]).reset_index(drop=True)


def persist_slate(slate: pd.DataFrame, run_date: str) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    slate.to_csv(PREDICTIONS_DIR / f"slate_{run_date}.csv", index=False)
