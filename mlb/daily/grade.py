"""Grade a prior day's slate predictions against actual results.

Reads `data/mlb/predictions/slate_{date}.csv` (written the morning the games
were played, from data available before them - no hindsight), joins actuals
from the games file on (date, away, home, game_num), and writes:

- `data/mlb/predictions/graded_{date}.csv` - per-game outcomes
- a (re)computed row for that date in `data/mlb/predictions/grades.csv`,
  the running ledger that feeds the History tab

Games that were predicted but never played (postponements) are dropped from
the metrics; makeup games that were never predicted are ignored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlb.daily.config import GRADES_CSV, PREDICTIONS_DIR

GRADE_COLUMNS = [
    "date", "games", "correct", "accuracy", "log_loss", "brier",
    "avg_margin_err", "avg_total_err", "skipped",
    "cum_games", "cum_correct", "cum_accuracy", "cum_log_loss", "cum_brier",
]


def slate_path(d: str):
    return PREDICTIONS_DIR / f"slate_{d}.csv"


def graded_path(d: str):
    return PREDICTIONS_DIR / f"graded_{d}.csv"


def grade_day(d: str, games: pd.DataFrame) -> pd.DataFrame | None:
    """Grade slate for date `d`. Returns the per-game graded frame, or None
    when no slate file exists for that date."""
    path = slate_path(d)
    if not path.exists():
        return None
    slate = pd.read_csv(path)

    actual = games[(games.date == d)][
        ["date", "away_fr", "home_fr", "game_num", "away_score", "home_score"]
    ].rename(columns={"away_fr": "away", "home_fr": "home"})
    merged = slate.merge(actual, on=["date", "away", "home", "game_num"], how="left")

    played = merged.dropna(subset=["home_score", "away_score"]).copy()
    if played.empty:
        graded = merged.assign(played=False)
    else:
        played["played"] = True
        played["winner"] = np.where(
            played.home_score > played.away_score, played.home, played.away
        )
        played["pick_correct"] = played.pick == played.winner
        p = played.p_home.clip(0.001, 0.999)
        home_won = (played.home_score > played.away_score).astype(float)
        played["game_log_loss"] = -(
            home_won * np.log(p) + (1 - home_won) * np.log(1 - p)
        )
        played["game_brier"] = (p - home_won) ** 2
        played["margin_err"] = (
            (played.pred_home_score - played.pred_away_score)
            - (played.home_score - played.away_score)
        ).abs()
        played["total_err"] = (
            (played.pred_home_score + played.pred_away_score)
            - (played.home_score + played.away_score)
        ).abs()
        unplayed = merged[merged.home_score.isna()].assign(played=False)
        graded = pd.concat([played, unplayed], ignore_index=True)

    graded.to_csv(graded_path(d), index=False)
    return graded


def update_ledger(d: str, graded: pd.DataFrame) -> pd.Series:
    """Insert/replace the ledger row for date `d` and recompute cumulative
    columns across the whole ledger."""
    played = graded[graded.played == True]  # noqa: E712
    n = len(played)
    row = {
        "date": d,
        "games": n,
        "correct": int(played.pick_correct.sum()) if n else 0,
        "accuracy": round(played.pick_correct.mean(), 4) if n else np.nan,
        "log_loss": round(played.game_log_loss.mean(), 4) if n else np.nan,
        "brier": round(played.game_brier.mean(), 4) if n else np.nan,
        "avg_margin_err": round(played.margin_err.mean(), 2) if n else np.nan,
        "avg_total_err": round(played.total_err.mean(), 2) if n else np.nan,
        "skipped": int((graded.played == False).sum()),  # noqa: E712
    }

    if GRADES_CSV.exists():
        ledger = pd.read_csv(GRADES_CSV)
        ledger = ledger[ledger.date != d]
    else:
        ledger = pd.DataFrame(columns=GRADE_COLUMNS)
    ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
    ledger = ledger.sort_values("date").reset_index(drop=True)

    # Cumulative metrics are game-weighted across all graded days.
    g = ledger.games.fillna(0)
    ledger["cum_games"] = g.cumsum().astype(int)
    ledger["cum_correct"] = ledger.correct.fillna(0).cumsum().astype(int)
    ledger["cum_accuracy"] = (
        ledger.cum_correct / ledger.cum_games.replace(0, np.nan)
    ).round(4)
    ledger["cum_log_loss"] = (
        (ledger.log_loss * g).cumsum() / ledger.cum_games.replace(0, np.nan)
    ).round(4)
    ledger["cum_brier"] = (
        (ledger.brier * g).cumsum() / ledger.cum_games.replace(0, np.nan)
    ).round(4)

    ledger[GRADE_COLUMNS].to_csv(GRADES_CSV, index=False)
    return ledger[ledger.date == d].iloc[0]
