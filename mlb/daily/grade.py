"""Grade a prior day's slate predictions against actual results.

Reads `slate_{date}.csv` from a model-version bucket (written the morning
the games were played, from data available before them - no hindsight),
joins actuals from the games file on (date, away, home, game_num), and
writes:

- `graded_{date}.csv` in the same bucket - per-game outcomes
- a (re)computed row for that date in that bucket's `grades.csv`, the
  running ledger that feeds the History tab

Each graded game also scores the always-pick-home reference forecast (the
frozen constant ALWAYS_HOME_P) on the same game, so the ledger can carry a
cumulative PAIRED log-loss delta with a running standard error - a per-game
paired difference, not a comparison of two independent means.

Games that were predicted but never played (postponements) are dropped from
the metrics; makeup games that were never predicted are ignored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlb.daily.config import ALWAYS_HOME_P, PREDICTIONS_DIR

GRADE_COLUMNS = [
    "date", "games", "correct", "accuracy", "log_loss", "brier",
    "avg_margin_err", "avg_total_err", "skipped",
    "home_log_loss", "home_correct", "d_ll_sum", "d_ll_sq_sum",
    "cum_games", "cum_correct", "cum_accuracy", "cum_log_loss", "cum_brier",
    "cum_d_ll_mean", "cum_d_ll_se",
]


def slate_path(d: str, pred_dir: Path | None = None):
    return (pred_dir or PREDICTIONS_DIR) / f"slate_{d}.csv"


def graded_path(d: str, pred_dir: Path | None = None):
    return (pred_dir or PREDICTIONS_DIR) / f"graded_{d}.csv"


def grades_path(pred_dir: Path | None = None):
    return (pred_dir or PREDICTIONS_DIR) / "grades.csv"


def grade_day(d: str, games: pd.DataFrame,
              pred_dir: Path | None = None) -> pd.DataFrame | None:
    """Grade slate for date `d`. Returns the per-game graded frame, or None
    when no slate file exists for that date."""
    path = slate_path(d, pred_dir)
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
        # Always-pick-home reference on the SAME game (paired baseline).
        played["home_game_log_loss"] = -(
            home_won * np.log(ALWAYS_HOME_P)
            + (1 - home_won) * np.log(1 - ALWAYS_HOME_P)
        )
        played["d_ll"] = played.game_log_loss - played.home_game_log_loss
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

    graded.to_csv(graded_path(d, pred_dir), index=False)
    return graded


def _paired_day_stats(graded: pd.DataFrame) -> dict:
    """Per-day sums that let the ledger recompose the cumulative paired
    delta and its SE from daily rows alone."""
    played = graded[graded.played == True]  # noqa: E712
    if not len(played):
        return {"home_log_loss": np.nan, "home_correct": 0,
                "d_ll_sum": 0.0, "d_ll_sq_sum": 0.0}
    if "d_ll" not in played.columns:  # pre-upgrade graded file
        p = played.p_home.clip(0.001, 0.999)
        home_won = (played.home_score > played.away_score).astype(float)
        ll = -(home_won * np.log(p) + (1 - home_won) * np.log(1 - p))
        ll_home = -(home_won * np.log(ALWAYS_HOME_P)
                    + (1 - home_won) * np.log(1 - ALWAYS_HOME_P))
        d = ll - ll_home
        home_correct = int(home_won.sum())
    else:
        d = played.d_ll
        ll_home = played.home_game_log_loss
        home_correct = int((played.home_score > played.away_score).sum())
    return {
        "home_log_loss": round(float(ll_home.mean()), 4),
        "home_correct": home_correct,
        "d_ll_sum": float(d.sum()),
        "d_ll_sq_sum": float((d ** 2).sum()),
    }


def update_ledger(d: str, graded: pd.DataFrame,
                  pred_dir: Path | None = None) -> pd.Series:
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
        **_paired_day_stats(graded),
    }

    grades_csv = grades_path(pred_dir)
    if grades_csv.exists():
        ledger = pd.read_csv(grades_csv)
        ledger = ledger[ledger.date != d]
    else:
        ledger = pd.DataFrame(columns=GRADE_COLUMNS)
    ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
    ledger = ledger.sort_values("date").reset_index(drop=True)

    # Ledger rows written before the paired-baseline upgrade lack the day
    # sums; rebuild them once from the graded files on disk.
    for col in ("home_log_loss", "home_correct", "d_ll_sum", "d_ll_sq_sum"):
        if col not in ledger.columns:
            ledger[col] = np.nan
    needs = ledger[ledger.d_ll_sum.isna()]
    for r in needs.itertuples():
        gpath = graded_path(r.date, pred_dir)
        if gpath.exists():
            stats = _paired_day_stats(pd.read_csv(gpath))
            for k, v in stats.items():
                ledger.loc[ledger.date == r.date, k] = v
    ledger["d_ll_sum"] = ledger.d_ll_sum.fillna(0.0).astype(float)
    ledger["d_ll_sq_sum"] = ledger.d_ll_sq_sum.fillna(0.0).astype(float)
    ledger["home_correct"] = ledger.home_correct.fillna(0).astype(int)

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

    # Paired delta vs always-pick-home: mean and SE of the per-game
    # differences, recomposed from the daily (sum, sum of squares) pairs.
    cn = ledger.cum_games.replace(0, np.nan)
    cs = ledger.d_ll_sum.cumsum()
    css = ledger.d_ll_sq_sum.cumsum()
    mean = cs / cn
    var = (css - cn * mean**2) / (cn - 1)
    ledger["cum_d_ll_mean"] = mean.round(5)
    ledger["cum_d_ll_se"] = (np.sqrt(var.clip(lower=0) / cn)).round(5)

    ledger[GRADE_COLUMNS].to_csv(grades_csv, index=False)
    return ledger[ledger.date == d].iloc[0]
