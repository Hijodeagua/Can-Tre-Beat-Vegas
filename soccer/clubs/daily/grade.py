"""
Grade persisted slates against played results and maintain the running
ledger at data/soccer_clubs/predictions/grades.csv.

Idempotent by (date, league, home_team, away_team): a match already in the
ledger is never regraded, so re-runs and overlapping slate windows (the
same fixture can appear on two consecutive days' slates) don't double
count. A slate row is graded as soon as its result shows up in
results.csv, whether that takes one day or a postponement's three weeks.
"""

import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from soccer.clubs.daily.config import GRADES_CSV, PREDICTIONS_DIR, ROLLING_WINDOWS

LEDGER_COLUMNS = [
    "date", "league", "season", "home_team", "away_team",
    "pick", "p_H", "p_D", "p_A",
    "outcome", "home_score", "away_score",
    "pick_correct", "log_loss", "graded_on",
]


def _already_graded() -> set[tuple]:
    if not GRADES_CSV.exists():
        return set()
    g = pd.read_csv(GRADES_CSV)
    return set(zip(g["date"], g["league"], g["home_team"], g["away_team"]))


def grade_all(results: pd.DataFrame, run_date: str) -> pd.DataFrame:
    """Grade every ungraded slate row with a known result; return the new
    ledger rows (already appended to grades.csv)."""
    played = results.dropna(subset=["home_score", "away_score"])
    finals = {
        (r.date, r.league, r.home_team, r.away_team): (int(r.home_score), int(r.away_score))
        for r in played.itertuples()
    }
    done = _already_graded()

    new_rows = []
    for path in sorted(Path(PREDICTIONS_DIR).glob("slate_*.csv")):
        slate = pd.read_csv(path)
        for _, m in slate.iterrows():
            key = (m["date"], m["league"], m["home_team"], m["away_team"])
            if key in done or key not in finals:
                continue
            hs, as_ = finals[key]
            outcome = "H" if hs > as_ else ("A" if hs < as_ else "D")
            p_outcome = float(m[f"p_{outcome}"])
            new_rows.append(
                {
                    "date": m["date"],
                    "league": m["league"],
                    "season": m["season"],
                    "home_team": m["home_team"],
                    "away_team": m["away_team"],
                    "pick": m["pick"],
                    "p_H": m["p_H"], "p_D": m["p_D"], "p_A": m["p_A"],
                    "outcome": outcome,
                    "home_score": hs,
                    "away_score": as_,
                    "pick_correct": bool(m["pick"] == outcome),
                    "log_loss": round(-math.log(max(p_outcome, 1e-12)), 4),
                    "graded_on": run_date,
                }
            )
            done.add(key)

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=LEDGER_COLUMNS)
        GRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
        header = not GRADES_CSV.exists()
        new_df.to_csv(GRADES_CSV, mode="a", header=header, index=False)
        return new_df
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def _window_stats(g: pd.DataFrame) -> dict:
    return {
        "graded": int(len(g)),
        "accuracy": round(float(g["pick_correct"].mean()), 4),
        "log_loss": round(float(g["log_loss"].mean()), 4),
    }


def ledger_summary(run_date: str | None = None) -> dict:
    """Cumulative record plus, when a run date is given, rolling windows
    over the last N days of *match* dates (grades.csv `date`, not the day
    the grade landed) — the "how has the model been doing lately" view."""
    if not GRADES_CSV.exists():
        return {"graded": 0}
    g = pd.read_csv(GRADES_CSV)
    if g.empty:
        return {"graded": 0}
    out = {
        **_window_stats(g),
        "by_league": {
            lg: _window_stats(sub) for lg, sub in g.groupby("league")
        },
    }
    if run_date:
        rolling = {}
        for days in ROLLING_WINDOWS:
            start = (date.fromisoformat(run_date) - timedelta(days=days)).isoformat()
            sub = g[(g["date"] >= start) & (g["date"] <= run_date)]
            rolling[f"{days}d"] = (
                _window_stats(sub) if len(sub) else {"graded": 0}
            )
        out["rolling"] = rolling
    return out


def recent_grades(run_date: str, days: int = 7) -> pd.DataFrame:
    """Graded rows whose match date falls in the trailing window — the
    "past week" section of the update email."""
    if not GRADES_CSV.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    g = pd.read_csv(GRADES_CSV)
    start = (date.fromisoformat(run_date) - timedelta(days=days)).isoformat()
    return (
        g[(g["date"] >= start) & (g["date"] <= run_date)]
        .sort_values(["date", "league", "home_team"])
        .reset_index(drop=True)
    )
