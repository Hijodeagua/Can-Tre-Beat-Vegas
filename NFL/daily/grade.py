"""Grade persisted slates against final scores and maintain the running
ledger at data/nfl/predictions/grades.csv.

Idempotent by game id (the nflverse `game_id`): a game already in the
ledger is never regraded, and slate files are read oldest first, so the
pick that counts is the first morning it was persisted — a Sunday game
locks on Tuesday's slate even though Saturday's run re-predicted it.

Each graded game also scores the always-pick-home reference forecast on
the same game (the frozen constants ALWAYS_HOME_P / NEUTRAL_HOME_P), so
the summary carries a PAIRED log-loss delta with a standard error.

Ties are real in the NFL (about one a season): the outcome is 0.5, the
log loss scores both halves, and neither side's pick is correct.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from NFL.daily.config import (
    ALWAYS_HOME_P, GRADES_CSV, NEUTRAL_HOME_P, PREDICTIONS_DIR, ROLLING_WINDOWS,
)
from NFL.elo.teams import week_label

LEDGER_COLUMNS = [
    "game_id", "date", "season", "week", "game_type", "week_label",
    "home_team", "away_team", "neutral",
    "pick", "p_home", "pick_prob", "elo_spread",
    "pred_home_score", "pred_away_score",
    "home_score", "away_score", "home_win", "tie", "pick_correct",
    "log_loss", "brier", "home_log_loss", "d_ll",
    "margin_err", "total_err", "graded_on",
]


def _already_graded() -> set[str]:
    if not GRADES_CSV.exists():
        return set()
    g = pd.read_csv(GRADES_CSV)
    return set(g["game_id"].astype(str))


def _baseline_p(neutral: bool) -> float:
    return NEUTRAL_HOME_P if neutral else ALWAYS_HOME_P


def _ll(p: float, y: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def grade_all(games: pd.DataFrame, run_date: str) -> pd.DataFrame:
    """Grade every ungraded slate row with a known final; return the new
    ledger rows (already appended to grades.csv)."""
    played = games[games["completed"].astype(bool)]
    finals = {
        str(r.game_id): (float(r.home_score), float(r.away_score))
        for r in played.itertuples()
    }
    done = _already_graded()

    new_rows = []
    for path in sorted(Path(PREDICTIONS_DIR).glob("slate_*.csv")):
        slate = pd.read_csv(path)
        for m in slate.itertuples(index=False):
            gid = str(m.game_id)
            if gid in done or gid not in finals:
                continue
            hs, as_ = finals[gid]
            home_win = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
            tie = hs == as_
            p = float(m.p_home)
            ll = _ll(p, home_win)
            ll_home = _ll(_baseline_p(bool(m.neutral)), home_win)
            winner = None if tie else (m.home_team if home_win else m.away_team)
            pred_margin = float(m.pred_home_score) - float(m.pred_away_score)
            pred_total = float(m.pred_home_score) + float(m.pred_away_score)
            new_rows.append({
                "game_id": gid, "date": m.date, "season": int(m.season),
                "week": int(m.week), "game_type": m.game_type,
                "week_label": week_label(int(m.week), m.game_type),
                "home_team": m.home_team, "away_team": m.away_team,
                "neutral": bool(m.neutral),
                "pick": m.pick, "p_home": p, "pick_prob": float(m.pick_prob),
                "elo_spread": float(m.elo_spread),
                "pred_home_score": float(m.pred_home_score),
                "pred_away_score": float(m.pred_away_score),
                "home_score": hs, "away_score": as_, "home_win": home_win,
                "tie": tie,
                "pick_correct": bool(winner is not None and m.pick == winner),
                "log_loss": round(ll, 4),
                "brier": round((p - home_win) ** 2, 4),
                "home_log_loss": round(ll_home, 4),
                "d_ll": round(ll - ll_home, 4),
                "margin_err": round(abs(pred_margin - (hs - as_)), 1),
                "total_err": round(abs(pred_total - (hs + as_)), 1),
                "graded_on": run_date,
            })
            done.add(gid)

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=LEDGER_COLUMNS)
        GRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
        header = not GRADES_CSV.exists()
        new_df.to_csv(GRADES_CSV, mode="a", header=header, index=False)
        return new_df
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def _window_stats(g: pd.DataFrame) -> dict:
    n = int(len(g))
    out = {
        "graded": n,
        "correct": int(g["pick_correct"].astype(bool).sum()),
        "ties": int(g["tie"].astype(bool).sum()) if "tie" in g else 0,
        "accuracy": round(float(g["pick_correct"].astype(bool).mean()), 4),
        "log_loss": round(float(g["log_loss"].mean()), 4),
        "brier": round(float(g["brier"].mean()), 4),
        "home_log_loss": round(float(g["home_log_loss"].mean()), 4),
        "avg_margin_err": round(float(g["margin_err"].mean()), 1),
        "avg_total_err": round(float(g["total_err"].mean()), 1),
    }
    d = g["d_ll"].astype(float)
    out["d_ll_mean"] = round(float(d.mean()), 5)
    out["d_ll_se"] = (round(float(d.std(ddof=1) / math.sqrt(n)), 5)
                      if n > 1 else None)
    return out


def ledger_summary(run_date: str | None = None) -> dict:
    """Cumulative record (with the paired delta vs. always-pick-home), a
    per-week breakdown, and rolling windows over the last N days of *game*
    dates when a run date is given."""
    if not GRADES_CSV.exists():
        return {"graded": 0}
    g = pd.read_csv(GRADES_CSV)
    if g.empty:
        return {"graded": 0}
    out = {
        **_window_stats(g),
        "first_date": str(g["date"].min()),
        "last_date": str(g["date"].max()),
        "by_week": {
            str(label): _window_stats(sub)
            for (_, _, label), sub in g.groupby(["season", "week", "week_label"], sort=True)
        },
    }
    if run_date:
        rolling = {}
        for days in ROLLING_WINDOWS:
            start = (date.fromisoformat(run_date) - timedelta(days=days)).isoformat()
            sub = g[(g["date"] >= start) & (g["date"] <= run_date)]
            rolling[f"{days}d"] = _window_stats(sub) if len(sub) else {"graded": 0}
        out["rolling"] = rolling
    return out


def recent_grades(run_date: str, days: int = 7) -> pd.DataFrame:
    """Graded rows whose game date falls in the trailing window."""
    if not GRADES_CSV.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    g = pd.read_csv(GRADES_CSV)
    start = (date.fromisoformat(run_date) - timedelta(days=days)).isoformat()
    return (
        g[(g["date"] >= start) & (g["date"] <= run_date)]
        .sort_values(["date", "home_team"])
        .reset_index(drop=True)
    )
