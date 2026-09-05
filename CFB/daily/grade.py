"""Grade persisted slates against final scores and maintain the running
ledger at data/college_football/predictions/grades.csv.

Idempotent by game id (the ESPN event id the spine carries): a game already
in the ledger is never regraded, so re-runs and overlapping slate windows
(the same Saturday game appears on Friday's and Saturday's slates) don't
double count. A slate row is graded as soon as its final lands in
games.csv — the next morning, or weeks later for a postponement.

Each graded game also scores the always-pick-home reference forecast on
the same game (the frozen constants ALWAYS_HOME_P / NEUTRAL_HOME_P), so
the summary carries a PAIRED log-loss delta with a standard error — the
MLB pipeline's honesty device, on a per-game ledger like soccer's.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from CFB.daily.config import (
    ALWAYS_HOME_P, GRADES_CSV, NEUTRAL_HOME_P, PREDICTIONS_DIR, ROLLING_WINDOWS,
)

LEDGER_COLUMNS = [
    "game_id", "date", "season", "week", "home_team", "away_team", "neutral",
    "home_fcs", "away_fcs", "pick", "p_home", "pick_prob",
    "pred_home_score", "pred_away_score",
    "home_points", "away_points", "home_win", "pick_correct",
    "log_loss", "brier", "home_log_loss", "d_ll",
    "margin_err", "total_err", "graded_on",
]


def _already_graded() -> set[int]:
    if not GRADES_CSV.exists():
        return set()
    g = pd.read_csv(GRADES_CSV)
    return set(g["game_id"].astype(int))


def _baseline_p(neutral: bool) -> float:
    return NEUTRAL_HOME_P if neutral else ALWAYS_HOME_P


def grade_all(games: pd.DataFrame, run_date: str) -> pd.DataFrame:
    """Grade every ungraded slate row with a known final; return the new
    ledger rows (already appended to grades.csv)."""
    played = games[games["completed"].astype(bool)]
    finals = {
        int(r.game_id): (float(r.home_points), float(r.away_points))
        for r in played.itertuples()
    }
    done = _already_graded()

    new_rows = []
    for path in sorted(Path(PREDICTIONS_DIR).glob("slate_*.csv")):
        slate = pd.read_csv(path)
        for m in slate.itertuples(index=False):
            gid = int(m.game_id)
            if gid in done or gid not in finals:
                continue
            hp, ap = finals[gid]
            if hp == ap:
                continue  # FBS games cannot tie; a 0-0 is a data hole
            home_win = 1.0 if hp > ap else 0.0
            p = min(max(float(m.p_home), 1e-3), 1 - 1e-3)
            ll = -(home_win * math.log(p) + (1 - home_win) * math.log(1 - p))
            pb = _baseline_p(bool(m.neutral))
            ll_home = -(home_win * math.log(pb) + (1 - home_win) * math.log(1 - pb))
            winner = m.home_team if home_win else m.away_team
            new_rows.append({
                "game_id": gid, "date": m.date, "season": int(m.season),
                "week": int(m.week), "home_team": m.home_team, "away_team": m.away_team,
                "neutral": bool(m.neutral),
                "home_fcs": bool(m.home_fcs), "away_fcs": bool(m.away_fcs),
                "pick": m.pick, "p_home": float(m.p_home), "pick_prob": float(m.pick_prob),
                "pred_home_score": float(m.pred_home_score),
                "pred_away_score": float(m.pred_away_score),
                "home_points": hp, "away_points": ap, "home_win": home_win,
                "pick_correct": bool(m.pick == winner),
                "log_loss": round(ll, 4),
                "brier": round((p - home_win) ** 2, 4),
                "home_log_loss": round(ll_home, 4),
                "d_ll": round(ll - ll_home, 4),
                "margin_err": round(abs((float(m.pred_home_score) - float(m.pred_away_score))
                                        - (hp - ap)), 1),
                "total_err": round(abs((float(m.pred_home_score) + float(m.pred_away_score))
                                       - (hp + ap)), 1),
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
    """Cumulative record (with the paired delta vs. always-pick-home) plus,
    when a run date is given, rolling windows over the last N days of
    *game* dates — the "how has the model been doing lately" view."""
    if not GRADES_CSV.exists():
        return {"graded": 0}
    g = pd.read_csv(GRADES_CSV)
    if g.empty:
        return {"graded": 0}
    out = {
        **_window_stats(g),
        "first_date": str(g["date"].min()),
        "last_date": str(g["date"].max()),
        "fbs_only": _window_stats(g[~g["home_fcs"].astype(bool) & ~g["away_fcs"].astype(bool)])
                    if (~g["home_fcs"].astype(bool) & ~g["away_fcs"].astype(bool)).any()
                    else {"graded": 0},
        "by_week": {
            int(w): _window_stats(sub) for w, sub in g.groupby("week")
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
