"""Write the JSON consumed by web/app/mlb (the Can Tre Beat Vegas model card)
and the static per-day history snapshot pages.

`web/public/data/mlb/latest.json` carries all four tabs; the history rows
link to `web/public/data/mlb/history/{date}.html`, a self-contained render of
that day's graded slate (same fragments as the emails - cheaper than a
PDF/PNG step and versioned with the repo).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from mlb.daily.config import (
    GRADES_CSV, SITE_DATA_DIR, SITE_HISTORY_DIR, TEAM_DIVISION, TEAM_NAMES,
)
from mlb.daily.emails import grade_html, slate_html


def _records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def write_history_snapshot(date: str, graded: pd.DataFrame | None,
                           ledger_row: pd.Series | None,
                           slate: pd.DataFrame | None) -> str | None:
    """Static page for one graded day; returns the site-relative link."""
    if graded is None:
        return None
    SITE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    parts = [grade_html(date, graded, ledger_row)]
    if slate is not None and not slate.empty:
        parts.append("<hr style='margin:32px 0;border:none;"
                     "border-top:1px solid #ddd;'>")
        parts.append(slate_html(date, slate))
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>MLB {date} - Can Tre Beat Vegas</title>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "</head><body style='margin:24px;background:#fff;'>"
        + "".join(parts) + "</body></html>"
    )
    (SITE_HISTORY_DIR / f"{date}.html").write_text(page, encoding="utf-8")
    return f"/vegas/data/mlb/history/{date}.html"


def export_latest(run_date: str, slate: pd.DataFrame,
                  futures: pd.DataFrame, standings: pd.DataFrame,
                  graded: pd.DataFrame | None,
                  graded_date: str | None,
                  calibration: dict | None = None) -> None:
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    st = standings.set_index("team")
    futures_out = futures.copy()
    futures_out["name"] = futures_out.team.map(TEAM_NAMES)
    futures_out["division"] = futures_out.team.map(TEAM_DIVISION)
    futures_out["wins"] = futures_out.team.map(st.wins)
    futures_out["losses"] = futures_out.team.map(st.losses)
    futures_out["run_diff"] = futures_out.team.map(st.run_diff)

    history = []
    if GRADES_CSV.exists():
        ledger = pd.read_csv(GRADES_CSV).sort_values("date", ascending=False)
        for r in ledger.itertuples():
            snapshot = SITE_HISTORY_DIR / f"{r.date}.html"
            history.append({
                "date": r.date,
                "games": int(r.games),
                "correct": int(r.correct),
                "accuracy": None if pd.isna(r.accuracy) else float(r.accuracy),
                "log_loss": None if pd.isna(r.log_loss) else float(r.log_loss),
                "brier": None if pd.isna(r.brier) else float(r.brier),
                "avg_margin_err": None if pd.isna(r.avg_margin_err) else float(r.avg_margin_err),
                "avg_total_err": None if pd.isna(r.avg_total_err) else float(r.avg_total_err),
                "cum_accuracy": None if pd.isna(r.cum_accuracy) else float(r.cum_accuracy),
                "cum_log_loss": None if pd.isna(r.cum_log_loss) else float(r.cum_log_loss),
                "link": (f"/vegas/data/mlb/history/{r.date}.html"
                         if snapshot.exists() else None),
            })

    graded_out = None
    if graded is not None:
        g = graded.copy()
        for col in ("home_score", "away_score"):
            g[col] = g[col].astype("Int64")
        graded_out = _records(g)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "date": run_date,
        "slate": _records(slate),
        "futures": _records(futures_out),
        "graded_date": graded_date,
        "graded": graded_out,
        "history": history,
        "calibration": calibration,
        "team_names": TEAM_NAMES,
    }
    (SITE_DATA_DIR / "latest.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8"
    )
