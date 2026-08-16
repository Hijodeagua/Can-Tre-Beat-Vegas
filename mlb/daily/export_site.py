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
    ACTIVE_MODEL, MODEL_V1, SHADOW_MODEL, SITE_DATA_DIR,
    SITE_HISTORY_DIR, TEAM_DIVISION, TEAM_NAMES, predictions_dir,
)
from mlb.daily.emails import grade_html, slate_html


def _bucket_summary(version: str) -> dict | None:
    """Cumulative record for one model version's ledger - kept separate so
    a model change never silently contaminates the live record."""
    grades = predictions_dir(version) / "grades.csv"
    if not grades.exists():
        return None
    ledger = pd.read_csv(grades).sort_values("date")
    if ledger.empty:
        return None
    last = ledger.iloc[-1]
    def _f(key):
        v = last.get(key)
        return None if v is None or pd.isna(v) else float(v)
    if version == ACTIVE_MODEL:
        role = "active"
    elif version == SHADOW_MODEL:
        role = "shadow"
    else:
        # Neither role today - a prior active or shadow model, frozen in
        # place after a cutover moved on without it.
        role = "historical"
    return {
        "version": version,
        "role": role,
        "first_date": str(ledger.date.min()),
        "last_date": str(ledger.date.max()),
        "games": int(last.cum_games),
        "correct": int(last.cum_correct),
        "accuracy": _f("cum_accuracy"),
        "log_loss": _f("cum_log_loss"),
        "brier": _f("cum_brier"),
        "d_ll_vs_home_mean": _f("cum_d_ll_mean"),
        "d_ll_vs_home_se": _f("cum_d_ll_se"),
    }


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
                  shadow_slate: pd.DataFrame | None = None) -> None:
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    st = standings.set_index("team")
    futures_out = futures.copy()
    futures_out["name"] = futures_out.team.map(TEAM_NAMES)
    futures_out["division"] = futures_out.team.map(TEAM_DIVISION)
    futures_out["wins"] = futures_out.team.map(st.wins)
    futures_out["losses"] = futures_out.team.map(st.losses)
    futures_out["run_diff"] = futures_out.team.map(st.run_diff)

    # The day-by-day History tab tracks whichever model is active now - on
    # a model change it starts a fresh run from that model's own bucket
    # rather than continuing to show the frozen pre-change ledger (that
    # ledger is still visible, unabridged, in the model-version table below).
    history = []
    active_grades = predictions_dir(ACTIVE_MODEL) / "grades.csv"
    if active_grades.exists():
        ledger = pd.read_csv(active_grades).sort_values("date", ascending=False)
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

    # Always show MODEL_V1 (the historical anchor, even once frozen post-
    # cutover) plus whichever versions are actually active/shadow now.
    versions = list(dict.fromkeys(
        v for v in (MODEL_V1, ACTIVE_MODEL, SHADOW_MODEL) if v
    ))
    models = [s for s in (_bucket_summary(v) for v in versions) if s]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "date": run_date,
        "model_version": ACTIVE_MODEL,
        "models": models,
        "slate": _records(slate),
        "shadow_slate": _records(shadow_slate),
        "futures": _records(futures_out),
        "graded_date": graded_date,
        "graded": graded_out,
        "history": history,
        "team_names": TEAM_NAMES,
    }
    (SITE_DATA_DIR / "latest.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8"
    )
