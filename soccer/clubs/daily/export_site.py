"""
Write the site JSON: web/public/data/soccer/latest.json plus a per-day
history snapshot — the club-soccer sibling of the MLB card's data files.
The web app doesn't render a soccer page yet; this is the contract it will
read when it does, shaped like the MLB export (ratings / slate / futures /
ledger under one roof with a generated_at stamp).
"""

import json
from datetime import datetime, timezone

import pandas as pd

from soccer.clubs.daily.config import SITE_DIR, SITE_HISTORY, SITE_LATEST
from soccer.clubs.daily.state import DailyState
from soccer.clubs.data.leagues import LEAGUES


def ratings_payload(state: DailyState) -> dict:
    out = {}
    for league, engine in state.engines.items():
        latest = state.history[state.history["league"] == league]["season"].max()
        table = engine.table(season=latest)
        out[league] = {
            "name": LEAGUES[league].name,
            "asOfSeason": latest,
            "clubs": [
                {"team": r["team"], "elo": round(float(r["elo"]), 1),
                 "matches": int(r["matches"])}
                for _, r in table.iterrows()
            ],
        }
    return out


def export(state: DailyState, run_date: str, slate: pd.DataFrame,
           futures: dict, ledger: dict, graded_today: pd.DataFrame) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_date": run_date,
        "ratings": ratings_payload(state),
        "slate": slate.to_dict(orient="records"),
        "graded_today": graded_today.to_dict(orient="records"),
        "ledger": ledger,
        "futures": futures,
    }
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    SITE_HISTORY.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    SITE_LATEST.write_text(text)
    (SITE_HISTORY / f"{run_date}.json").write_text(text)
