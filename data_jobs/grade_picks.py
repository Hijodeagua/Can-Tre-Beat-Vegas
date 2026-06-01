"""Grade published picks against final scores and maintain the season record.

Reuses schedule.py (which downloads/caches the nflverse games.csv) to pull final
scores. New picks from picks/picks_today.json are folded into picks/record.json
as pending entries; any pending entry whose game is now final gets graded.

record.json entry fields (per the spec):
    pick_date  — date of the game (ET, YYYY-MM-DD)
    game_id    — odds-feed game id (stable key)
    pick       — human-readable pick (team + spread, or "No pick")
    result     — "win" | "loss" | "push" | null (null = not yet final)
    correct    — true | false | null

Idempotent: keyed on game_id, so running it repeatedly never duplicates a pick
and re-grading an already-final game is a no-op.

Usage:
    python data_jobs/grade_picks.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "NFL" / "model"))

from schedule import load_schedule  # noqa: E402
from predict import FULL_TO_PFR  # noqa: E402  (reuse the single source of truth)

PICKS_TODAY = REPO_ROOT / "picks" / "picks_today.json"
RECORD_PATH = REPO_ROOT / "picks" / "record.json"


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return default


def _pick_label(home: str, away: str, spread_home: float, lean: str) -> str:
    if lean == "cover":
        return f"{home} {spread_home:+g}"
    if lean == "fade":
        return f"{away} {(-spread_home):+g}"
    return "No pick (too close)"


def _ingest_today(record: dict) -> None:
    """Add any new picks_today entries to the record as pending (result=null)."""
    today = _load_json(PICKS_TODAY, {"picks": []})
    by_id = {e["game_id"]: e for e in record["picks"]}
    for p in today.get("picks", []):
        gid = p["game_id"]
        if gid in by_id:
            continue
        game_date = str(p.get("game_time", ""))[:10]
        record["picks"].append({
            "pick_date": game_date,
            "game_id": gid,
            "home_team": p["home_team"],
            "away_team": p["away_team"],
            "vegas_spread": p["vegas_spread"],
            "model_ats_prob": p.get("model_ats_prob"),
            "model_lean": p.get("model_lean"),
            "pick": _pick_label(p["home_team"], p["away_team"], p["vegas_spread"], p.get("model_lean", "push")),
            "result": None,
            "correct": None,
        })


def _score_index() -> dict[tuple, tuple]:
    """(home_pfr, away_pfr, date) -> (home_score, away_score) for completed games."""
    try:
        sched = load_schedule()
    except Exception as exc:  # pragma: no cover
        print(f"  [warn] schedule unavailable ({exc}); cannot grade this run")
        return {}
    sched = sched.dropna(subset=["home_score", "away_score"])
    idx: dict[tuple, tuple] = {}
    for _, g in sched.iterrows():
        d = pd.to_datetime(g["gameday"]).date().isoformat()
        idx[(g["home_team"], g["away_team"], d)] = (float(g["home_score"]), float(g["away_score"]))
    return idx


def _grade_entry(entry: dict, scores: dict[tuple, tuple]) -> bool:
    """Grade one pending entry in place. Returns True if it became final."""
    home = FULL_TO_PFR.get(entry["home_team"])
    away = FULL_TO_PFR.get(entry["away_team"])
    key = (home, away, entry["pick_date"])
    if key not in scores:
        return False

    home_score, away_score = scores[key]
    spread = entry["vegas_spread"]
    lean = entry.get("model_lean", "push")

    # Positive => home beat the spread.
    ats_margin = (home_score - away_score) + spread

    if lean == "push":
        entry["result"] = "no_bet"
        entry["correct"] = None
    elif abs(ats_margin) < 1e-9:
        entry["result"] = "push"
        entry["correct"] = None
    else:
        home_covered = ats_margin > 0
        backed_home = lean == "cover"
        correct = (home_covered and backed_home) or ((not home_covered) and (not backed_home))
        entry["result"] = "win" if correct else "loss"
        entry["correct"] = bool(correct)

    entry["final_score"] = f"{entry['home_team']} {home_score:g} – {entry['away_team']} {away_score:g}"
    return True


def main() -> None:
    record = _load_json(RECORD_PATH, {"picks": []})
    record.setdefault("picks", [])

    _ingest_today(record)

    scores = _score_index()
    newly_graded = 0
    for entry in record["picks"]:
        if entry.get("result") is None:  # still pending
            if _grade_entry(entry, scores):
                newly_graded += 1

    graded = [e for e in record["picks"] if e.get("result") is not None]
    record["updated_at"] = datetime.now(timezone.utc).isoformat()
    record["total_picks"] = len(record["picks"])
    record["graded_picks"] = len(graded)

    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(json.dumps(record, indent=2))
    print(
        f"Record: {len(record['picks'])} total, {len(graded)} graded "
        f"(+{newly_graded} this run) -> {RECORD_PATH.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
