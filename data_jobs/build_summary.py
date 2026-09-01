#!/usr/bin/env python3
"""
Aggregate every forecasting model's graded record into one cross-sport file.

The `/vegas` home page opens with a track-record strip covering all models at
once. No existing job produces that: the MLB pipeline writes its own ledger and
the odds job writes the slate, but nothing spans them. This module reads what
those two already publish and writes:

    web/public/data/summary.json

Rules that the front end depends on and that are enforced here:

* Metrics are **game-weighted** across every graded day, never a mean of daily
  accuracies. A 15-game day and a 4-game day do not count equally.
* A model appears in `models` always, but only reports figures once it has
  graded games. Until then every metric is `null`, which the site renders as an
  em dash — never a zero, which would read as a real and terrible score.
* `roi` is `null` for every model by design. Nothing here stakes money, so
  there is no return to report.

Usage:
    python -m data_jobs.build_summary [--output web/public/data]
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

MLB_GRADES_CSV = REPO_ROOT / "data" / "mlb" / "predictions" / "grades.csv"
SOCCER_GRADES_CSV = REPO_ROOT / "data" / "soccer_clubs" / "predictions" / "grades.csv"

# A sport counts as in season while its last graded day is this recent. Longer
# than an All-Star break, shorter than an off-season.
IN_SEASON_GRACE_DAYS = 21

# Season starts are month-precision: no file in the repo carries real season
# start dates, so the site says "Kickoff September 2026" rather than a day.
# Stored as a month number and resolved to the next occurrence, so the copy
# does not rot into a past year.
SPORTS: List[Dict[str, Any]] = [
    {"key": "mlb", "sport": "MLB", "emoji": "⚾", "source": "mlb", "start_month": 3},
    {"key": "soccer", "sport": "Soccer", "emoji": "⚽", "source": "soccer", "start_month": 8},
    {"key": "nfl", "sport": "NFL", "emoji": "\U0001f3c8", "source": "odds", "start_month": 9},
    {"key": "nba", "sport": "NBA", "emoji": "\U0001f3c0", "source": "odds", "start_month": 10},
]


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _f(value: Any) -> Optional[float]:
    """CSV cell -> float, treating blanks and NaN text as missing."""
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _i(value: Any) -> Optional[int]:
    f = _f(value)
    return None if f is None else int(f)


def next_season_start(start_month: int, today: date) -> str:
    """Month-precision season start, resolved forward from today."""
    year = today.year if start_month >= today.month else today.year + 1
    return f"{year}-{start_month:02d}"


def _days_since(day: Optional[str], today: date) -> Optional[int]:
    if not day:
        return None
    try:
        return (today - datetime.strptime(day[:10], "%Y-%m-%d").date()).days
    except ValueError:
        return None


def mlb_record(grades_csv: Path = MLB_GRADES_CSV) -> Dict[str, Any]:
    """
    Cumulative MLB record from the daily grading ledger.

    The ledger's `cum_*` columns are already game-weighted across every graded
    day — they are accumulated from game-level outcomes, not averaged from the
    per-day rows — so they are read directly rather than recomputed from the
    rounded daily figures, which would lose precision.
    """
    empty = {"games": 0, "correct": 0, "accuracy": None, "log_loss": None,
             "brier": None, "last_graded": None}
    if not grades_csv.exists():
        return empty

    rows: List[Dict[str, str]] = []
    with grades_csv.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("date"):
                rows.append(row)
    if not rows:
        return empty

    rows.sort(key=lambda r: r["date"])
    last = rows[-1]
    games = _i(last.get("cum_games")) or 0
    correct = _i(last.get("cum_correct")) or 0
    if games <= 0:
        return empty

    accuracy = _f(last.get("cum_accuracy"))
    if accuracy is None:
        accuracy = correct / games
    return {
        "games": games,
        "correct": correct,
        "accuracy": accuracy,
        "log_loss": _f(last.get("cum_log_loss")),
        "brier": _f(last.get("cum_brier")),
        "last_graded": last["date"],
    }


def soccer_record(grades_csv: Path = SOCCER_GRADES_CSV) -> Dict[str, Any]:
    """
    Cumulative soccer record from the club pipeline's game-level ledger
    (one row per graded match: pick_correct + log loss on the three-way
    W/D/L pick). Brier stays None — the 2-way Brier the other sports report
    isn't defined for a 3-way outcome, and a lookalike number would invite
    a false comparison.
    """
    empty = {"games": 0, "correct": 0, "accuracy": None, "log_loss": None,
             "brier": None, "last_graded": None}
    if not grades_csv.exists():
        return empty

    rows: List[Dict[str, str]] = []
    with grades_csv.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("date"):
                rows.append(row)
    if not rows:
        return empty

    games = len(rows)
    correct = sum(1 for r in rows if str(r.get("pick_correct", "")).lower() == "true")
    losses = [f for r in rows if (f := _f(r.get("log_loss"))) is not None]
    return {
        "games": games,
        "correct": correct,
        "accuracy": round(correct / games, 4),
        "log_loss": round(sum(losses) / len(losses), 4) if losses else None,
        "brier": None,
        "last_graded": max(r["date"] for r in rows),
    }


def odds_sport_games(slate: Optional[Dict[str, Any]], key: str) -> int:
    """How many games this sport has on the current odds slate."""
    for sport in (slate or {}).get("sports", []):
        if sport.get("key") == key:
            return len(sport.get("games") or [])
    return 0


def build_models(
    today: date,
    mlb_latest: Optional[Dict[str, Any]],
    slate: Optional[Dict[str, Any]],
    grades_csv: Path = MLB_GRADES_CSV,
    soccer_latest: Optional[Dict[str, Any]] = None,
    soccer_grades_csv: Path = SOCCER_GRADES_CSV,
) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []

    for spec in SPORTS:
        if spec["source"] == "mlb":
            record = mlb_record(grades_csv)
            slate_games = len((mlb_latest or {}).get("slate") or [])
            since = _days_since(record["last_graded"], today)
            in_season = slate_games > 0 or (since is not None and since <= IN_SEASON_GRACE_DAYS)
        elif spec["source"] == "soccer":
            record = soccer_record(soccer_grades_csv)
            slate_games = len((soccer_latest or {}).get("slate") or [])
            since = _days_since(record["last_graded"], today)
            in_season = slate_games > 0 or (since is not None and since <= IN_SEASON_GRACE_DAYS)
        else:
            record = {"games": 0, "correct": 0, "accuracy": None, "log_loss": None,
                      "brier": None, "last_graded": None}
            slate_games = odds_sport_games(slate, spec["key"])
            in_season = slate_games > 0

        reporting = record["games"] > 0
        model: Dict[str, Any] = {
            "sport": spec["sport"],
            "emoji": spec["emoji"],
            # Soccer picks are three-way (W/D/L): its log loss lives on a
            # different scale from the binary sports (baseline ~1.10 vs
            # ~0.69), so build_overall keeps it out of the blended log loss.
            "outcomes": 3 if spec["source"] == "soccer" else 2,
            "status": "in_season" if in_season else "off_season",
            # Hyphen here; the site renders it with an en dash.
            "record": f"{record['correct']}-{record['games'] - record['correct']}" if reporting else None,
            "games": record["games"],
            "accuracy": record["accuracy"] if reporting else None,
            "log_loss": record["log_loss"] if reporting else None,
            "brier": record["brier"] if reporting else None,
            # Nothing here stakes money, so there is never a return to report.
            "roi": None,
            "last_graded": record["last_graded"] if reporting else None,
            "slate_games": slate_games,
        }
        if not in_season:
            model["season_starts"] = next_season_start(spec["start_month"], today)
        models.append(model)

    return models


def build_overall(models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Cross-sport totals, game-weighted.

    Every model that has graded games contributes in proportion to how many —
    so a sport with 400 graded games does not get averaged down to parity with
    one that has 12.
    """
    reporting = [m for m in models if m["games"] > 0]
    games = sum(m["games"] for m in reporting)
    if games == 0:
        return {"record": None, "games": 0, "accuracy": None, "log_loss": None,
                "brier": None, "roi": None, "sports_reporting": 0}

    correct = 0
    for m in reporting:
        wins, _, losses = (m["record"] or "0-0").partition("-")
        correct += int(wins or 0)

    def weighted(key: str) -> Optional[float]:
        # Only binary-outcome models blend: a three-way log loss averaged
        # into a two-way one would read as a sudden collapse in skill.
        parts = [(m[key], m["games"]) for m in reporting
                 if m.get(key) is not None and m.get("outcomes", 2) == 2]
        total = sum(n for _, n in parts)
        if total == 0:
            return None
        return round(sum(v * n for v, n in parts) / total, 4)

    return {
        "record": f"{correct}-{games - correct}",
        "games": games,
        "accuracy": round(correct / games, 4),
        "log_loss": weighted("log_loss"),
        "brier": weighted("brier"),
        "roi": None,
        "sports_reporting": len(reporting),
    }


def build_summary(
    output_dir: Path,
    today: Optional[date] = None,
    grades_csv: Path = MLB_GRADES_CSV,
) -> Dict[str, Any]:
    today = today or datetime.now(timezone.utc).date()
    mlb_latest = _read_json(output_dir / "mlb" / "latest.json")
    soccer_latest = _read_json(output_dir / "soccer" / "latest.json")
    slate = _read_json(output_dir / "slate.json")

    models = build_models(today, mlb_latest, slate, grades_csv,
                          soccer_latest=soccer_latest)
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall": build_overall(models),
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cross-sport summary JSON")
    parser.add_argument("--output", default="web/public/data")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    overall = summary["overall"]
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"  overall: {overall['record'] or '—'} across {overall['games']} graded games "
          f"({overall['sports_reporting']} sport(s) reporting)")
    for m in summary["models"]:
        print(f"  {m['sport']}: {m['status']} · {m['record'] or '—'} · "
              f"last graded {m['last_graded'] or '—'}")


if __name__ == "__main__":
    main()
