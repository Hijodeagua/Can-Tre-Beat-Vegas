#!/usr/bin/env python3
"""
Staleness checker for odds data files.

Exits non-zero if a sport's `_latest.csv` newest "Timestamp Pulled" is older
than a threshold WHILE the current fetch run reported upcoming games for that
sport. A sport with zero upcoming games (offseason) never alarms, so a frozen
NBA file in July is fine, but a frozen file during the season is a failure.

Reads the per-run status JSON written by fetch_odds (data/fetch_status.json)
to know how many games the API reported for each sport in the current run.

Usage:
    python -m data_jobs.odds_api.check_staleness [--sport nfl|nba|all]
        [--threshold-days 3] [--base-dir .] [--status-file PATH]
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd
import pytz

# Add parent directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_jobs.odds_api.config import SUPPORTED_SPORTS

DEFAULT_THRESHOLD_DAYS = 3
TIMESTAMP_COLUMN = "Timestamp Pulled"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
EASTERN = pytz.timezone("US/Eastern")


def load_fetch_status(status_path: str) -> dict:
    """Load the per-run fetch status JSON. Returns {} if missing/unreadable."""
    if not os.path.exists(status_path):
        return {}
    try:
        with open(status_path) as f:
            return json.load(f).get("sports", {})
    except (json.JSONDecodeError, OSError):
        return {}


def newest_timestamp_pulled(latest_path: str) -> Optional[datetime]:
    """
    Return the newest "Timestamp Pulled" in a _latest.csv as an aware
    US/Eastern datetime, or None if the file/column is missing or empty.
    """
    if not os.path.exists(latest_path):
        return None
    try:
        df = pd.read_csv(latest_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return None

    if TIMESTAMP_COLUMN not in df.columns or df.empty:
        return None

    parsed = pd.to_datetime(df[TIMESTAMP_COLUMN], format=TIMESTAMP_FORMAT, errors="coerce")
    parsed = parsed.dropna()
    if parsed.empty:
        return None

    return EASTERN.localize(parsed.max().to_pydatetime())


def check_sport_staleness(
    sport: str,
    fetch_status: dict,
    base_dir: str = ".",
    threshold_days: int = DEFAULT_THRESHOLD_DAYS,
    now: Optional[datetime] = None,
) -> tuple[bool, str]:
    """
    Check one sport's _latest.csv for staleness.

    Returns:
        (ok, message) - ok is False only when the API reported upcoming games
        for this sport in the current run AND the latest data is older than
        the threshold (or missing entirely).
    """
    sport_config = SUPPORTED_SPORTS[sport.lower()]
    latest_path = os.path.join(
        base_dir, sport_config["data_dir"], f"{sport_config['file_prefix']}_latest.csv"
    )

    sport_status = fetch_status.get(sport.lower())
    if sport_status is None:
        return True, f"{sport.upper()}: no fetch status for this run; skipping staleness check"

    games_count = sport_status.get("games_count", 0)
    if not games_count:
        return True, f"{sport.upper()}: API reported 0 upcoming games (offseason); OK"

    newest = newest_timestamp_pulled(latest_path)
    if newest is None:
        return False, (
            f"{sport.upper()}: API reported {games_count} upcoming games but "
            f"{latest_path} is missing or has no usable '{TIMESTAMP_COLUMN}' data"
        )

    now = now or datetime.now(pytz.UTC).astimezone(EASTERN)
    age_days = (now - newest).total_seconds() / 86400.0

    if age_days > threshold_days:
        return False, (
            f"{sport.upper()}: STALE DATA - {latest_path} newest '{TIMESTAMP_COLUMN}' "
            f"is {age_days:.1f} days old (threshold {threshold_days}) while the API "
            f"reported {games_count} upcoming games. The latest file is not being updated."
        )

    return True, f"{sport.upper()}: data is {age_days:.1f} days old; OK"


def main():
    parser = argparse.ArgumentParser(description="Check odds data files for staleness")
    parser.add_argument(
        "--sport",
        choices=["nfl", "nba", "all"],
        default="all",
        help="Sport to check (default: all)",
    )
    parser.add_argument(
        "--threshold-days",
        type=int,
        default=DEFAULT_THRESHOLD_DAYS,
        help=f"Max age in days for _latest.csv data (default: {DEFAULT_THRESHOLD_DAYS})",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory for data files (default: current directory)",
    )
    parser.add_argument(
        "--status-file",
        default=None,
        help="Path to the fetch status JSON (default: <base-dir>/data/fetch_status.json)",
    )

    args = parser.parse_args()

    status_path = args.status_file or os.path.join(args.base_dir, "data", "fetch_status.json")
    fetch_status = load_fetch_status(status_path)

    if not fetch_status:
        print(f"Warning: no fetch status found at {status_path}; nothing to check.")
        return

    sports = list(SUPPORTED_SPORTS) if args.sport == "all" else [args.sport]

    failures = []
    for sport in sports:
        ok, message = check_sport_staleness(
            sport, fetch_status, args.base_dir, args.threshold_days
        )
        print(message)
        if not ok:
            failures.append(sport)

    if failures:
        names = ", ".join(s.upper() for s in failures)
        print(f"\nERROR: Stale odds data detected for: {names}")
        sys.exit(1)

    print("\nAll staleness checks passed.")


if __name__ == "__main__":
    main()
