"""
Refresh the committed MLS match results from philo92/mls-elo — a single CSV,
1996 to present, updated after every match day (CC-BY 4.0). Team names are
already unified to each club's current name across all history, so unlike
the European fetchers this needs no per-season alias resolution; `canonical`
is still applied for consistency with the rest of the pipeline (a no-op
today, forward compatible with a future rename).

Only completed matches: the source is an Elo-history log, not a fixture
list, so there is no upcoming-MLS-game data to merge in — the daily slate
and rest-of-season Monte Carlo naturally have nothing to show for "mls"
(both already skip a league with no unplayed rows). This is a ratings +
squad-economics integration, not a daily-picks one.

Unlike fetch_results.py (which owns results.csv outright and rewrites it
whole), this script MERGES: it only ever touches rows with league == "mls",
leaving every other league's rows exactly as they were. That makes fetch
order in the daily pipeline safe either way, but fetch_results.py's own
rewrite doesn't know about MLS rows and would drop them if it ran after
this script — so refresh_data() always runs this one second.

Usage:
    python -m soccer.clubs.data.fetch_mls
"""

import csv
import sys
from pathlib import Path

import requests

from soccer.clubs.data.leagues import canonical

DATA_DIR = Path(__file__).resolve().parent
OUT_CSV = DATA_DIR / "results.csv"
SOURCE_URL = "https://raw.githubusercontent.com/philo92/mls-elo/main/results.csv"
LEAGUE_KEY = "mls"
FIRST_SEASON = "2013"  # matches LEAGUES["mls"].first_season

COLUMNS = [
    "date", "season", "league", "home_team", "away_team",
    "home_score", "away_score",
]


def fetch(timeout: int = 30) -> list[dict] | None:
    try:
        resp = requests.get(SOURCE_URL, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ! mls: fetch failed ({exc})")
        return None

    rows = []
    for r in csv.DictReader(resp.text.splitlines()):
        date = r["date"]
        if date < f"{FIRST_SEASON}-01-01":
            continue
        if not r.get("home_score") or not r.get("away_score"):
            continue  # shouldn't happen in this source, but stay defensive
        rows.append({
            "date": date,
            "season": date[:4],
            "league": LEAGUE_KEY,
            "home_team": canonical(LEAGUE_KEY, r["home_team"]),
            "away_team": canonical(LEAGUE_KEY, r["away_team"]),
            "home_score": r["home_score"],
            "away_score": r["away_score"],
        })
    return rows


def merge(new_rows: list[dict], out: Path = OUT_CSV) -> list[dict]:
    """Existing results.csv with every "mls" row replaced by `new_rows`."""
    kept: list[dict] = []
    if out.exists():
        with out.open(encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f) if r["league"] != LEAGUE_KEY]
    combined = kept + new_rows
    combined.sort(key=lambda r: (r["date"], r["league"], r["home_team"]))
    return combined


def write(rows: list[dict], out: Path = OUT_CSV) -> None:
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("Fetching MLS results from philo92/mls-elo…")
    rows = fetch()
    if rows is None or not rows:
        print("Nothing fetched; keeping the committed results.csv untouched.")
        sys.exit(1)
    combined = merge(rows)
    write(combined)
    print(f"  + mls {FIRST_SEASON}–present: {len(rows)} matches")
    print(f"Wrote {len(combined)} total rows to {OUT_CSV}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
