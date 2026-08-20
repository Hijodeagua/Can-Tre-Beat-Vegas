"""
Refresh the committed top-5-league club results from openfootball.

Source: openfootball/football.json (public domain), one JSON per league per
season, `score.ft` filled in as matches are played. Same posture as
`soccer/data/fetch_results.py`: network-gated and best-effort — a failed
fetch keeps the committed copy of that season and moves on.

Every run walks each league from its first upstream season through the
current one *plus one probe season beyond*, so a newly published season file
(e.g. 2026-27 once openfootball adds it) starts flowing in with no code
change. Team names are canonicalized via `leagues.ALIASES` before writing.

Rows without a final score are kept only for the current season (upcoming
fixtures, usable by a future predict step); past-season scoreless rows are
abandoned matches (COVID-cut Ligue 1 2019-20) and are dropped.

Usage:
    python -m soccer.clubs.data.fetch_results
"""

import csv
import sys
from datetime import date
from pathlib import Path

import requests

from soccer.clubs.data.leagues import (
    LEAGUES,
    canonical,
    next_season,
    season_for_date,
)

DATA_DIR = Path(__file__).resolve().parent
OUT_CSV = DATA_DIR / "results.csv"
RAW_BASE = "https://raw.githubusercontent.com/openfootball/football.json/master"

COLUMNS = [
    "date", "season", "league", "home_team", "away_team",
    "home_score", "away_score",
]


def seasons_for(league_key: str, today: date) -> list[str]:
    """First upstream season through the season after the current one."""
    current = season_for_date(today.isoformat())
    seasons = [LEAGUES[league_key].first_season]
    while seasons[-1] != next_season(current):
        seasons.append(next_season(seasons[-1]))
    return seasons


def fetch_season(league_key: str, season: str, timeout: int = 30) -> list[dict] | None:
    """One league-season -> normalized rows, or None if unavailable."""
    lg = LEAGUES[league_key]
    url = f"{RAW_BASE}/{season}/{lg.code}.json"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        matches = resp.json()["matches"]
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"  ! {league_key} {season}: fetch failed ({exc})")
        return None

    rows = []
    for m in matches:
        score = m.get("score") or {}
        # Two upstream shapes: {"ft": [h, a], ...} normally, but a bare
        # [0, 0] list for 0-0 finals in the newest season files.
        ft = score.get("ft") if isinstance(score, dict) else score
        played = isinstance(ft, (list, tuple)) and len(ft) == 2
        rows.append(
            {
                "date": m["date"],
                "season": season,
                "league": league_key,
                "home_team": canonical(league_key, m["team1"]),
                "away_team": canonical(league_key, m["team2"]),
                "home_score": ft[0] if played else "",
                "away_score": ft[1] if played else "",
            }
        )
    return rows


def fetch(today: date | None = None) -> list[dict]:
    today = today or date.today()
    current = season_for_date(today.isoformat())
    all_rows: list[dict] = []
    for key in LEAGUES:
        league_rows: list[dict] = []
        for season in seasons_for(key, today):
            rows = fetch_season(key, season)
            if rows is None:
                # Expected for seasons openfootball hasn't published yet;
                # noisy only if a *past* season goes missing.
                if season <= current:
                    print(f"  ! {key} {season}: missing upstream")
                continue
            if season < current:
                rows = [r for r in rows if r["home_score"] != ""]
            played = sum(1 for r in rows if r["home_score"] != "")
            print(f"  + {key} {season}: {len(rows)} matches ({played} played)")
            league_rows.extend(rows)
        all_rows.extend(league_rows)
    all_rows.sort(key=lambda r: (r["date"], r["league"], r["home_team"]))
    return all_rows


def write(rows: list[dict], out: Path = OUT_CSV) -> None:
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("Fetching top-5 league club results from openfootball…")
    rows = fetch()
    played = sum(1 for r in rows if r["home_score"] != "")
    if played == 0:
        print("Nothing fetched; keeping the committed results.csv untouched.")
        sys.exit(1)
    write(rows)
    print(f"Wrote {len(rows)} rows ({played} played) to {OUT_CSV}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
