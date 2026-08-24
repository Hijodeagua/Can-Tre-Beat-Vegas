"""
Refresh the committed top-5-league club results from openfootball.

Two upstream layers, both public domain, both best-effort (a failed fetch
keeps the committed copy of that season and moves on):

- openfootball/football.json — one JSON per league per season; the
  historical backbone.
- the openfootball country repos (england, deutschland, espana, italy,
  france) in Football.TXT format — these publish each new season's fixtures
  well before football.json does, so they are the fallback for any season
  the JSON layer doesn't have yet. That's what makes the daily runner live
  during 2026-27 while `2026-27/en.1.json` is still unpublished.

Every run walks each league from its first upstream season through the
current one *plus one probe season beyond*. Team names are canonicalized
via `leagues.ALIASES` before writing.

Rows without a final score are kept only for the current season onward
(upcoming fixtures — the daily slate); past-season scoreless rows are
abandoned matches (COVID-cut Ligue 1 2019-20) and are dropped.

Usage:
    python -m soccer.clubs.data.fetch_results
"""

import csv
import sys
from datetime import date
from pathlib import Path

import requests

from soccer.clubs.data import football_txt
from soccer.clubs.data.leagues import (
    LEAGUES,
    canonical,
    next_season,
    season_for_date,
)

DATA_DIR = Path(__file__).resolve().parent
OUT_CSV = DATA_DIR / "results.csv"
RAW_BASE = "https://raw.githubusercontent.com/openfootball/football.json/master"
TXT_BASE = "https://raw.githubusercontent.com/openfootball"

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


def _row(league_key: str, season: str, date: str, team1: str, team2: str,
         s1, s2) -> dict:
    played = s1 is not None and s2 is not None
    return {
        "date": date,
        "season": season,
        "league": league_key,
        "home_team": canonical(league_key, team1),
        "away_team": canonical(league_key, team2),
        "home_score": s1 if played else "",
        "away_score": s2 if played else "",
    }


def fetch_season_json(league_key: str, season: str, timeout: int = 30) -> list[dict] | None:
    """One league-season from football.json, or None if unavailable."""
    lg = LEAGUES[league_key]
    url = f"{RAW_BASE}/{season}/{lg.code}.json"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        matches = resp.json()["matches"]
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"  ! {league_key} {season}: json fetch failed ({exc})")
        return None

    rows = []
    for m in matches:
        score = m.get("score") or {}
        # Two upstream shapes: {"ft": [h, a], ...} normally, but a bare
        # [0, 0] list for 0-0 finals in the newest season files.
        ft = score.get("ft") if isinstance(score, dict) else score
        played = isinstance(ft, (list, tuple)) and len(ft) == 2
        rows.append(
            _row(league_key, season, m["date"], m["team1"], m["team2"],
                 ft[0] if played else None, ft[1] if played else None)
        )
    return rows


def fetch_season_txt(league_key: str, season: str, timeout: int = 30) -> list[dict] | None:
    """One league-season from the country Football.TXT repo, or None."""
    lg = LEAGUES[league_key]
    url = f"{TXT_BASE}/{lg.txt_repo}/master/{season}/{lg.txt_file}.txt"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        matches = football_txt.parse(resp.text, season)
    except requests.RequestException as exc:
        print(f"  ! {league_key} {season}: txt fetch failed ({exc})")
        return None
    if not matches:
        return None
    return [
        _row(league_key, season, m.date, m.team1, m.team2, m.score1, m.score2)
        for m in matches
    ]


def fetch_season(league_key: str, season: str) -> tuple[list[dict] | None, str]:
    """Fetch both layers and keep whichever has more played matches — the
    json layer occasionally stalls mid-season (2. Bundesliga 2025-26) while
    the country txt repo is complete, and vice versa for old seasons the
    json layer never got (Championship 2016-18)."""
    json_rows = fetch_season_json(league_key, season)
    txt_rows = fetch_season_txt(league_key, season)

    def played(rows):
        return sum(1 for r in rows if r["home_score"] != "") if rows else -1

    if json_rows is None and txt_rows is None:
        return None, "none"
    if played(txt_rows) > played(json_rows):
        return txt_rows, "txt"
    return json_rows, "json"


def fetch(today: date | None = None) -> list[dict]:
    today = today or date.today()
    current = season_for_date(today.isoformat())
    all_rows: list[dict] = []
    for key in LEAGUES:
        if LEAGUES[key].source != "openfootball":
            continue  # e.g. "mls" — fetched separately by fetch_mls.py
        league_rows: list[dict] = []
        for season in seasons_for(key, today):
            rows, src = fetch_season(key, season)
            if rows is None:
                # Expected for seasons openfootball hasn't published yet;
                # noisy only if a *past* season goes missing.
                if season <= current:
                    print(f"  ! {key} {season}: missing upstream")
                continue
            if season < current:
                rows = [r for r in rows if r["home_score"] != ""]
            played = sum(1 for r in rows if r["home_score"] != "")
            print(f"  + {key} {season} [{src}]: {len(rows)} matches ({played} played)")
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
