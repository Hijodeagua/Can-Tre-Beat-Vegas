"""
Refresh the committed UEFA club-competition results (Champions League,
Europa League, Conference League) from openfootball/champions-league.

These are the cross-league glue: the only competitive matches where clubs
from different top-5 leagues meet, letting rating points flow between the
otherwise-closed league Elo pools. Coverage follows the upstream:
CL from 2014-15, EL from 2020-21, Conference from 2021-22.

Each team is mapped to a league pool via its (CCC) country code + the league
alias maps; clubs from outside the five tracked leagues (Porto, Ajax, …)
keep an empty league field and are ignored by the Elo glue. A club from a
top-5 country whose name can't be matched to the league's history is
reported loudly — that's an alias gap to patch in `leagues.UEFA_ALIASES`,
not a row to guess at.

Usage:
    python -m soccer.clubs.data.fetch_uefa
"""

import csv
import sys
from datetime import date
from pathlib import Path

import requests

from soccer.clubs.data import football_txt
from soccer.clubs.data.leagues import (
    COUNTRY_TO_LEAGUE,
    UEFA_ALIASES,
    canonical,
    next_season,
    season_for_date,
)

DATA_DIR = Path(__file__).resolve().parent
OUT_CSV = DATA_DIR / "uefa_results.csv"
RESULTS_CSV = DATA_DIR / "results.csv"
RAW_BASE = "https://raw.githubusercontent.com/openfootball/champions-league/master"

COMPETITIONS = {
    "ucl": ("cl", "2014-15"),
    "uel": ("el", "2020-21"),
    "uecl": ("conf", "2021-22"),
}

COLUMNS = [
    "date", "season", "competition", "round", "neutral",
    "home_team", "home_league", "away_team", "away_league",
    "home_score", "away_score",
]


def known_clubs() -> dict[str, set]:
    """League -> every club name results.csv has ever seen."""
    clubs: dict[str, set] = {}
    with RESULTS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clubs.setdefault(row["league"], set()).update(
                (row["home_team"], row["away_team"])
            )
    return clubs


def map_team(name: str, country: str | None, clubs: dict[str, set],
             unmatched: set) -> tuple[str, str]:
    """(canonical name, league key or "") for one UEFA entry."""
    league = COUNTRY_TO_LEAGUE.get(country or "")
    if not league:
        return name, ""
    resolved = canonical(league, UEFA_ALIASES.get(league, {}).get(name, name))
    if resolved in clubs.get(league, set()):
        return resolved, league
    unmatched.add((league, name))
    return name, ""


def is_neutral(round_name: str) -> bool:
    r = round_name.lower()
    return r == "final" or r.endswith(", final")


def fetch(today: date | None = None) -> list[dict]:
    today = today or date.today()
    current = season_for_date(today.isoformat())
    clubs = known_clubs()
    unmatched: set = set()
    rows: list[dict] = []

    for comp, (stem, first) in COMPETITIONS.items():
        season = first
        while season <= next_season(current):
            url = f"{RAW_BASE}/{season}/{stem}.txt"
            try:
                resp = requests.get(url, timeout=30)
            except requests.RequestException as exc:
                print(f"  ! {comp} {season}: fetch failed ({exc})")
                season = next_season(season)
                continue
            if resp.status_code == 404:
                season = next_season(season)
                continue
            resp.raise_for_status()
            matches = football_txt.parse(resp.text, season)
            n_tracked = 0
            for m in matches:
                if m.score1 is None:
                    continue
                home, home_lg = map_team(m.team1, m.country1, clubs, unmatched)
                away, away_lg = map_team(m.team2, m.country2, clubs, unmatched)
                if home_lg and away_lg:
                    n_tracked += 1
                rows.append(
                    {
                        "date": m.date,
                        "season": season,
                        "competition": comp,
                        "round": m.round,
                        "neutral": is_neutral(m.round),
                        "home_team": home,
                        "home_league": home_lg,
                        "away_team": away,
                        "away_league": away_lg,
                        "home_score": m.score1,
                        "away_score": m.score2,
                    }
                )
            print(f"  + {comp} {season}: {len(matches)} matches, {n_tracked} cross-league tracked")
            season = next_season(season)

    if unmatched:
        print("\n  ! UNMATCHED top-5-country clubs (patch UEFA_ALIASES):")
        for league, name in sorted(unmatched):
            print(f"      {league}: {name!r}")

    rows.sort(key=lambda r: (r["date"], r["competition"], r["home_team"]))
    return rows


def main() -> None:
    print("Fetching UEFA club competition results from openfootball…")
    rows = fetch()
    if not rows:
        print("Nothing fetched; keeping the committed uefa_results.csv untouched.")
        sys.exit(1)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    both = sum(1 for r in rows if r["home_league"] and r["away_league"])
    print(f"Wrote {len(rows)} rows ({both} with both clubs tracked) to {OUT_CSV}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
