"""
Refresh `fd_results.csv`: final scores from football-data.co.uk for the
five top flights and the Championship.

Why a second results source: openfootball, the fixture backbone of
`results.csv`, stalled on 2026-09-20. Understat (`results_overlay.py`)
fills the top flights from then on, but it does not publish the
Championship. football-data.co.uk does - the E1 file sits next to the E0
file the shots feed (`fetch_shots.py`) has pulled from the same publisher
for years, under the same free-to-download terms, and it updates within a
day of each round. This module fetches those files, maps the publisher's
short club names onto the canonical ones, and commits the result so the
overlay step can read it offline and the merge is reproducible from the
repo alone.

Same best-effort contract as every fetcher here: a league-season that
fails to download keeps its committed rows. A club name the alias table
does not know is reported loudly and its rows are skipped - never written
under a guessed spelling - and `results_overlay.py` would list any such
row as unmatched rather than appending a duplicate fixture.

NETWORK NOTE: www.football-data.co.uk is blocked by the sandbox proxy
used for development, so this only runs for real from the Actions job;
the parser is tested on a fixture of the publisher's file format.

    python -m soccer.clubs.data.fd_results                 # current season
    python -m soccer.clubs.data.fd_results --from-season 2025-26
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

from soccer.clubs.data.fetch_shots import FD_ALIASES, _get, _iso_date, season_code
from soccer.clubs.data.leagues import canonical, next_season, season_for_date

DATA_DIR = Path(__file__).resolve().parent
OUT_CSV = DATA_DIR / "fd_results.csv"
PUBLISHER_BASE = "https://www.football-data.co.uk/mmz4281"

# our league key -> football-data.co.uk division code
DIVISIONS = {
    "epl": "E0",
    "championship": "E1",
    "la_liga": "SP1",
    "bundesliga": "D1",
    "serie_a": "I1",
    "ligue_1": "F1",
}

COLUMNS = ["league", "season", "date", "home_team", "away_team",
           "home_score", "away_score"]

# football-data.co.uk short name -> canonical name, Championship. The
# publisher spells a club the same way in every division, so the Premier
# League table from the shots fetcher covers every club that has been in
# both; the rest are the publisher's long-standing spellings for the
# second-tier regulars. An unlisted name is reported, not guessed.
CHAMPIONSHIP_ALIASES: dict[str, str] = {
    **FD_ALIASES.get("epl", {}),
    "Barnsley": "Barnsley FC",
    "Blackpool": "Blackpool FC",
    "Bolton": "Bolton Wanderers FC",
    "Bradford": "Bradford City AFC",
    "Bristol City": "Bristol City FC",
    "Charlton": "Charlton Athletic FC",
    "Coventry": "Coventry City FC",
    "Derby": "Derby County FC",
    "Lincoln": "Lincoln City FC",
    "Millwall": "Millwall FC",
    "Oxford": "Oxford United FC",
    "Peterboro": "Peterborough United FC",
    "Plymouth": "Plymouth Argyle FC",
    "Portsmouth": "Portsmouth FC",
    "Preston": "Preston North End FC",
    "Rotherham": "Rotherham United FC",
    "Sheffield Weds": "Sheffield Wednesday FC",
    "Wigan": "Wigan Athletic FC",
    "Wolves": "Wolverhampton Wanderers FC",
    "Wrexham": "Wrexham AFC",
    "Wycombe": "Wycombe Wanderers FC",
}


def aliases_for(league: str) -> dict[str, str]:
    if league == "championship":
        return CHAMPIONSHIP_ALIASES
    return FD_ALIASES.get(league, {})


def parse_csv(text: str, league: str, season: str) -> tuple[list[dict], set[str]]:
    """Final-score rows from one publisher file, plus unmapped names."""
    aliases = aliases_for(league)
    rows, unmatched = [], set()
    for r in csv.DictReader(text.lstrip("﻿").splitlines()):
        home = (r.get("HomeTeam") or "").strip()
        away = (r.get("AwayTeam") or "").strip()
        if not home or not away:
            continue
        iso = _iso_date(r.get("Date") or "")
        if iso is None:
            continue
        ch, ca = aliases.get(home), aliases.get(away)
        if ch is None:
            unmatched.add(home)
        if ca is None:
            unmatched.add(away)
        if ch is None or ca is None:
            continue
        try:
            fthg, ftag = int(r["FTHG"]), int(r["FTAG"])
        except (KeyError, TypeError, ValueError):
            continue    # not played yet / blank row
        rows.append({
            "league": league, "season": season, "date": iso,
            "home_team": canonical(league, ch), "away_team": canonical(league, ca),
            "home_score": fthg, "away_score": ftag,
        })
    return rows, unmatched


def seasons_for(today: date) -> list[str]:
    return [season_for_date(today.isoformat())]


def fetch_league_season(league: str, season: str, timeout: int = 30) -> str | None:
    url = f"{PUBLISHER_BASE}/{season_code(season)}/{DIVISIONS[league]}.csv"
    return _get(url, timeout)


def merge(new_rows: list[dict], out: Path = OUT_CSV) -> list[dict]:
    """Committed rows with refetched league-seasons replaced wholesale."""
    refetched = {(r["league"], r["season"]) for r in new_rows}
    kept: list[dict] = []
    if out.exists():
        with out.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r["league"], r["season"]) not in refetched:
                    kept.append(r)
    combined = kept + [{k: str(v) for k, v in r.items()} for r in new_rows]
    combined.sort(key=lambda r: (r["date"], r["league"], r["home_team"]))
    return combined


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-season", default=None,
                    help="earliest season to fetch (default: the current one)")
    ap.add_argument("--leagues", nargs="*", default=list(DIVISIONS))
    args = ap.parse_args(argv)

    today = date.today()
    current = season_for_date(today.isoformat())
    seasons = [current]
    if args.from_season:
        s = args.from_season
        seasons = []
        while s <= current:
            seasons.append(s)
            s = next_season(s)

    print("Fetching finals from football-data.co.uk…")
    all_rows: list[dict] = []
    served = False
    for league in args.leagues:
        for season in seasons:
            text = fetch_league_season(league, season)
            if text is None:
                print(f"  ! {league} {season}: fetch failed; keeping committed rows")
                continue
            served = True
            rows, unmatched = parse_csv(text, league, season)
            for name in sorted(unmatched):
                print(f"  ! {league}: unmapped name {name!r} - add it to the alias table")
            newest = max((r["date"] for r in rows), default="-")
            print(f"  + {league} {season}: {len(rows)} finals (newest {newest})")
            all_rows.extend(rows)
    if not served or not all_rows:
        print("Nothing fetched; keeping the committed fd_results.csv untouched.")
        return 1
    combined = merge(all_rows)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(combined)
    print(f"Wrote {len(combined)} rows to {OUT_CSV}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
