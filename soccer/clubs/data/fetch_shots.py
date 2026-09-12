"""
Refresh the committed per-match shot counts (data/shots_matches.csv) from
football-data.co.uk — total shots and shots on target for both sides of
every top-5-league match, back to each league's first season in
results.csv.

Why shots when the model already has xG: the Understat feed
(`fetch_xg.py`) is the better chance-quality signal but it is a single
point of failure, and a stale one is worse than none (the xG feature
self-voids past MAX_AGE_DAYS and silently falls back to Elo-only). Shots
on target come from an independent publisher that posts within hours of
full time, reach back four seasons further than Understat's 2014-15
start, and correlate with xG strongly enough to carry most of the same
chance-creation information. The two layers are additive, not redundant:
`shots.py` features whichever is live.

Two upstream layers, same best-effort contract as every other fetcher (a
failed fetch keeps the committed rows for that league-season):

- football-data.co.uk — the publisher, current within hours of full time.
  This is the live layer the daily Actions job uses.
- datasets/football-datasets on GitHub — a public-domain mirror of the
  same files, one season per CSV, ISO dates. Used as the fallback, and as
  the source of the committed backfill: football-data.co.uk is blocked by
  the sandbox proxy used for development, raw.githubusercontent.com is
  not, so `--source mirror` is what a developer can actually run locally.
  The mirror lags the publisher by a season, which is exactly why it is
  the fallback rather than the primary.

Both layers publish the same columns: HS/AS (shots) and HST/AST (shots on
target), plus FTHG/FTAG, which the merge uses as a join check — a row
whose score disagrees with results.csv is dropped rather than written,
since that means the name mapping put it on the wrong match.

COVERAGE: top five flights only. football-data.co.uk does publish the
second divisions (E1, SP2, I1... ) but the mirror does not, so their name
mapping can't be derived or verified from here; those leagues 0-impute
through `shots.py` exactly as they already do for xG.

Usage:
    python -m soccer.clubs.data.fetch_shots [--source publisher|mirror|auto]
                                            [--from-season 2024-25]
"""

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

import requests

from soccer.clubs.data.leagues import LEAGUES, next_season, season_for_date

DATA_DIR = Path(__file__).resolve().parent
OUT_CSV = DATA_DIR / "shots_matches.csv"

PUBLISHER_BASE = "https://www.football-data.co.uk/mmz4281"
MIRROR_BASE = ("https://raw.githubusercontent.com/datasets/football-datasets"
               "/main/datasets")

# our league key -> (football-data.co.uk division code, mirror directory)
SOURCES = {
    "epl": ("E0", "premier-league"),
    "la_liga": ("SP1", "la-liga"),
    "bundesliga": ("D1", "bundesliga"),
    "serie_a": ("I1", "serie-a"),
    "ligue_1": ("F1", "ligue-1"),
}

COLUMNS = ["league", "season", "date", "home_team", "away_team",
           "shots_home", "shots_away", "sot_home", "sot_away"]

# football-data.co.uk short name -> our canonical name. Derived by joining
# the publisher's files against results.csv on (date +/- 1 day, final
# score) over every season both cover, keeping only unambiguous matches,
# then freezing the result here — so the fetcher never guesses at write
# time. A name missing from this table is reported and its rows skipped,
# same posture as fetch_xg.py's UNDERSTAT_ALIASES.
FD_ALIASES: dict[str, dict[str, str]] = {
    "epl": {
        'Arsenal': 'Arsenal FC',
        'Aston Villa': 'Aston Villa FC',
        'Birmingham': 'Birmingham City FC',
        'Blackburn': 'Blackburn Rovers FC',
        'Blackpool': 'Blackpool FC',
        'Bolton': 'Bolton Wanderers FC',
        'Bournemouth': 'AFC Bournemouth',
        'Brentford': 'Brentford FC',
        'Brighton': 'Brighton & Hove Albion FC',
        'Burnley': 'Burnley FC',
        'Cardiff': 'Cardiff City FC',
        'Chelsea': 'Chelsea FC',
        'Crystal Palace': 'Crystal Palace FC',
        'Everton': 'Everton FC',
        'Fulham': 'Fulham FC',
        'Huddersfield': 'Huddersfield Town AFC',
        'Hull': 'Hull City AFC',
        'Ipswich': 'Ipswich Town FC',
        'Leeds': 'Leeds United FC',
        'Leicester': 'Leicester City FC',
        'Liverpool': 'Liverpool FC',
        'Luton': 'Luton Town FC',
        'Man City': 'Manchester City FC',
        'Man United': 'Manchester United FC',
        'Middlesbrough': 'Middlesbrough FC',
        'Newcastle': 'Newcastle United FC',
        'Norwich': 'Norwich City FC',
        "Nott'm Forest": 'Nottingham Forest FC',
        'QPR': 'Queens Park Rangers FC',
        'Reading': 'Reading FC',
        'Sheffield United': 'Sheffield United FC',
        'Southampton': 'Southampton FC',
        'Stoke': 'Stoke City FC',
        'Sunderland': 'Sunderland AFC',
        'Swansea': 'Swansea City AFC',
        'Tottenham': 'Tottenham Hotspur FC',
        'Watford': 'Watford FC',
        'West Brom': 'West Bromwich Albion FC',
        'West Ham': 'West Ham United FC',
        'Wigan': 'Wigan Athletic FC',
        'Wolves': 'Wolverhampton Wanderers FC',
    },
    "la_liga": {
        'Alaves': 'Deportivo Alavés',
        'Almeria': 'UD Almería',
        'Ath Bilbao': 'Athletic Club',
        'Ath Madrid': 'Club Atlético de Madrid',
        'Barcelona': 'FC Barcelona',
        'Betis': 'Real Betis Balompié',
        'Cadiz': 'Cádiz CF',
        'Celta': 'RC Celta de Vigo',
        'Cordoba': 'Córdoba CF',
        'Eibar': 'SD Eibar',
        'Elche': 'Elche CF',
        'Espanol': 'RCD Espanyol de Barcelona',
        'Getafe': 'Getafe CF',
        'Girona': 'Girona FC',
        'Granada': 'Granada CF',
        'Huesca': 'SD Huesca',
        'La Coruna': 'RC Deportivo La Coruña',
        'Las Palmas': 'UD Las Palmas',
        'Leganes': 'CD Leganés',
        'Levante': 'Levante UD',
        'Malaga': 'Málaga CF',
        'Mallorca': 'RCD Mallorca',
        'Osasuna': 'CA Osasuna',
        'Oviedo': 'Real Oviedo',
        'Real Madrid': 'Real Madrid CF',
        'Sevilla': 'Sevilla FC',
        'Sociedad': 'Real Sociedad de Fútbol',
        'Sp Gijon': 'Sporting Gijón',
        'Valencia': 'Valencia CF',
        'Valladolid': 'Real Valladolid CF',
        'Vallecano': 'Rayo Vallecano de Madrid',
        'Villarreal': 'Villarreal CF',
        'Zaragoza': 'Real Zaragoza',
    },
    "bundesliga": {
        'Augsburg': 'FC Augsburg',
        'Bayern Munich': 'FC Bayern München',
        'Bielefeld': 'Arminia Bielefeld',
        'Bochum': 'VfL Bochum 1848',
        'Braunschweig': 'Eintracht Braunschweig',
        'Darmstadt': 'SV Darmstadt 98',
        'Dortmund': 'Borussia Dortmund',
        'Ein Frankfurt': 'Eintracht Frankfurt',
        'FC Koln': '1. FC Köln',
        'Fortuna Dusseldorf': 'Fortuna Düsseldorf',
        'Freiburg': 'SC Freiburg',
        'Greuther Furth': 'SpVgg Greuther Fürth 1903',
        'Hamburg': 'Hamburger SV',
        'Hannover': 'Hannover 96',
        'Heidenheim': '1. FC Heidenheim 1846',
        'Hertha': 'Hertha BSC',
        'Hoffenheim': 'TSG 1899 Hoffenheim',
        'Holstein Kiel': 'Holstein Kiel',
        'Ingolstadt': 'FC Ingolstadt 04',
        'Kaiserslautern': '1. FC Kaiserslautern',
        'Leverkusen': 'Bayer 04 Leverkusen',
        "M'gladbach": 'Borussia Mönchengladbach',
        'Mainz': '1. FSV Mainz 05',
        'Nurnberg': '1. FC Nürnberg',
        'Paderborn': 'SC Paderborn 07',
        'RB Leipzig': 'RB Leipzig',
        'Schalke 04': 'FC Schalke 04',
        'St Pauli': 'FC St. Pauli 1910',
        'Stuttgart': 'VfB Stuttgart',
        'Union Berlin': '1. FC Union Berlin',
        'Werder Bremen': 'SV Werder Bremen',
        'Wolfsburg': 'VfL Wolfsburg',
    },
    "serie_a": {
        'Atalanta': 'Atalanta BC',
        'Benevento': 'Benevento Calcio',
        'Bologna': 'Bologna FC 1909',
        'Brescia': 'Brescia Calcio',
        'Cagliari': 'Cagliari Calcio',
        'Carpi': 'Carpi FC',
        'Catania': 'Calcio Catania',
        'Cesena': 'Cesena FC',
        'Chievo': 'Chievo Verona',
        'Como': 'Como 1907',
        'Cremonese': 'US Cremonese',
        'Crotone': 'FC Crotone',
        'Empoli': 'Empoli FC',
        'Fiorentina': 'ACF Fiorentina',
        'Frosinone': 'Frosinone Calcio',
        'Genoa': 'Genoa CFC',
        'Inter': 'FC Internazionale Milano',
        'Juventus': 'Juventus FC',
        'Lazio': 'SS Lazio',
        'Lecce': 'US Lecce',
        'Livorno': 'AS Livorno',
        'Milan': 'AC Milan',
        'Monza': 'AC Monza',
        'Napoli': 'SSC Napoli',
        'Palermo': 'Palermo FC',
        'Parma': 'Parma Calcio 1913',
        'Pescara': 'Delfino Pescara',
        'Pisa': 'AC Pisa 1909',
        'Roma': 'AS Roma',
        'Salernitana': 'US Salernitana 1919',
        'Sampdoria': 'UC Sampdoria',
        'Sassuolo': 'US Sassuolo Calcio',
        'Spal': 'SPAL 2013 Ferrara',
        'Spezia': 'Spezia Calcio',
        'Torino': 'Torino FC',
        'Udinese': 'Udinese Calcio',
        'Venezia': 'Venezia FC',
        'Verona': 'Hellas Verona FC',
    },
    "ligue_1": {
        'Ajaccio': 'AC Ajaccio',
        'Ajaccio GFCO': 'Gazélec FC Ajaccio',
        'Amiens': 'Amiens SC',
        'Angers': 'Angers SCO',
        'Auxerre': 'AJ Auxerre',
        'Bastia': 'SC Bastia',
        'Bordeaux': 'Girondins Bordeaux',
        'Brest': 'Stade Brestois 29',
        'Caen': 'SM Caen',
        'Clermont': 'Clermont Foot 63',
        'Dijon': 'Dijon FCO',
        'Evian Thonon Gaillard': 'Évian Thonon Gaillard',
        'Guingamp': 'EA Guingamp',
        'Le Havre': 'Havre AC',
        'Lens': 'Racing Club de Lens',
        'Lille': 'Lille OSC',
        'Lorient': 'FC Lorient',
        'Lyon': 'Olympique Lyonnais',
        'Marseille': 'Olympique de Marseille',
        'Metz': 'FC Metz',
        'Monaco': 'AS Monaco FC',
        'Montpellier': 'Montpellier HSC',
        'Nancy': 'AS Nancy Lorraine',
        'Nantes': 'FC Nantes',
        'Nice': 'OGC Nice',
        'Nimes': 'Nîmes Olympique',
        'Paris FC': 'Paris FC',
        'Paris SG': 'Paris Saint-Germain FC',
        'Reims': 'Stade de Reims',
        'Rennes': 'Stade Rennais FC 1901',
        'St Etienne': 'AS Saint-Étienne',
        'Strasbourg': 'RC Strasbourg Alsace',
        'Toulouse': 'Toulouse FC',
        'Troyes': 'ESTAC Troyes',
    },}


def season_code(season: str) -> str:
    """"2024-25" -> "2425", the season stem both sources use."""
    return f"{season[2:4]}{season[-2:]}"


def seasons_for(league_key: str, today: date) -> list[str]:
    """That league's first season in results.csv through the current one."""
    current = season_for_date(today.isoformat())
    seasons = [LEAGUES[league_key].first_season]
    while seasons[-1] != current:
        seasons.append(next_season(seasons[-1]))
    return seasons


def _get(url: str, timeout: int) -> str | None:
    try:
        resp = requests.get(url, timeout=timeout,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException:
        return None
    return resp.text


def _iso_date(raw: str) -> str | None:
    """Both layers' date spellings: ISO from the mirror, dd/mm/yy or
    dd/mm/yyyy from the publisher."""
    raw = raw.strip()
    if len(raw) >= 10 and raw[4] == "-":
        return raw[:10]
    parts = raw.split("/")
    if len(parts) != 3:
        return None
    d, m, y = parts
    if len(y) == 2:
        y = f"20{y}"
    try:
        return date(int(y), int(m), int(d)).isoformat()
    except ValueError:
        return None


def parse_csv(text: str, league: str, season: str) -> tuple[list[dict], set[str]]:
    """Shot rows from one season file, plus the names we couldn't map."""
    aliases = FD_ALIASES.get(league, {})
    rows, unmatched = [], set()
    # Both layers ship a UTF-8 BOM on some seasons and trailing blank lines
    # on most; DictReader over splitlines handles both.
    for r in csv.DictReader(text.lstrip("﻿").splitlines()):
        home, away = (r.get("HomeTeam") or "").strip(), (r.get("AwayTeam") or "").strip()
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
            shots_home, shots_away = int(r["HS"]), int(r["AS"])
            sot_home, sot_away = int(r["HST"]), int(r["AST"])
            fthg, ftag = int(r["FTHG"]), int(r["FTAG"])
        except (KeyError, TypeError, ValueError):
            continue    # a postponed or not-yet-played row carries blanks
        if sot_home > shots_home or sot_away > shots_away:
            # Self-contradicting row — upstream typo (one in ~25k: West
            # Ham at Newcastle, 2021-08-15, 8 shots and 9 on target). A
            # row that disagrees with itself can't be repaired from here,
            # so drop the match rather than feed a bad count to the form.
            continue
        rows.append({
            "league": league,
            "season": season,
            "date": iso,
            "home_team": ch,
            "away_team": ca,
            "shots_home": shots_home,
            "shots_away": shots_away,
            "sot_home": sot_home,
            "sot_away": sot_away,
            # kept only for the score cross-check, stripped before writing
            "_score": (fthg, ftag),
        })
    return rows, unmatched


def _results_index() -> dict[tuple, tuple[str, int, int]]:
    """(league, season, home, away) -> (date, home_score, away_score) from
    the committed results.csv. A pairing occurs once per season in a
    double round-robin, so the key is unique."""
    import pandas as pd

    res = pd.read_csv(DATA_DIR / "results.csv").dropna(
        subset=["home_score", "away_score"])
    return {
        (r.league, r.season, r.home_team, r.away_team):
            (r.date, int(r.home_score), int(r.away_score))
        for r in res.itertuples()
    }


def align(rows: list[dict], index: dict[tuple, tuple[str, int, int]]) -> tuple[list[dict], int, int]:
    """Snap each row onto its results.csv match and verify the score.

    The two publishers disagree about the calendar date of a late kickoff
    by up to a day, so the date is taken from results.csv rather than
    trusted — `shots.py` joins on (league, date, home, away) and an
    off-by-one date is a silent miss. A row whose score disagrees with the
    committed result is dropped: that means the name mapping landed it on
    the wrong match, and a wrong row is worse than a missing one."""
    kept, unknown, mismatched = [], 0, 0
    for row in rows:
        hit = index.get((row["league"], row["season"],
                         row["home_team"], row["away_team"]))
        if hit is None:
            unknown += 1
            continue
        res_date, hs, as_ = hit
        if (hs, as_) != row.pop("_score"):
            mismatched += 1
            continue
        row["date"] = res_date
        kept.append(row)
    return kept, unknown, mismatched


def fetch_league_season(league: str, season: str, source: str,
                        timeout: int = 30) -> tuple[list[dict], set[str]] | None:
    """One league-season's shot rows, or None if no layer served it."""
    code, slug = SOURCES[league]
    urls = []
    if source in ("auto", "publisher"):
        urls.append(f"{PUBLISHER_BASE}/{season_code(season)}/{code}.csv")
    if source in ("auto", "mirror"):
        urls.append(f"{MIRROR_BASE}/{slug}/season-{season_code(season)}.csv")
    for url in urls:
        text = _get(url, timeout)
        if text is None:
            continue
        rows, unmatched = parse_csv(text, league, season)
        if rows or unmatched:
            return rows, unmatched
    return None


def merge(new_rows: list[dict], out: Path = OUT_CSV) -> list[dict]:
    """Committed rows with fetched league-seasons replacing their own.

    Replacement is by (league, season), not by match: a season refetched
    mid-way through has fewer rows than the committed copy only if the
    upstream file itself shrank, and rewriting the window wholesale is how
    a corrected upstream score reaches us."""
    refetched = {(r["league"], r["season"]) for r in new_rows}
    kept: list[dict] = []
    if out.exists():
        with out.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r["league"], r["season"]) not in refetched:
                    kept.append(r)
    combined = kept + new_rows
    combined.sort(key=lambda r: (r["date"], r["league"], r["home_team"]))
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("auto", "publisher", "mirror"),
                        default="auto")
    parser.add_argument("--from-season", default=None,
                        help="earliest season to refetch (default: the "
                             "current one — history is already committed)")
    args = parser.parse_args()

    today = date.today()
    index = _results_index()
    print(f"Fetching per-match shots ({args.source})…")
    all_rows: list[dict] = []
    served = False
    for league in SOURCES:
        seasons = seasons_for(league, today)
        if args.from_season:
            seasons = [s for s in seasons if s >= args.from_season]
        else:
            seasons = seasons[-1:]
        for season in seasons:
            got = fetch_league_season(league, season, args.source)
            if got is None:
                print(f"  ! {league} {season}: no layer served this season")
                continue
            served = True
            rows, unmatched = got
            for name in sorted(unmatched):
                print(f"  ! {league}: unmapped name {name!r} — "
                      f"add it to FD_ALIASES")
            rows, unknown, mismatched = align(rows, index)
            note = ""
            if unknown or mismatched:
                note = f" ({unknown} not in results.csv, {mismatched} score mismatch)"
            print(f"  + {league} {season}: {len(rows)} matches with shots{note}")
            all_rows.extend(rows)

    if not served or not all_rows:
        print("Nothing fetched; keeping the committed shots_matches.csv untouched.")
        sys.exit(1)
    combined = merge(all_rows)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(combined)
    print(f"Wrote {len(combined)} rows to {OUT_CSV}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
