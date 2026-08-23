"""
Refresh the committed per-match xG data (data/xg_matches.csv) from
understat.com — Champions of the "expected goals" metric for the five top
flights (EPL, La Liga, Bundesliga, Serie A, Ligue 1) back to 2014-15.
Understat does not cover the second divisions or MLS; those leagues simply
never appear in this file and the model's xG features 0-impute for them.

Understat has no API; each league-season page embeds the season's full
match list as a hex-escaped JSON blob (`var datesData = JSON.parse('…')`).
That blob carries per-match xG for both sides, which is all the model
needs — no shot-level scraping, one request per league-season.

NETWORK NOTE: understat.com is blocked by the sandbox proxy used for
development, so this fetcher can only actually run from the daily GitHub
Actions job (open egress). Same best-effort contract as every other
fetcher: any failure leaves the committed CSV (backfilled from the
archived worldfootballR_data Understat mirror, complete 2014-15 →
2025-01-04) untouched.

Merge posture: rewrite only the fetched (league, season-window) rows,
keyed by (league, date, home, away); everything else in the CSV survives.
By default only the two most recent Understat season labels are fetched —
history is already committed and doesn't change.

Usage:
    python -m soccer.clubs.data.fetch_xg [--from-season YYYY]
"""

import argparse
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent
OUT_CSV = DATA_DIR / "xg_matches.csv"

# our league key -> understat league slug
LEAGUES = {
    "epl": "EPL",
    "la_liga": "La_liga",
    "bundesliga": "Bundesliga",
    "serie_a": "Serie_A",
    "ligue_1": "Ligue_1",
}

# Understat team name -> our canonical name. Only names that differ after
# understat's short-name style is accounted for; grown as the fetcher
# reports unmatched names (it refuses to write a row it can't map, loudly).
UNDERSTAT_ALIASES: dict[str, dict[str, str]] = {
    "epl": {
        "Arsenal": "Arsenal FC", "Aston Villa": "Aston Villa FC",
        "Bournemouth": "AFC Bournemouth", "Brentford": "Brentford FC",
        "Brighton": "Brighton & Hove Albion FC", "Burnley": "Burnley FC",
        "Cardiff": "Cardiff City FC", "Chelsea": "Chelsea FC",
        "Coventry": "Coventry City FC",
        "Crystal Palace": "Crystal Palace FC", "Everton": "Everton FC",
        "Fulham": "Fulham FC", "Huddersfield": "Huddersfield Town AFC",
        "Hull": "Hull City AFC", "Ipswich": "Ipswich Town FC",
        "Leeds": "Leeds United FC", "Leicester": "Leicester City FC",
        "Liverpool": "Liverpool FC", "Luton": "Luton Town FC",
        "Manchester City": "Manchester City FC",
        "Manchester United": "Manchester United FC",
        "Middlesbrough": "Middlesbrough FC",
        "Newcastle United": "Newcastle United FC",
        "Norwich": "Norwich City FC", "Nottingham Forest": "Nottingham Forest FC",
        "Queens Park Rangers": "Queens Park Rangers FC",
        "Sheffield United": "Sheffield United FC", "Southampton": "Southampton FC",
        "Stoke": "Stoke City FC", "Sunderland": "Sunderland AFC",
        "Swansea": "Swansea City AFC", "Tottenham": "Tottenham Hotspur FC",
        "Watford": "Watford FC", "West Bromwich Albion": "West Bromwich Albion FC",
        "West Ham": "West Ham United FC", "Wolverhampton Wanderers": "Wolverhampton Wanderers FC",
    },
    "la_liga": {
        "Alaves": "Deportivo Alavés", "Almeria": "UD Almería",
        "Athletic Club": "Athletic Club", "Atletico Madrid": "Club Atlético de Madrid",
        "Barcelona": "FC Barcelona", "Cadiz": "Cádiz CF",
        "Celta Vigo": "RC Celta de Vigo", "Cordoba": "Córdoba CF",
        "Eibar": "SD Eibar", "Elche": "Elche CF", "Espanyol": "RCD Espanyol de Barcelona",
        "Getafe": "Getafe CF", "Girona": "Girona FC", "Granada": "Granada CF",
        "Las Palmas": "UD Las Palmas", "Leganes": "CD Leganés",
        "Levante": "Levante UD", "Malaga": "Málaga CF", "Mallorca": "RCD Mallorca",
        "Osasuna": "CA Osasuna", "Rayo Vallecano": "Rayo Vallecano de Madrid",
        "Real Betis": "Real Betis Balompié", "Real Madrid": "Real Madrid CF",
        "Real Oviedo": "Real Oviedo", "Real Sociedad": "Real Sociedad de Fútbol",
        "Real Valladolid": "Real Valladolid CF", "Sevilla": "Sevilla FC",
        "SD Huesca": "SD Huesca", "Sporting Gijon": "Sporting Gijón",
        "Racing Santander": "Real Racing Club de Santander",
        "Valencia": "Valencia CF", "Villarreal": "Villarreal CF",
        "Deportivo La Coruna": "RC Deportivo La Coruña",
    },
    "bundesliga": {
        "Arminia Bielefeld": "Arminia Bielefeld", "Augsburg": "FC Augsburg",
        "Bayer Leverkusen": "Bayer 04 Leverkusen", "Bayern Munich": "FC Bayern München",
        "Bochum": "VfL Bochum 1848", "Borussia Dortmund": "Borussia Dortmund",
        "Borussia M.Gladbach": "Borussia Mönchengladbach",
        "Darmstadt": "SV Darmstadt 98", "Eintracht Frankfurt": "Eintracht Frankfurt",
        "FC Cologne": "1. FC Köln", "FC Heidenheim": "1. FC Heidenheim 1846",
        "Fortuna Duesseldorf": "Fortuna Düsseldorf", "Freiburg": "SC Freiburg",
        "Greuther Fuerth": "SpVgg Greuther Fürth 1903", "Hamburger SV": "Hamburger SV",
        "Hannover 96": "Hannover 96", "Hertha Berlin": "Hertha BSC",
        "Hoffenheim": "TSG 1899 Hoffenheim", "Holstein Kiel": "Holstein Kiel",
        "Ingolstadt": "FC Ingolstadt 04", "Mainz 05": "1. FSV Mainz 05",
        "Nuernberg": "1. FC Nürnberg", "Paderborn": "SC Paderborn 07",
        "RasenBallsport Leipzig": "RB Leipzig", "Schalke 04": "FC Schalke 04",
        "St. Pauli": "FC St. Pauli 1910", "Union Berlin": "1. FC Union Berlin",
        "VfB Stuttgart": "VfB Stuttgart", "Werder Bremen": "SV Werder Bremen",
        "Elversberg": "SV 07 Elversberg",
        "Wolfsburg": "VfL Wolfsburg", "Eintracht Braunschweig": "Eintracht Braunschweig",
    },
    "serie_a": {
        "AC Milan": "AC Milan", "Atalanta": "Atalanta BC", "Benevento": "Benevento Calcio",
        "Bologna": "Bologna FC 1909", "Brescia": "Brescia Calcio",
        "Cagliari": "Cagliari Calcio", "Carpi": "Carpi FC",
        "Cesena": "Cesena FC", "Chievo": "Chievo Verona", "Como": "Como 1907",
        "Cremonese": "US Cremonese", "Crotone": "FC Crotone",
        "Empoli": "Empoli FC", "Fiorentina": "ACF Fiorentina",
        "Frosinone": "Frosinone Calcio", "Genoa": "Genoa CFC",
        "Inter": "FC Internazionale Milano", "Juventus": "Juventus FC",
        "Lazio": "SS Lazio", "Lecce": "US Lecce", "Monza": "AC Monza",
        "Napoli": "SSC Napoli", "Palermo": "Palermo FC",
        "Parma Calcio 1913": "Parma Calcio 1913", "Parma": "Parma Calcio 1913",
        "Pescara": "Delfino Pescara",
        "Pisa": "AC Pisa 1909", "Roma": "AS Roma", "Salernitana": "US Salernitana 1919",
        "Sampdoria": "UC Sampdoria", "Sassuolo": "US Sassuolo Calcio",
        "SPAL 2013": "SPAL 2013 Ferrara", "Spezia": "Spezia Calcio",
        "Torino": "Torino FC", "Udinese": "Udinese Calcio",
        "Venezia": "Venezia FC", "Verona": "Hellas Verona FC",
    },
    "ligue_1": {
        "Ajaccio": "AC Ajaccio", "Amiens": "Amiens SC", "Angers": "Angers SCO",
        "Auxerre": "AJ Auxerre", "Bordeaux": "Girondins Bordeaux",
        "Brest": "Stade Brestois 29", "Caen": "SM Caen",
        "Clermont Foot": "Clermont Foot 63", "Dijon": "Dijon FCO",
        "Guingamp": "EA Guingamp", "Le Havre": "Havre AC", "Lens": "Racing Club de Lens",
        "Lille": "Lille OSC", "Lorient": "FC Lorient", "Lyon": "Olympique Lyonnais",
        "Marseille": "Olympique de Marseille", "Metz": "FC Metz",
        "Monaco": "AS Monaco FC", "Montpellier": "Montpellier HSC",
        "Nantes": "FC Nantes", "Nancy": "AS Nancy Lorraine", "Nice": "OGC Nice",
        "Nimes": "Nîmes Olympique", "Paris FC": "Paris FC",
        "Paris Saint Germain": "Paris Saint-Germain FC", "Reims": "Stade de Reims",
        "Rennes": "Stade Rennais FC 1901", "Saint-Etienne": "AS Saint-Étienne",
        "Strasbourg": "RC Strasbourg Alsace", "Toulouse": "Toulouse FC",
        "Troyes": "ESTAC Troyes", "Bastia": "SC Bastia", "SC Bastia": "SC Bastia",
        "GFC Ajaccio": "Gazélec FC Ajaccio", "Evian Thonon Gaillard": "Évian Thonon Gaillard",
        "Le Mans": "Le Mans FC",
    },
}

DATES_RE = re.compile(r"datesData\s*=\s*JSON\.parse\('([^']*)'\)")


def _decode_blob(hex_escaped: str) -> list[dict]:
    raw = hex_escaped.encode().decode("unicode_escape").encode("latin1").decode("utf-8")
    return json.loads(raw)


def fetch_league_season(league_key: str, understat_season: int,
                        timeout: int = 30) -> list[dict] | None:
    """One understat league-season -> canonical match xG rows, or None."""
    slug = LEAGUES[league_key]
    url = f"https://understat.com/league/{slug}/{understat_season}"
    try:
        resp = requests.get(url, timeout=timeout,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ! {league_key} {understat_season}: fetch failed ({exc})")
        return None
    m = DATES_RE.search(resp.text)
    if not m:
        print(f"  ! {league_key} {understat_season}: no datesData blob on page")
        return None
    matches = _decode_blob(m.group(1))

    aliases = UNDERSTAT_ALIASES[league_key]
    rows, unmatched = [], set()
    for match in matches:
        if not match.get("isResult"):
            continue
        h, a = match["h"]["title"], match["a"]["title"]
        ch, ca = aliases.get(h), aliases.get(a)
        if ch is None:
            unmatched.add(h)
        if ca is None:
            unmatched.add(a)
        if ch is None or ca is None:
            continue
        rows.append({
            "league": league_key,
            "date": match["datetime"][:10],
            "home_team": ch,
            "away_team": ca,
            "xg_home": round(float(match["xG"]["h"]), 3),
            "xg_away": round(float(match["xG"]["a"]), 3),
        })
    for name in sorted(unmatched):
        print(f"  ! {league_key}: unmatched understat name {name!r} — "
              f"add it to UNDERSTAT_ALIASES")
    return rows


def merge(new_rows: list[dict], out: Path = OUT_CSV) -> list[dict]:
    """Committed rows with fetched matches replacing same-keyed rows."""
    fresh = {(r["league"], r["date"], r["home_team"], r["away_team"]): r
             for r in new_rows}
    kept: list[dict] = []
    if out.exists():
        with out.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r["league"], r["date"], r["home_team"], r["away_team"]) not in fresh:
                    kept.append(r)
    combined = kept + new_rows
    combined.sort(key=lambda r: (r["date"], r["league"], r["home_team"]))
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Understat labels a season by its starting year: 2026-27 is "2026".
    this_season = date.today().year if date.today().month >= 7 else date.today().year - 1
    parser.add_argument("--from-season", type=int, default=this_season - 1)
    args = parser.parse_args()

    print("Fetching per-match xG from understat.com…")
    all_rows: list[dict] = []
    ok = False
    for league_key in LEAGUES:
        for season in range(args.from_season, this_season + 1):
            rows = fetch_league_season(league_key, season)
            if rows is None:
                continue
            ok = True
            print(f"  + {league_key} {season}: {len(rows)} matches with xG")
            all_rows.extend(rows)
    if not ok or not all_rows:
        print("Nothing fetched; keeping the committed xg_matches.csv untouched.")
        sys.exit(1)
    combined = merge(all_rows)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["league", "date", "home_team", "away_team",
                           "xg_home", "xg_away"])
        writer.writeheader()
        writer.writerows(combined)
    print(f"Wrote {len(combined)} rows to {OUT_CSV}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
