"""
Understat: the free xG / npxG / PPDA / deep-completion feed for the five
top flights, from 2014-15 on.

Two layers, kept apart on purpose:

- **Fetch** (`fetch_league_season`) hits the JSON league endpoint,
  `https://understat.com/getLeagueData/{slug}/{season}`, with the
  XMLHttpRequest headers the site expects, and caches the raw response
  under `data/soccer_clubs/raw/understat/` (gitignored). If the endpoint
  ever answers with the league HTML page instead, the legacy embedded
  blobs (`datesData = JSON.parse('…')`, `teamsData = …`) are parsed from
  it — the shape the old scraper depended on — so a format change
  degrades to the old behaviour rather than to nothing.
- **Parse** (`parse_payload`, `build_rows`) is shape-tolerant and pure:
  it accepts the JSON keys under any of the spellings Understat has used
  (`dates` / `datesData` / `matches`, `teams` / `teamsData`), values as
  objects or as JSON-encoded strings, and turns them into one processed
  row per match carrying both sides' xG, npxG, xPts, PPDA and deep
  completions. It is tested on fixtures of every shape, which is how it
  can be trusted from a sandbox that cannot reach the site.

NETWORK NOTE: understat.com is blocked by the sandbox proxy used for
development, so the fetch layer only actually runs from the GitHub
Actions job. `--probe` prints the live response's shape (top-level type,
keys, one sample) without writing anything — run it once from Actions
after any change here, because the endpoint's exact shape is confirmed
there, not assumed here.

Processed output: `soccer/clubs/data/understat_matches.csv`, one row per
completed match, canonical club names, and — for compatibility — the
legacy `xg_matches.csv` that `model/xg.py` still reads.

    python -m soccer.clubs.data.understat                 # this + last season
    python -m soccer.clubs.data.understat --all           # 2014-15 onward
    python -m soccer.clubs.data.understat --probe epl 2025
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parents[2]
RAW_DIR = REPO_ROOT / "data" / "soccer_clubs" / "raw" / "understat"
OUT_CSV = DATA_DIR / "understat_matches.csv"
LEGACY_XG_CSV = DATA_DIR / "xg_matches.csv"

FIRST_SEASON = 2014          # Understat's first season for all five leagues
ENDPOINT = "https://understat.com/getLeagueData/{slug}/{season}"
PAGE = "https://understat.com/league/{slug}/{season}"

COLUMNS = [
    "league", "season", "date", "match_id", "home_team", "away_team",
    "goals_home", "goals_away",
    "xg_home", "xg_away", "npxg_home", "npxg_away",
    "xpts_home", "xpts_away",
    "ppda_att_home", "ppda_def_home", "ppda_att_away", "ppda_def_away",
    "deep_home", "deep_away",
]

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


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------
def headers(slug: str, season: int) -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": PAGE.format(slug=slug, season=season),
    }


def season_label(understat_season: int) -> str:
    """Understat labels a season by its starting year: 2024 -> "2024-25"."""
    return f"{understat_season}-{str(understat_season + 1)[-2:]}"


def current_understat_season(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def raw_path(league_key: str, season: int) -> Path:
    return RAW_DIR / f"{league_key}_{season}.json"


def fetch_league_season(league_key: str, season: int, timeout: int = 30,
                        session: requests.Session | None = None) -> dict | None:
    """Raw payload for one league-season, or None if unreachable. Tries the
    JSON endpoint first; a response that turns out to be the HTML page is
    parsed for its embedded blobs. Whatever came back is cached raw."""
    slug = LEAGUES[league_key]
    get = (session or requests).get
    try:
        resp = get(ENDPOINT.format(slug=slug, season=season), timeout=timeout,
                   headers=headers(slug, season))
        resp.raise_for_status()
        payload = _payload_from_response(resp.text)
        if payload is None:
            # Endpoint answered with something we can't read: fall back
            # to the page and its embedded blobs.
            resp = get(PAGE.format(slug=slug, season=season), timeout=timeout,
                       headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            payload = _payload_from_response(resp.text)
    except requests.RequestException as exc:
        print(f"  ! {league_key} {season}: fetch failed ({exc})")
        return None
    if payload is None:
        print(f"  ! {league_key} {season}: response carried neither JSON nor the page blobs")
        return None
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path(league_key, season).write_text(json.dumps(payload), encoding="utf-8")
    return payload


BLOB_RE = re.compile(r"(datesData|teamsData|playersData)\s*=\s*JSON\.parse\('([^']*)'\)")


def _decode_blob(hex_escaped: str):
    raw = hex_escaped.encode().decode("unicode_escape").encode("latin1").decode("utf-8")
    return json.loads(raw)


def _payload_from_response(text: str) -> dict | None:
    """JSON body -> dict; HTML page -> dict of its embedded blobs; else None."""
    stripped = text.lstrip()
    if stripped[:1] in "{[":
        try:
            body = json.loads(stripped)
        except json.JSONDecodeError:
            body = None
        if isinstance(body, dict):
            return body
        if isinstance(body, list):
            return {"dates": body}
    blobs = {name: _decode_blob(blob) for name, blob in BLOB_RE.findall(text)}
    return blobs or None


# --------------------------------------------------------------------------
# parse
# --------------------------------------------------------------------------
_DATES_KEYS = ("dates", "datesData", "matches", "results")
_TEAMS_KEYS = ("teams", "teamsData")


def _maybe_json(value):
    """Understat has shipped nested values as JSON-encoded strings."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def parse_payload(payload: dict) -> tuple[list[dict], dict]:
    """(matches, teams) out of any payload shape this module accepts.

    matches: Understat match dicts (id, datetime, h/a titles, xG, goals,
    isResult). teams: {title: [history rows]} — the per-team, per-match
    lines carrying npxG, PPDA, deep, xPts. Either may be empty.
    """
    matches: list[dict] = []
    for key in _DATES_KEYS:
        if key in payload:
            value = _maybe_json(payload[key])
            if isinstance(value, dict):
                value = list(value.values())
            matches = [m for m in value if isinstance(m, dict)]
            break

    teams: dict[str, list[dict]] = {}
    for key in _TEAMS_KEYS:
        if key in payload:
            value = _maybe_json(payload[key])
            entries = value.values() if isinstance(value, dict) else value
            for entry in entries:
                entry = _maybe_json(entry)
                if not isinstance(entry, dict):
                    continue
                title = entry.get("title") or entry.get("name")
                history = _maybe_json(entry.get("history", []))
                if title and isinstance(history, list):
                    teams[title] = [_maybe_json(h) for h in history]
            break
    return matches, teams


def _num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _ppda(value) -> tuple[float | None, float | None]:
    value = _maybe_json(value)
    if isinstance(value, dict):
        return _num(value.get("att")), _num(value.get("def"))
    return None, None


def _team_lines(teams: dict[str, list[dict]]) -> dict[tuple[str, str, str], dict]:
    """(title, ISO date, 'h'|'a') -> that team's history line."""
    out = {}
    for title, history in teams.items():
        for line in history:
            d = str(line.get("date", ""))[:10]
            ha = line.get("h_a")
            if d and ha in ("h", "a"):
                out[(title, d, ha)] = line
    return out


def build_rows(league_key: str, season: int, matches: list[dict],
               teams: dict[str, list[dict]]) -> tuple[list[dict], set[str]]:
    """Processed rows for one league-season plus the Understat names that
    had no alias (a row with an unmapped side is refused, loudly, rather
    than written under a wrong club)."""
    aliases = UNDERSTAT_ALIASES[league_key]
    lines = _team_lines(teams)
    rows, unmatched = [], set()
    for m in matches:
        if not m.get("isResult"):
            continue
        h, a = m["h"]["title"], m["a"]["title"]
        ch, ca = aliases.get(h), aliases.get(a)
        if ch is None:
            unmatched.add(h)
        if ca is None:
            unmatched.add(a)
        if ch is None or ca is None:
            continue
        d = str(m["datetime"])[:10]
        xg = _maybe_json(m.get("xG", {}))
        goals = _maybe_json(m.get("goals", {}))
        hl = lines.get((h, d, "h"), {})
        al = lines.get((a, d, "a"), {})
        h_att, h_def = _ppda(hl.get("ppda"))
        a_att, a_def = _ppda(al.get("ppda"))
        rows.append({
            "league": league_key,
            "season": season_label(season),
            "date": d,
            "match_id": str(m.get("id", "")),
            "home_team": ch,
            "away_team": ca,
            "goals_home": _num(goals.get("h")),
            "goals_away": _num(goals.get("a")),
            "xg_home": _num(xg.get("h")),
            "xg_away": _num(xg.get("a")),
            "npxg_home": _num(hl.get("npxG")),
            "npxg_away": _num(al.get("npxG")),
            "xpts_home": _num(hl.get("xpts")),
            "xpts_away": _num(al.get("xpts")),
            "ppda_att_home": h_att, "ppda_def_home": h_def,
            "ppda_att_away": a_att, "ppda_def_away": a_def,
            "deep_home": _num(hl.get("deep")),
            "deep_away": _num(al.get("deep")),
        })
    return rows, unmatched


# --------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------
KEY = ("league", "date", "home_team", "away_team")


def _fmt(v):
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))
    return v


def merge_processed(new_rows: list[dict], out: Path = OUT_CSV) -> list[dict]:
    """Committed rows with fetched matches replacing same-keyed rows."""
    fresh = {tuple(r[k] for k in KEY): r for r in new_rows}
    kept = []
    if out.exists():
        with out.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if tuple(r[k] for k in KEY) not in fresh:
                    kept.append(r)
    combined = kept + [{k: _fmt(v) for k, v in r.items()} for r in new_rows]
    combined.sort(key=lambda r: (r["date"], r["league"], r["home_team"]))
    return combined


def write_processed(rows: list[dict], out: Path = OUT_CSV) -> int:
    combined = merge_processed(rows, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(combined)
    return len(combined)


def write_legacy_xg(rows: list[dict], out: Path = LEGACY_XG_CSV) -> int:
    """Keep `xg_matches.csv` (league, date, home, away, xg_home, xg_away)
    in step, so `model/xg.py` and everything fitted on it keep working."""
    legacy = [{"league": r["league"], "date": r["date"],
               "home_team": r["home_team"], "away_team": r["away_team"],
               "xg_home": _fmt(r["xg_home"]), "xg_away": _fmt(r["xg_away"])}
              for r in rows if r.get("xg_home") is not None]
    fresh = {tuple(r[k] for k in KEY) for r in legacy}
    kept = []
    if out.exists():
        with out.open(encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f) if tuple(r[k] for k in KEY) not in fresh]
    combined = kept + legacy
    combined.sort(key=lambda r: (r["date"], r["league"], r["home_team"]))
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["league", "date", "home_team", "away_team",
                                          "xg_home", "xg_away"])
        w.writeheader()
        w.writerows(combined)
    return len(combined)


def load_processed(path: Path = OUT_CSV):
    """The processed table as a DataFrame (numeric columns typed), or an
    empty frame with the right columns when nothing has been fetched."""
    import pandas as pd
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    for c in COLUMNS[6:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["match_id"] = df["match_id"].astype(str)
    return df


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------
def probe(league_key: str, season: int) -> None:
    """Print the live response's shape and exit — the check to run from
    Actions before trusting a parser written blind."""
    slug = LEAGUES[league_key]
    try:
        resp = requests.get(ENDPOINT.format(slug=slug, season=season), timeout=30,
                            headers=headers(slug, season))
    except requests.RequestException as exc:
        print(f"unreachable: {exc}")
        print("(understat.com is blocked by the dev sandbox proxy — run this from Actions)")
        return
    print(f"HTTP {resp.status_code}  content-type={resp.headers.get('content-type')}  "
          f"bytes={len(resp.content)}")
    payload = _payload_from_response(resp.text)
    if payload is None:
        print("unparseable; first 400 chars:\n" + resp.text[:400])
        return
    print("top-level keys:", {k: type(_maybe_json(v)).__name__ for k, v in payload.items()})
    matches, teams = parse_payload(payload)
    print(f"matches parsed: {len(matches)}; teams parsed: {len(teams)}")
    if matches:
        print("sample match:", json.dumps(matches[0])[:600])
    if teams:
        title, history = next(iter(teams.items()))
        print(f"sample team {title!r} first history line:",
              json.dumps(history[0])[:600] if history else "(empty)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    this_season = current_understat_season()
    parser.add_argument("--from-season", type=int, default=this_season - 1)
    parser.add_argument("--all", action="store_true",
                        help=f"backfill every season from {FIRST_SEASON}")
    parser.add_argument("--leagues", nargs="*", default=list(LEAGUES))
    parser.add_argument("--probe", nargs=2, metavar=("LEAGUE", "SEASON"))
    args = parser.parse_args()

    if args.probe:
        probe(args.probe[0], int(args.probe[1]))
        return

    first = FIRST_SEASON if args.all else args.from_season
    print(f"Fetching Understat league data {first}-{this_season}…")
    all_rows, ok = [], False
    with requests.Session() as session:
        for league_key in args.leagues:
            for season in range(first, this_season + 1):
                payload = fetch_league_season(league_key, season, session=session)
                if payload is None:
                    continue
                matches, teams = parse_payload(payload)
                rows, unmatched = build_rows(league_key, season, matches, teams)
                for name in sorted(unmatched):
                    print(f"  ! {league_key}: unmatched understat name {name!r} — "
                          f"add it to UNDERSTAT_ALIASES")
                with_adv = sum(1 for r in rows if r["npxg_home"] is not None)
                print(f"  + {league_key} {season}: {len(rows)} matches "
                      f"({with_adv} with npxG/PPDA/deep)")
                ok = ok or bool(rows)
                all_rows.extend(rows)
    if not ok:
        print("Nothing fetched; committed files untouched.")
        sys.exit(1)
    n = write_processed(all_rows)
    m = write_legacy_xg(all_rows)
    print(f"Wrote {n} rows to {OUT_CSV} and {m} rows to {LEGACY_XG_CSV}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
