"""
Understat player seasons -> the site's player-comparison data.

The league endpoint `understat.py` already fetches carries a third block
next to `dates` and `teams`: `players`, one line per player-season with
minutes, goals, assists, shots, key passes, xG, xA, npxG, xGChain and
xGBuildup. This module turns that block into one compact JSON file per
season under `web/public/data/soccer/players/`, plus an `index.json`
listing the seasons, which is everything `/soccer/compare` reads.

Fetching is `understat.fetch_league_season` and nothing else - same
endpoint, same headers, same raw cache - so a league-season already
cached under `data/soccer_clubs/raw/understat/` is reused with
`--from-cache` and costs no request. Parsing is shape-tolerant in the
same way as the match parser (`players` / `playersData`, values as
objects or JSON-encoded strings, a list or an id-keyed dict).

One row per player per league-season. A player who moved inside a league
mid-season is one Understat line with a comma-joined `team_title`; that
is kept as-is ("Chelsea, Arsenal"). A player who moved *between* two of
the five leagues has one line in each and stays two rows - they are
different peer groups.

    python -m soccer.clubs.data.understat_players            # this + last season
    python -m soccer.clubs.data.understat_players --all      # 2014-15 onward
    python -m soccer.clubs.data.understat_players --from-cache
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from soccer.clubs.data import understat as u

OUT_DIR = u.REPO_ROOT / "web" / "public" / "data" / "soccer" / "players"
INDEX = "index.json"

LEAGUE_NAMES = {
    "epl": "Premier League",
    "la_liga": "La Liga",
    "bundesliga": "Bundesliga",
    "serie_a": "Serie A",
    "ligue_1": "Ligue 1",
}

# Row layout of every season file. The page reads rows by these names
# (the file carries the list), so appending a column is safe; reordering
# is too, as long as the names stay.
COLUMNS = [
    "id", "name", "team", "league", "pos", "group",
    "games", "min", "goals", "assists", "shots", "key_passes",
    "xg", "xa", "npg", "npxg", "xgchain", "xgbuildup", "yellow", "red",
]

# our column -> Understat field
_INT_FIELDS = {
    "games": "games", "min": "time", "goals": "goals", "assists": "assists",
    "shots": "shots", "key_passes": "key_passes", "npg": "npg",
    "yellow": "yellow_cards", "red": "red_cards",
}
_FLOAT_FIELDS = {
    "xg": "xG", "xa": "xA", "npxg": "npxG",
    "xgchain": "xGChain", "xgbuildup": "xGBuildup",
}

_PLAYERS_KEYS = ("players", "playersData")
_GROUPS = {"GK": "GK", "D": "DF", "M": "MF", "F": "FW"}


def parse_players(payload: dict) -> list[dict]:
    """The raw Understat player lines out of a league payload, or []."""
    for key in _PLAYERS_KEYS:
        if key in payload:
            value = u._maybe_json(payload[key])
            if isinstance(value, dict):
                value = list(value.values())
            if not isinstance(value, list):
                return []
            return [p for p in (u._maybe_json(v) for v in value) if isinstance(p, dict)]
    return []


def position_group(position: str | None) -> str | None:
    """Understat lists every role a player filled in the season, most
    used first, e.g. "F M S" (S = substitute). The first one that is a
    real position is the peer group: GK / DF / MF / FW. A player who only
    ever came off the bench ("S") has no group and is left out of every
    percentile pool."""
    for token in str(position or "").split():
        if token in _GROUPS:
            return _GROUPS[token]
    return None


def build_rows(league_key: str, players: list[dict]) -> list[dict]:
    """Typed rows for one league-season. Lines with no id, no name or no
    minutes are dropped: a player who never played has nothing to rank."""
    rows = []
    for p in players:
        pid, name = str(p.get("id") or "").strip(), str(p.get("player_name") or "").strip()
        minutes = u._num(p.get("time"), 0) or 0
        if not pid or not name or minutes <= 0:
            continue
        row = {
            "id": pid,
            "name": name,
            "team": ", ".join(t.strip() for t in str(p.get("team_title") or "").split(",")
                              if t.strip()),
            "league": league_key,
            "pos": str(p.get("position") or "").strip(),
            "group": position_group(p.get("position")),
        }
        for col, field in _INT_FIELDS.items():
            v = u._num(p.get(field))
            row[col] = None if v is None else int(round(v))
        for col, field in _FLOAT_FIELDS.items():
            v = u._num(p.get(field))
            row[col] = None if v is None else round(v, 3)
        rows.append(row)
    rows.sort(key=lambda r: (-r["min"], r["name"]))
    return rows


def season_file(season: int) -> str:
    return f"{season}.json"


def write_season(season: int, rows_by_league: dict[str, list[dict]],
                 out_dir: Path = OUT_DIR) -> int:
    """Write one season file, replacing only the leagues fetched this run:
    a league whose request failed keeps the rows already committed for it
    rather than vanishing from the page."""
    path = out_dir / season_file(season)
    kept: list[dict] = []
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        cols = old.get("columns", [])
        for values in old.get("rows", []):
            row = dict(zip(cols, values))
            if row.get("league") not in rows_by_league:
                kept.append(row)
    rows = kept + [r for league_rows in rows_by_league.values() for r in league_rows]
    rows.sort(key=lambda r: (-(r.get("min") or 0), r.get("name") or ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "season": u.season_label(season),
        "columns": COLUMNS,
        "rows": [[r.get(c) for c in COLUMNS] for r in rows],
    }
    path.write_text(json.dumps(body, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8")
    return len(rows)


def write_index(out_dir: Path = OUT_DIR, now: datetime | None = None) -> dict:
    """Rebuild index.json from the season files on disk, newest first."""
    seasons = []
    for path in sorted(out_dir.glob("[0-9][0-9][0-9][0-9].json"), reverse=True):
        body = json.loads(path.read_text(encoding="utf-8"))
        cols = body.get("columns", [])
        li = cols.index("league") if "league" in cols else None
        leagues = sorted({r[li] for r in body.get("rows", [])} if li is not None else [],
                         key=lambda k: list(LEAGUE_NAMES).index(k) if k in LEAGUE_NAMES else 99)
        seasons.append({
            "key": int(path.stem),
            "label": body.get("season") or u.season_label(int(path.stem)),
            "file": path.name,
            "players": len(body.get("rows", [])),
            "leagues": leagues,
        })
    index = {
        "generated_at": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "Understat",
        "leagues": LEAGUE_NAMES,
        "seasons": seasons,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / INDEX).write_text(json.dumps(index, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    return index


def _cached(league_key: str, season: int) -> dict | None:
    path = u.raw_path(league_key, season)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    this_season = u.current_understat_season()
    parser.add_argument("--from-season", type=int, default=this_season - 1)
    parser.add_argument("--all", action="store_true",
                        help=f"backfill every season from {u.FIRST_SEASON}")
    parser.add_argument("--leagues", nargs="*", default=list(u.LEAGUES))
    parser.add_argument("--from-cache", action="store_true",
                        help="use raw payloads already cached by understat.py; "
                             "only fetch what is missing")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    first = u.FIRST_SEASON if args.all else args.from_season
    print(f"Understat player seasons {first}-{this_season}…")
    wrote = 0
    with requests.Session() as session:
        for season in range(first, this_season + 1):
            by_league: dict[str, list[dict]] = {}
            for league_key in args.leagues:
                payload = _cached(league_key, season) if args.from_cache else None
                if payload is None:
                    payload = u.fetch_league_season(league_key, season, session=session)
                if payload is None:
                    continue
                rows = build_rows(league_key, parse_players(payload))
                if not rows:
                    print(f"  ! {league_key} {season}: payload carried no player lines")
                    continue
                by_league[league_key] = rows
                print(f"  + {league_key} {season}: {len(rows)} players")
            if by_league:
                total = write_season(season, by_league, args.out_dir)
                print(f"  = {u.season_label(season)}: {total} player rows")
                wrote += 1
    index = write_index(args.out_dir)
    print(f"index: {len(index['seasons'])} season file(s) in {args.out_dir}")
    if not wrote:
        print("Nothing fetched; committed player files untouched.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
