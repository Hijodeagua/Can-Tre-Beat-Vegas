"""Daily results + schedule pull from the MLB Stats API.

Two jobs, both idempotent:

1. Refresh `data/mlb/games_2009_2026.csv` with final scores for every 2026
   regular-season game completed through `through_date`. The window re-fetches
   from the last recorded date forward, replacing that tail, so a re-run
   self-heals suspended games or late corrections.
2. Rewrite `data/mlb/schedule_2026_remaining.csv` with every not-yet-final
   game from `through_date` forward (that file feeds both the daily slate and
   the rest-of-season Monte Carlo).

Parsing conventions are shared with mlb/build_games.py (same NAME_TO_BREF
mapping, same final-game filter).
"""

from __future__ import annotations

import csv
import json
import urllib.request
from datetime import date, timedelta

from mlb.build_games import NAME_TO_BREF, franchise
from mlb.daily.config import CURRENT_SEASON, GAMES_CSV, SCHEDULE_CSV

SCHEDULE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R"
    "&startDate={start}&endDate={end}&hydrate=probablePitcher"
)
SEASON_END = f"{CURRENT_SEASON}-11-30"

GAME_KEY = ("date", "away", "home", "game_num")


def fetch_schedule(start: str, end: str) -> dict:
    url = SCHEDULE_URL.format(start=start, end=end)
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def parse_games(data: dict) -> tuple[list[dict], list[dict]]:
    """Split a statsapi schedule payload into (finals, upcoming) rows."""
    finals, upcoming = [], []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            teams = g["teams"]
            away_name = teams["away"]["team"]["name"]
            home_name = teams["home"]["team"]["name"]
            if away_name not in NAME_TO_BREF or home_name not in NAME_TO_BREF:
                continue  # exhibition vs non-MLB opponent
            away, home = NAME_TO_BREF[away_name], NAME_TO_BREF[home_name]
            base = {
                "date": d["date"],
                "season": CURRENT_SEASON,
                "game_num": int(g.get("gameNumber", 1)),
                "away": away, "home": home,
                "away_fr": franchise(away), "home_fr": franchise(home),
            }
            state = g.get("status", {}).get("codedGameState")
            if state == "F":
                if "score" not in teams["away"] or "score" not in teams["home"]:
                    continue
                finals.append({
                    **base,
                    "away_score": int(teams["away"]["score"]),
                    "home_score": int(teams["home"]["score"]),
                })
            elif state not in ("C", "D"):  # skip cancelled; postponed games
                # reappear on their makeup date in a later pull
                upcoming.append({
                    **base,
                    # Probable starters (hydrate=probablePitcher). Display
                    # only - the Elo model does not use starter identity.
                    # Empty string when MLB hasn't announced one yet.
                    "away_sp": _probable(teams["away"]),
                    "home_sp": _probable(teams["home"]),
                    "away_sp_id": _probable_id(teams["away"]),
                    "home_sp_id": _probable_id(teams["home"]),
                })
    return finals, upcoming


def _probable(side: dict) -> str:
    return (side.get("probablePitcher") or {}).get("fullName", "")


def _probable_id(side: dict) -> str:
    pid = (side.get("probablePitcher") or {}).get("id")
    return str(pid) if pid is not None else ""


def read_games() -> list[dict]:
    with open(GAMES_CSV, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def last_recorded_date(rows: list[dict]) -> str:
    return max(r["date"] for r in rows if int(r["season"]) == CURRENT_SEASON)


def update(through_date: str) -> dict:
    """Pull finals through `through_date` (inclusive) and the remaining
    schedule from `through_date` forward. Returns a small summary dict."""
    rows = read_games()
    fetch_start = last_recorded_date(rows)

    finals, _ = parse_games(fetch_schedule(fetch_start, through_date))
    # Replace the refetched tail rather than appending, so re-runs are clean.
    kept = [
        r for r in rows
        if not (int(r["season"]) == CURRENT_SEASON and r["date"] >= fetch_start)
    ]
    merged = kept + [{k: str(v) for k, v in f.items()} for f in finals]
    merged.sort(key=lambda r: (r["date"], int(r["game_num"]), r["home"]))
    with open(GAMES_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["date", "season", "game_num", "away", "home",
                        "away_fr", "home_fr", "away_score", "home_score"],
        )
        w.writeheader()
        w.writerows(merged)

    # Remaining schedule: everything not yet final from the day after
    # `through_date` onward, plus any unfinished games on through_date itself.
    day_after = (date.fromisoformat(through_date) + timedelta(days=1)).isoformat()
    _, upcoming = parse_games(fetch_schedule(through_date, SEASON_END))
    write_schedule(upcoming)

    return {
        "finals_window_start": fetch_start,
        "finals_added": len(finals),
        "total_games": len(merged),
        "remaining_scheduled": len(upcoming),
        "remaining_from": day_after,
    }


def write_schedule(upcoming: list[dict]) -> None:
    SCHEDULE_CSV.parent.mkdir(parents=True, exist_ok=True)
    upcoming = sorted(upcoming, key=lambda r: (r["date"], int(r["game_num"]), r["home"]))
    with open(SCHEDULE_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["date", "season", "game_num", "away", "home",
                        "away_fr", "home_fr",
                        "away_sp", "home_sp", "away_sp_id", "home_sp_id"],
        )
        w.writeheader()
        w.writerows(upcoming)


def read_schedule() -> list[dict]:
    if not SCHEDULE_CSV.exists():
        return []
    with open(SCHEDULE_CSV, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
