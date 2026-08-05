"""
MLB Stats API job: in-progress season results + probable pitchers.

statsapi.mlb.com is blocked from the dev session's egress policy but open
from GitHub Actions, so this module runs on the mlb-data workflow schedule
(same pattern as data_jobs/odds_api). It covers the gap Retrosheet leaves:
the current season, which Retrosheet publishes only after it ends. Once the
Retrosheet release lands, the refresh job supersedes these rows.

Outputs:
    MLB/data/statsapi/games_{year}.csv        finals, one row per gamePk,
                                              upserted (re-runs are idempotent)
    MLB/data/statsapi/probables_latest.csv    upcoming games w/ probables
    MLB/data/statsapi/probables/probables_{stamp}.csv
                                              timestamped snapshot, so the
                                              announcement history is kept —
                                              each row carries pulled_at_utc,
                                              which is what makes probables
                                              leakage-legal to use

Usage:
    python -m MLB.data_jobs.statsapi --finals-days 3 --probables-days 7
"""

import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from .config import MLB_ID_TO_FRANCHISE, STATSAPI_BASE, STATSAPI_DIR

SCHEDULE_URL = STATSAPI_BASE + "/schedule"
# The live feed is v1.1; it carries weather, officials, and final linescores.
FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

TIMEOUT = 30


def _get(session: requests.Session, url: str, **params) -> dict:
    resp = session.get(url, params=params or None, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _franchise(team: dict) -> str:
    return MLB_ID_TO_FRANCHISE.get(team.get("id"), f"MLB{team.get('id')}")


def fetch_finals(session: requests.Session, start_date: str, end_date: str) -> pd.DataFrame:
    """Final games in [start_date, end_date], enriched from the live feed."""
    sched = _get(
        session, SCHEDULE_URL,
        sportId=1, startDate=start_date, endDate=end_date,
        gameType="R,F,D,L,W",  # regular season + all postseason rounds
    )
    rows = []
    for day in sched.get("dates", []):
        for game in day.get("games", []):
            if game.get("status", {}).get("codedGameState") != "F":
                continue
            rows.append(_final_row(session, game))
    return pd.DataFrame(rows)


def _final_row(session: requests.Session, game: dict) -> dict:
    game_pk = game["gamePk"]
    home = game["teams"]["home"]
    away = game["teams"]["away"]

    row = {
        "game_pk": game_pk,
        "date": game.get("officialDate"),
        "game_type": game.get("gameType"),
        "day_night": game.get("dayNight"),
        "home_team": _franchise(home["team"]),
        "away_team": _franchise(away["team"]),
        "home_mlb_id": home["team"].get("id"),
        "away_mlb_id": away["team"].get("id"),
        "home_score": home.get("score"),
        "away_score": away.get("score"),
        "venue": game.get("venue", {}).get("name"),
        "venue_id": game.get("venue", {}).get("id"),
        "doubleheader": game.get("doubleHeader"),
        "game_num": game.get("gameNumber"),
    }

    # One feed call per final game: weather, wind, officials, linescore,
    # starting pitchers, attendance, duration.
    try:
        feed = _get(session, FEED_URL.format(game_pk=game_pk))
    except requests.RequestException:
        return row

    game_data = feed.get("gameData", {})
    live = feed.get("liveData", {})

    weather = game_data.get("weather", {})
    row["weather_condition"] = weather.get("condition")
    row["temp_f"] = weather.get("temp")
    row["wind"] = weather.get("wind")

    info = game_data.get("gameInfo", {})
    row["attendance"] = info.get("attendance")
    row["duration_minutes"] = info.get("gameDurationMinutes")

    linescore = live.get("linescore", {})
    row["innings"] = len(linescore.get("innings", []))

    probables = game_data.get("probablePitchers", {})
    boxscore = live.get("boxscore", {})
    for side in ("home", "away"):
        sp = _starting_pitcher(boxscore, side) or probables.get(side, {})
        row[f"{side}_sp_mlb_id"] = sp.get("id")
        row[f"{side}_sp_name"] = sp.get("fullName")

    for official in boxscore.get("officials", []):
        if official.get("officialType") == "Home Plate":
            row["ump_hp_mlb_id"] = official.get("official", {}).get("id")
            row["ump_hp_name"] = official.get("official", {}).get("fullName")
    return row


def _starting_pitcher(boxscore: dict, side: str) -> dict:
    """First pitcher listed in the boxscore pitching order (the starter)."""
    team_box = boxscore.get("teams", {}).get(side, {})
    pitcher_ids = team_box.get("pitchers", [])
    if not pitcher_ids:
        return {}
    player = team_box.get("players", {}).get(f"ID{pitcher_ids[0]}", {})
    person = player.get("person", {})
    return {"id": person.get("id"), "fullName": person.get("fullName")}


def fetch_probables(session: requests.Session, days_ahead: int) -> pd.DataFrame:
    """Upcoming schedule with probable pitchers as currently announced."""
    today = datetime.now(timezone.utc).date()
    sched = _get(
        session, SCHEDULE_URL,
        sportId=1,
        startDate=str(today),
        endDate=str(today + timedelta(days=days_ahead)),
        hydrate="probablePitcher",
    )
    pulled_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for day in sched.get("dates", []):
        for game in day.get("games", []):
            home = game["teams"]["home"]
            away = game["teams"]["away"]
            rows.append({
                "pulled_at_utc": pulled_at,
                "game_pk": game["gamePk"],
                "date": game.get("officialDate"),
                "game_datetime_utc": game.get("gameDate"),
                "home_team": _franchise(home["team"]),
                "away_team": _franchise(away["team"]),
                "home_probable_mlb_id": home.get("probablePitcher", {}).get("id"),
                "home_probable_name": home.get("probablePitcher", {}).get("fullName"),
                "away_probable_mlb_id": away.get("probablePitcher", {}).get("id"),
                "away_probable_name": away.get("probablePitcher", {}).get("fullName"),
                "venue": game.get("venue", {}).get("name"),
                "day_night": game.get("dayNight"),
            })
    return pd.DataFrame(rows)


def upsert_finals(finals: pd.DataFrame) -> None:
    """Merge finals into per-year files keyed by game_pk."""
    if finals.empty:
        print("no finals in window")
        return
    os.makedirs(STATSAPI_DIR, exist_ok=True)
    finals["year"] = finals["date"].str[:4]
    for year, chunk in finals.groupby("year"):
        path = os.path.join(STATSAPI_DIR, f"games_{year}.csv")
        if os.path.exists(path):
            existing = pd.read_csv(path)
            merged = pd.concat([existing, chunk.drop(columns="year")], ignore_index=True)
            merged = merged.drop_duplicates(subset="game_pk", keep="last")
        else:
            merged = chunk.drop(columns="year")
        merged = merged.sort_values("game_pk")
        merged.to_csv(path, index=False)
        print(f"{path}: {len(merged)} games")


def save_probables(probables: pd.DataFrame) -> None:
    os.makedirs(os.path.join(STATSAPI_DIR, "probables"), exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")
    snap_path = os.path.join(STATSAPI_DIR, "probables", f"probables_{stamp}.csv")
    latest_path = os.path.join(STATSAPI_DIR, "probables_latest.csv")
    probables.to_csv(snap_path, index=False)
    probables.to_csv(latest_path, index=False)
    print(f"{snap_path}: {len(probables)} scheduled games")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finals-days", type=int, default=3,
                        help="days of history to (re)pull finals for")
    parser.add_argument("--probables-days", type=int, default=7,
                        help="days ahead to snapshot probables for")
    args = parser.parse_args()

    session = requests.Session()
    today = datetime.now(timezone.utc).date()
    start = str(today - timedelta(days=args.finals_days))
    finals = fetch_finals(session, start, str(today))
    upsert_finals(finals)
    probables = fetch_probables(session, args.probables_days)
    save_probables(probables)


if __name__ == "__main__":
    main()
