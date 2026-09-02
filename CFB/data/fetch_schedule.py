"""Refresh the committed college-football game spine from cfbfastR-data.

The college analogue of the nflverse schedule the NFL model reads and of
openfootball for the club-soccer model: sportsdataverse's cfbfastR-data
repo commits one ESPN-derived schedule CSV per season, served over
raw.githubusercontent.com with no API key. It goes back to 2001 and carries
what the Elo needs and Sports Reference exports don't: a per-season
conference for both teams (realignment-proof), an FBS/FCS division tag for
both teams (so FCS opponents can be pooled), the neutral-site flag, and
final scores updated through the current season.

What lands in `data/college_football/games.csv` (one row per game, home
perspective, FBS-involved games only):

    game_id, season, week, season_type, date, start_utc, home_team,
    away_team, home_conference, away_conference, home_division,
    away_division, home_points, away_points, neutral_site, conference_game,
    completed, notes

`date` is the US/Eastern calendar date of kickoff — the daily slate and
grading are keyed on it, same as the MLB pipeline — while `start_utc` keeps
the exact ordering for the Elo replay.

Known upstream gaps, written down so nobody re-discovers them: bowl games
and the CFP are only present from the 2024 season (older files are regular
season only), and the neutral-site flag is blank for 2001-02. Neither
matters for a walk-forward regular-season model.

Two refresh modes, both idempotent — the current season's rows are always
replaced wholesale (same tail-replace posture as mlb/daily/update_games.py):

    python -m CFB.data.fetch_schedule            # current season only
    python -m CFB.data.fetch_schedule --all      # rebuild 2001-present

An optional second pass reads ESPN's public scoreboard for the trailing
week and fills in finals for games cfbfastR-data has not yet re-processed
(it is ESPN-derived, so game ids match one-to-one). Best-effort: any
failure leaves the cfbfastR rows as they were.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from CFB.data.teams import canonical

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "college_football"
GAMES_CSV = DATA_DIR / "games.csv"

RAW_BASE = (
    "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/"
    "schedules/csv/cfb_schedules_{season}.csv"
)
ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/"
    "scoreboard?dates={yyyymmdd}&groups=80&limit=400"
)

FIRST_SEASON = 2001
ET = ZoneInfo("America/New_York")

COLUMNS = [
    "game_id", "season", "week", "season_type", "date", "start_utc",
    "home_team", "away_team", "home_conference", "away_conference",
    "home_division", "away_division", "home_points", "away_points",
    "neutral_site", "conference_game", "completed", "notes",
]

# ESPN scoreboard days to re-check for finals cfbfastR-data hasn't
# re-processed yet. A week covers a Thursday-to-Saturday slate plus the
# lag of a missed nightly run upstream.
ESPN_LOOKBACK_DAYS = 7


def current_season(today: date | None = None) -> int:
    """The CFB season a date belongs to: bowls in January close out the
    prior calendar year's season."""
    today = today or datetime.now(ET).date()
    return today.year if today.month >= 6 else today.year - 1


def _et_date(start_utc: str) -> str:
    ts = pd.Timestamp(start_utc)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(ET).date().isoformat()


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """One cfbfastR season export -> spine rows (FBS-involved games only)."""
    df = raw.copy()
    for col in ("home_division", "away_division", "home_conference",
                "away_conference", "notes"):
        if col not in df.columns:
            df[col] = pd.NA
    fbs = (df["home_division"].eq("fbs")) | (df["away_division"].eq("fbs"))
    df = df[fbs].copy()

    out = pd.DataFrame({
        "game_id": df["game_id"].astype(int),
        "season": df["season"].astype(int),
        "week": df["week"].astype(int),
        "season_type": df["season_type"].astype(str),
        "date": df["start_date"].map(_et_date),
        "start_utc": pd.to_datetime(df["start_date"], utc=True)
                       .dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "home_team": df["home_team"].astype(str).str.strip().map(canonical),
        "away_team": df["away_team"].astype(str).str.strip().map(canonical),
        "home_conference": df["home_conference"],
        "away_conference": df["away_conference"],
        "home_division": df["home_division"],
        "away_division": df["away_division"],
        "home_points": pd.to_numeric(df["home_points"], errors="coerce"),
        "away_points": pd.to_numeric(df["away_points"], errors="coerce"),
        "neutral_site": df["neutral_site"].map(_as_bool).fillna(False).astype(bool),
        "conference_game": df["conference_game"].map(_as_bool).fillna(False).astype(bool),
        "completed": df["completed"].map(_as_bool).fillna(False).astype(bool),
        "notes": df["notes"],
    })
    # A "completed" row with no score is a cancellation or a data hole;
    # treat it as unplayed rather than as a 0-0 tie.
    scored = out["home_points"].notna() & out["away_points"].notna()
    out["completed"] = out["completed"] & scored
    out.loc[~out["completed"], ["home_points", "away_points"]] = pd.NA
    return out[COLUMNS].sort_values(["start_utc", "game_id"]).reset_index(drop=True)


def _as_bool(v) -> bool | None:
    if pd.isna(v):
        return None
    if isinstance(v, (bool,)):
        return bool(v)
    return str(v).strip().lower() in ("true", "1", "t", "yes")


def fetch_season(season: int, timeout: int = 60) -> pd.DataFrame | None:
    """Download and normalize one season; None when upstream has no file
    (a season not yet published) or the request fails."""
    url = RAW_BASE.format(season=season)
    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        print(f"  {season}: fetch failed ({exc})")
        return None
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    from io import StringIO
    raw = pd.read_csv(StringIO(resp.text), low_memory=False)
    return normalize(raw)


def load_games() -> pd.DataFrame:
    if not GAMES_CSV.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(GAMES_CSV, low_memory=False)


def write_games(games: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    games = games[COLUMNS].sort_values(["start_utc", "game_id"]).reset_index(drop=True)
    games.to_csv(GAMES_CSV, index=False)


def merge_season(existing: pd.DataFrame, season: int, fresh: pd.DataFrame) -> pd.DataFrame:
    """Replace one season's rows wholesale. Rerunning is a no-op."""
    kept = existing[existing["season"] != season] if len(existing) else existing
    return pd.concat([kept, fresh], ignore_index=True)


# --- ESPN scoreboard fill-in -----------------------------------------------

def parse_scoreboard(payload: dict) -> dict[int, tuple[float, float, bool]]:
    """ESPN scoreboard JSON -> {game_id: (home_points, away_points, neutral)}
    for completed events only. Game ids are ESPN event ids, which is also
    what cfbfastR-data uses, so no name matching is needed."""
    finals: dict[int, tuple[float, float, bool]] = {}
    for ev in payload.get("events", []) or []:
        try:
            comp = ev["competitions"][0]
        except (KeyError, IndexError):
            continue
        status = ((comp.get("status") or ev.get("status") or {}).get("type") or {})
        if not status.get("completed"):
            continue
        home = away = None
        for c in comp.get("competitors", []):
            if c.get("homeAway") == "home":
                home = c
            elif c.get("homeAway") == "away":
                away = c
        if home is None or away is None:
            continue
        try:
            gid = int(ev["id"])
            hp, ap = float(home["score"]), float(away["score"])
        except (KeyError, TypeError, ValueError):
            continue
        finals[gid] = (hp, ap, bool(comp.get("neutralSite", False)))
    return finals


def espn_fill(games: pd.DataFrame, today: date, days: int = ESPN_LOOKBACK_DAYS,
              timeout: int = 30) -> int:
    """Fill finals for known game ids from ESPN's scoreboard over the
    trailing window. Returns the number of rows updated. Mutates `games`."""
    updated = 0
    ids = set(games.loc[~games["completed"].astype(bool), "game_id"].astype(int))
    if not ids:
        return 0
    for back in range(days + 1):
        day = today - timedelta(days=back)
        url = ESPN_SCOREBOARD.format(yyyymmdd=day.strftime("%Y%m%d"))
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            finals = parse_scoreboard(resp.json())
        except (requests.RequestException, ValueError) as exc:
            print(f"  espn {day}: skipped ({exc})")
            continue
        for gid, (hp, ap, _neutral) in finals.items():
            if gid in ids:
                mask = games["game_id"].astype(int) == gid
                games.loc[mask, "home_points"] = hp
                games.loc[mask, "away_points"] = ap
                games.loc[mask, "completed"] = True
                ids.discard(gid)
                updated += 1
    return updated


# --- CLI ----------------------------------------------------------------------

def refresh(all_seasons: bool = False, use_espn: bool = True,
            today: date | None = None) -> dict:
    today = today or datetime.now(ET).date()
    season = current_season(today)
    games = load_games()

    seasons = list(range(FIRST_SEASON, season + 1)) if all_seasons else [season]
    fetched = 0
    for s in seasons:
        fresh = fetch_season(s)
        if fresh is None:
            print(f"  {s}: not available upstream")
            continue
        games = merge_season(games, s, fresh)
        fetched += 1
        print(f"  {s}: {len(fresh)} games ({int(fresh['completed'].sum())} final)")

    espn_updates = 0
    if use_espn:
        espn_updates = espn_fill(games, today)
        if espn_updates:
            print(f"  espn: filled {espn_updates} finals")

    write_games(games)
    cur = games[games["season"] == season]
    return {
        "season": season,
        "seasons_fetched": fetched,
        "espn_updates": espn_updates,
        "total_games": int(len(games)),
        "season_games": int(len(cur)),
        "season_final": int(cur["completed"].astype(bool).sum()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--all", action="store_true",
                    help="rebuild every season from 2001, not just the current one")
    ap.add_argument("--no-espn", action="store_true",
                    help="skip the ESPN scoreboard fill-in pass")
    args = ap.parse_args(argv)
    summary = refresh(all_seasons=args.all, use_espn=not args.no_espn)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
