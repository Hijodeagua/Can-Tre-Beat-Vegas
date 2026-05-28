"""Schedule loader + team-abbrev reconciliation.

The raw per-team game file uses pro-football-reference abbreviations
(GNB, KAN, NWE, ...) while nflverse uses (GB, KC, NE, ...). This module
loads the cached nflverse games.csv and exposes it keyed by the PFR
abbreviation set so it can be merged onto the existing feature frame.

Run `python schedule.py --refresh` to re-download from nflverse.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "data" / "schedules" / "nflverse_games.csv"
NFLVERSE_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"

# Map nflverse abbrev -> PFR abbrev (used in 2023-2025W3.csv)
NFLVERSE_TO_PFR = {
    "GB": "GNB", "KC": "KAN", "LA": "LAR", "LV": "LVR",
    "NE": "NWE", "NO": "NOR", "SF": "SFO", "TB": "TAM",
}


def refresh_cache() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {NFLVERSE_URL} -> {CACHE_PATH}")
    urllib.request.urlretrieve(NFLVERSE_URL, CACHE_PATH)


def _to_pfr(abbr: str) -> str:
    return NFLVERSE_TO_PFR.get(abbr, abbr)


def load_schedule(min_season: int = 2023) -> pd.DataFrame:
    if not CACHE_PATH.exists():
        refresh_cache()
    g = pd.read_csv(CACHE_PATH)
    g = g[g["season"] >= min_season].copy()
    g["gameday"] = pd.to_datetime(g["gameday"])
    g["home_team"] = g["home_team"].map(_to_pfr)
    g["away_team"] = g["away_team"].map(_to_pfr)
    keep = [
        "game_id", "season", "week", "gameday",
        "home_team", "away_team", "home_score", "away_score",
        "spread_line", "total_line", "home_rest", "away_rest",
        "roof", "surface", "temp", "wind",
        "home_qb_name", "away_qb_name",
        "home_coach", "away_coach", "div_game",
    ]
    return g[keep].reset_index(drop=True)


def to_team_perspective(schedule: pd.DataFrame) -> pd.DataFrame:
    """Explode each game into two team-perspective rows so it merges 1:1
    with the per-team-per-game stats file."""
    home = schedule.copy()
    home["Team"] = home["home_team"]
    home["Opp"] = home["away_team"]
    home["is_home"] = 1
    home["team_rest_sched"] = home["home_rest"]
    home["opp_rest_sched"] = home["away_rest"]
    home["team_qb"] = home["home_qb_name"]
    home["opp_qb"] = home["away_qb_name"]
    home["team_coach"] = home["home_coach"]
    home["opp_coach"] = home["away_coach"]
    home["team_score_sched"] = home["home_score"]
    home["opp_score_sched"] = home["away_score"]
    home["closing_spread_team"] = -home["spread_line"]

    away = schedule.copy()
    away["Team"] = away["away_team"]
    away["Opp"] = away["home_team"]
    away["is_home"] = 0
    away["team_rest_sched"] = away["away_rest"]
    away["opp_rest_sched"] = away["home_rest"]
    away["team_qb"] = away["away_qb_name"]
    away["opp_qb"] = away["home_qb_name"]
    away["team_coach"] = away["away_coach"]
    away["opp_coach"] = away["home_coach"]
    away["team_score_sched"] = away["away_score"]
    away["opp_score_sched"] = away["home_score"]
    away["closing_spread_team"] = away["spread_line"]

    cols = ["game_id", "season", "week", "gameday", "Team", "Opp", "is_home",
            "team_rest_sched", "opp_rest_sched", "team_qb", "opp_qb",
            "team_coach", "opp_coach", "team_score_sched", "opp_score_sched",
            "closing_spread_team", "total_line", "roof", "surface",
            "temp", "wind", "div_game"]
    return pd.concat([home[cols], away[cols]], ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Re-download from nflverse")
    args = parser.parse_args()
    if args.refresh:
        refresh_cache()
    s = load_schedule()
    print(f"Loaded {len(s)} games, seasons {sorted(s['season'].unique())}")
    print(s.head(3).to_string())
