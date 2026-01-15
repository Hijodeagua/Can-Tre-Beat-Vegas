"""
Data processors for odds API responses
"""

import os
from datetime import datetime
from typing import Optional

import pandas as pd
import pytz
from geopy.distance import geodesic

from .config import get_team_info, SUPPORTED_SPORTS


def process_game_data(games: list, sport: str) -> pd.DataFrame:
    """
    Process raw game data from the API into a structured DataFrame.

    Args:
        games: List of game data from API
        sport: Sport identifier ('nfl' or 'nba')

    Returns:
        Processed DataFrame with odds data
    """
    team_info = get_team_info(sport)
    sport_config = SUPPORTED_SPORTS[sport.lower()]

    rows = []
    eastern = pytz.timezone("US/Eastern")
    timestamp_pulled = datetime.now(pytz.UTC).astimezone(eastern).strftime("%Y-%m-%d %H:%M:%S")

    for game in games:
        row = _process_single_game(game, team_info, sport_config, eastern, timestamp_pulled)
        if row:
            rows.append(row)

    return pd.DataFrame(rows)


def _process_single_game(
    game: dict,
    team_info: dict,
    sport_config: dict,
    eastern: pytz.timezone,
    timestamp_pulled: str,
) -> Optional[dict]:
    """Process a single game's data"""

    # Parse game time
    utc_time = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
    eastern_time = utc_time.astimezone(eastern)
    prime_time = eastern_time.hour >= 19  # 7pm ET or later

    home_team = game["home_team"]
    away_team = game["away_team"]

    # Skip unknown teams
    if home_team not in team_info or away_team not in team_info:
        return None

    # Divisional matchup check
    div_match = (
        team_info[home_team]["conf"] == team_info[away_team]["conf"]
        and team_info[home_team]["div"] == team_info[away_team]["div"]
    )

    # Travel distance in miles
    travel_distance = geodesic(
        team_info[home_team]["loc"], team_info[away_team]["loc"]
    ).miles

    # Collect odds across all sportsbooks
    odds_data = _extract_odds(game, home_team, away_team)

    # Build base row
    row = {
        "League": sport_config["name"],
        "Timestamp Pulled": timestamp_pulled,
        "Date of Game (ET)": eastern_time.strftime("%Y-%m-%d %H:%M"),
        "Game ID": game.get("id", ""),
        "Home Team": home_team,
        "Away Team": away_team,
        "Divisional Matchup": "Yes" if div_match else "No",
        "Travel Distance (mi)": round(travel_distance, 1),
        "Prime Time (ET >= 7p)": "Yes" if prime_time else "No",
        "Home Conf": team_info[home_team]["conf"],
        "Home Div": team_info[home_team]["div"],
        "Away Conf": team_info[away_team]["conf"],
        "Away Div": team_info[away_team]["div"],
        **odds_data["averages"],
    }

    # Add per-bookmaker odds
    row.update(odds_data["per_book"])

    return row


def _extract_odds(game: dict, home_team: str, away_team: str) -> dict:
    """Extract and organize odds data from a game"""

    # Lists for calculating averages
    home_spread_odds = []
    away_spread_odds = []
    home_spread_points = []
    away_spread_points = []
    home_h2h_odds = []
    away_h2h_odds = []
    over_odds = []
    under_odds = []
    total_points = []

    per_book = {}

    for bookmaker in game.get("bookmakers", []):
        book_name = bookmaker.get("title", "Unknown")

        book_data = {
            "home_h2h": None, "away_h2h": None,
            "home_spread": None, "away_spread": None,
            "home_spread_point": None, "away_spread_point": None,
            "over": None, "under": None, "total": None,
        }

        for market in bookmaker.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", []) or []

            if key == "h2h":
                for outcome in outcomes:
                    if outcome.get("name") == home_team:
                        book_data["home_h2h"] = outcome.get("price")
                        home_h2h_odds.append(outcome.get("price"))
                    elif outcome.get("name") == away_team:
                        book_data["away_h2h"] = outcome.get("price")
                        away_h2h_odds.append(outcome.get("price"))

            elif key == "spreads":
                for outcome in outcomes:
                    if outcome.get("name") == home_team:
                        book_data["home_spread"] = outcome.get("price")
                        book_data["home_spread_point"] = outcome.get("point")
                        home_spread_odds.append(outcome.get("price"))
                        if outcome.get("point") is not None:
                            home_spread_points.append(outcome.get("point"))
                    elif outcome.get("name") == away_team:
                        book_data["away_spread"] = outcome.get("price")
                        book_data["away_spread_point"] = outcome.get("point")
                        away_spread_odds.append(outcome.get("price"))
                        if outcome.get("point") is not None:
                            away_spread_points.append(outcome.get("point"))

            elif key == "totals":
                for outcome in outcomes:
                    name = str(outcome.get("name", "")).lower()
                    if name == "over":
                        book_data["over"] = outcome.get("price")
                        book_data["total"] = outcome.get("point")
                        over_odds.append(outcome.get("price"))
                        if outcome.get("point") is not None:
                            total_points.append(outcome.get("point"))
                    elif name == "under":
                        book_data["under"] = outcome.get("price")
                        under_odds.append(outcome.get("price"))

        # Add per-book data to row
        per_book[f"Home {book_name} Spread Odds"] = book_data["home_spread"]
        per_book[f"Away {book_name} Spread Odds"] = book_data["away_spread"]
        per_book[f"Home {book_name} H2H Odds"] = book_data["home_h2h"]
        per_book[f"Away {book_name} H2H Odds"] = book_data["away_h2h"]
        per_book[f"Home {book_name} O/U Odds"] = book_data["over"]
        per_book[f"Away {book_name} O/U Odds"] = book_data["under"]

    # Calculate averages
    averages = {
        "Avg Home Spread Odds": _safe_avg(home_spread_odds),
        "Avg Away Spread Odds": _safe_avg(away_spread_odds),
        "Avg Home Spread Points": _safe_avg(home_spread_points),
        "Avg Away Spread Points": _safe_avg(away_spread_points),
        "Avg Home H2H Odds": _safe_avg(home_h2h_odds),
        "Avg Away H2H Odds": _safe_avg(away_h2h_odds),
        "Avg Over Odds": _safe_avg(over_odds),
        "Avg Under Odds": _safe_avg(under_odds),
        "Avg Total Points": _safe_avg(total_points),
    }

    return {"averages": averages, "per_book": per_book}


def _safe_avg(values: list) -> Optional[float]:
    """Calculate average, filtering None values"""
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return round(sum(filtered) / len(filtered), 2)


def save_odds_data(df: pd.DataFrame, sport: str, base_dir: str = ".") -> tuple[str, str]:
    """
    Save odds DataFrame to CSV files.

    Args:
        df: DataFrame with odds data
        sport: Sport identifier
        base_dir: Base directory for data files

    Returns:
        Tuple of (timestamped_path, latest_path)
    """
    sport_config = SUPPORTED_SPORTS[sport.lower()]
    save_dir = os.path.join(base_dir, sport_config["data_dir"])
    os.makedirs(save_dir, exist_ok=True)

    stamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M")
    prefix = sport_config["file_prefix"]

    timestamped_path = os.path.join(save_dir, f"{prefix}_{stamp}.csv")
    latest_path = os.path.join(save_dir, f"{prefix}_latest.csv")

    df.to_csv(timestamped_path, index=False)
    df.to_csv(latest_path, index=False)

    print(f"Saved {len(df)} rows to {timestamped_path}")

    return timestamped_path, latest_path
