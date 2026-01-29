#!/usr/bin/env python3
"""
Daily Odds Report Generator - Hornets Focus
Generates a weekly report focused on Charlotte Hornets games

Features:
- All Hornets games from the past week with odds breakdown
- Hornets-specific statistics (avg odds as favorite/underdog, home/away)
- Bookmaker performance analysis (best and worst performing bookies)

Usage:
    python -m data_jobs.reports.generate_daily_report [--days 7] [--output reports/]
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CI use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# DATA LOADING
# ============================================================================

def load_historical_data(data_dir: str, sport: str, days: int = 7) -> pd.DataFrame:
    """Load historical odds data for a sport."""
    if sport.lower() == "nba":
        pattern_dir = os.path.join(data_dir, "nba")
        prefix = "nba_odds_api_data_"
    else:
        pattern_dir = data_dir
        prefix = "odds_api_data_"

    cutoff = datetime.utcnow() - timedelta(days=days)

    all_data = []
    if os.path.exists(pattern_dir):
        for filename in os.listdir(pattern_dir):
            if filename.startswith(prefix) and filename.endswith(".csv") and "latest" not in filename:
                try:
                    date_str = filename.replace(prefix, "").replace(".csv", "")
                    file_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                    if file_date >= cutoff:
                        filepath = os.path.join(pattern_dir, filename)
                        df = pd.read_csv(filepath)
                        all_data.append(df)
                except (ValueError, IndexError):
                    continue

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        if "Timestamp Pulled" in combined.columns:
            combined["Timestamp Pulled"] = pd.to_datetime(combined["Timestamp Pulled"])
        return combined.sort_values("Timestamp Pulled")

    return pd.DataFrame()


def load_season_data(data_dir: str, sport: str) -> pd.DataFrame:
    """Load all available data for the season (for season-long charts)."""
    if sport.lower() == "nba":
        pattern_dir = os.path.join(data_dir, "nba")
        prefix = "nba_odds_api_data_"
    else:
        pattern_dir = data_dir
        prefix = "odds_api_data_"

    all_data = []
    if os.path.exists(pattern_dir):
        for filename in os.listdir(pattern_dir):
            if filename.startswith(prefix) and filename.endswith(".csv") and "latest" not in filename:
                try:
                    filepath = os.path.join(pattern_dir, filename)
                    df = pd.read_csv(filepath)
                    all_data.append(df)
                except Exception:
                    continue

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        if "Timestamp Pulled" in combined.columns:
            combined["Timestamp Pulled"] = pd.to_datetime(combined["Timestamp Pulled"])
        return combined.sort_values("Timestamp Pulled")

    return pd.DataFrame()


def load_actual_results(data_dir: str) -> pd.DataFrame:
    """
    Load actual game results from data/nba/actual_games/.

    Supports two formats:
    1. Simple: Date, Home Team, Away Team, Home Score, Away Score
    2. Basketball Reference: Date, Start (ET), Visitor/Neutral, PTS, Home/Neutral, PTS, ...
    """
    results_dir = os.path.join(data_dir, "nba", "actual_games")

    if not os.path.exists(results_dir):
        return pd.DataFrame()

    all_results = []
    for filename in os.listdir(results_dir):
        if filename.endswith(".csv"):
            try:
                filepath = os.path.join(results_dir, filename)
                df = pd.read_csv(filepath)

                # Handle Basketball Reference format
                # Columns: Date, Start (ET), Visitor/Neutral, PTS, Home/Neutral, PTS, Attend, LOG, Arena, Notes
                if 'Visitor/Neutral' in df.columns or 'Home/Neutral' in df.columns:
                    # Find the PTS columns by position
                    cols = df.columns.tolist()

                    # Rename columns to standard format
                    new_df = pd.DataFrame()
                    new_df['Date'] = df.get('Date', df.iloc[:, 0] if len(cols) > 0 else None)
                    new_df['Away Team'] = df.get('Visitor/Neutral', df.iloc[:, 2] if len(cols) > 2 else None)
                    new_df['Home Team'] = df.get('Home/Neutral', df.iloc[:, 4] if len(cols) > 4 else None)

                    # Find PTS columns - first one is visitor, second is home
                    pts_cols = [i for i, c in enumerate(cols) if c == 'PTS' or c.strip() == 'PTS']
                    if len(pts_cols) >= 2:
                        new_df['Away Score'] = df.iloc[:, pts_cols[0]]
                        new_df['Home Score'] = df.iloc[:, pts_cols[1]]
                    elif len(pts_cols) == 1:
                        # Try to find by position (column 3 = away pts, column 5 = home pts)
                        new_df['Away Score'] = df.iloc[:, 3] if len(cols) > 3 else None
                        new_df['Home Score'] = df.iloc[:, 5] if len(cols) > 5 else None

                    df = new_df
                else:
                    # Standard format - apply column mapping
                    col_map = {
                        'date': 'Date', 'game_date': 'Date', 'Date': 'Date',
                        'home_team': 'Home Team', 'Home Team': 'Home Team', 'home': 'Home Team',
                        'away_team': 'Away Team', 'Away Team': 'Away Team', 'away': 'Away Team',
                        'home_score': 'Home Score', 'Home Score': 'Home Score', 'home_pts': 'Home Score',
                        'away_score': 'Away Score', 'Away Score': 'Away Score', 'away_pts': 'Away Score',
                    }
                    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

                all_results.append(df)
            except Exception as e:
                print(f"  Warning: Could not load {filename}: {e}")
                continue

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)

        if 'Date' in combined.columns:
            combined['Date'] = pd.to_datetime(combined['Date'], errors='coerce')

        # Convert scores to numeric
        for col in ['Home Score', 'Away Score']:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors='coerce')

        # Drop rows with missing essential data
        combined = combined.dropna(subset=['Date', 'Home Team', 'Away Team', 'Home Score', 'Away Score'])

        return combined

    return pd.DataFrame()


# ============================================================================
# GAME FILTERING
# ============================================================================

def split_future_past_games(df: pd.DataFrame, hours_ahead: int = 48) -> tuple:
    """
    Split games into future (upcoming within hours_ahead) and past games.

    Returns: (future_df, past_df)
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Determine the game date column
    game_col = "Date of Game (ET)" if "Date of Game (ET)" in df.columns else "Date of Game"

    if game_col not in df.columns:
        return pd.DataFrame(), df

    # Get current time and cutoff time (timezone-naive for comparison)
    now = pd.Timestamp.now().tz_localize(None)
    cutoff = now + pd.Timedelta(hours=hours_ahead)

    # Parse game dates (ensure timezone-naive)
    df_copy = df.copy()
    df_copy['game_datetime'] = pd.to_datetime(df_copy[game_col], errors='coerce')

    # Remove timezone if present
    if hasattr(df_copy['game_datetime'].dtype, 'tz') and df_copy['game_datetime'].dtype.tz is not None:
        df_copy['game_datetime'] = df_copy['game_datetime'].dt.tz_localize(None)

    # Split into future and past
    future_mask = (df_copy['game_datetime'] >= now) & (df_copy['game_datetime'] <= cutoff)
    past_mask = df_copy['game_datetime'] < now

    future_df = df_copy[future_mask].copy()
    past_df = df_copy[past_mask].copy()

    # Drop the temporary column
    if 'game_datetime' in future_df.columns:
        future_df = future_df.drop(columns=['game_datetime'])
    if 'game_datetime' in past_df.columns:
        past_df = past_df.drop(columns=['game_datetime'])

    return future_df, past_df


# ============================================================================
# TEAM-SPECIFIC ANALYSIS (HORNETS FOCUS)
# ============================================================================

def get_team_games(df: pd.DataFrame, team_name: str = "Charlotte Hornets") -> pd.DataFrame:
    """
    Filter dataframe to only include games involving the specified team.
    Returns the latest odds snapshot for each unique game.
    """
    if df.empty:
        return pd.DataFrame()

    # Filter for games where team is home or away
    team_mask = (df["Home Team"] == team_name) | (df["Away Team"] == team_name)
    team_df = df[team_mask].copy()

    if team_df.empty:
        return pd.DataFrame()

    # Get the game date column
    game_col = "Date of Game (ET)" if "Date of Game (ET)" in team_df.columns else "Date of Game"

    # Get the latest snapshot for each game
    latest = team_df.sort_values("Timestamp Pulled").groupby(["Home Team", "Away Team", game_col]).last().reset_index()

    # Add a column indicating if team is home or away
    latest["Team Location"] = latest.apply(
        lambda r: "Home" if r["Home Team"] == team_name else "Away", axis=1
    )

    # Add team's odds columns
    latest["Team Odds"] = latest.apply(
        lambda r: r["Avg Home H2H Odds"] if r["Home Team"] == team_name else r["Avg Away H2H Odds"],
        axis=1
    )
    latest["Opponent Odds"] = latest.apply(
        lambda r: r["Avg Away H2H Odds"] if r["Home Team"] == team_name else r["Avg Home H2H Odds"],
        axis=1
    )
    latest["Opponent"] = latest.apply(
        lambda r: r["Away Team"] if r["Home Team"] == team_name else r["Home Team"],
        axis=1
    )
    latest["Is Favorite"] = latest["Team Odds"] < latest["Opponent Odds"]

    return latest.sort_values(game_col)


def calculate_team_odds_stats(team_df: pd.DataFrame, team_name: str = "Charlotte Hornets") -> Dict[str, Any]:
    """
    Calculate statistics on a team's odds across their games.
    """
    if team_df.empty:
        return {}

    stats = {
        "team_name": team_name,
        "total_games": len(team_df),
        "home_games": len(team_df[team_df["Team Location"] == "Home"]),
        "away_games": len(team_df[team_df["Team Location"] == "Away"]),
        "games_as_favorite": len(team_df[team_df["Is Favorite"]]),
        "games_as_underdog": len(team_df[~team_df["Is Favorite"]]),
    }

    # Average odds overall
    stats["avg_odds"] = round(team_df["Team Odds"].mean(), 1)

    # Average odds when home vs away
    home_df = team_df[team_df["Team Location"] == "Home"]
    away_df = team_df[team_df["Team Location"] == "Away"]

    if not home_df.empty:
        stats["avg_home_odds"] = round(home_df["Team Odds"].mean(), 1)
    if not away_df.empty:
        stats["avg_away_odds"] = round(away_df["Team Odds"].mean(), 1)

    # Average odds when favorite vs underdog
    fav_df = team_df[team_df["Is Favorite"]]
    dog_df = team_df[~team_df["Is Favorite"]]

    if not fav_df.empty:
        stats["avg_favorite_odds"] = round(fav_df["Team Odds"].mean(), 1)
    if not dog_df.empty:
        stats["avg_underdog_odds"] = round(dog_df["Team Odds"].mean(), 1)

    # Best and worst odds
    stats["best_odds"] = round(team_df["Team Odds"].max(), 1)  # Highest (most plus or least minus)
    stats["worst_odds"] = round(team_df["Team Odds"].min(), 1)  # Most negative

    return stats


def calculate_bookie_performance(df: pd.DataFrame, results_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculate how each bookmaker performs compared to market average.

    If results_df is provided, calculates actual prediction accuracy.
    Otherwise, calculates how much each bookie deviates from average (variance).

    Returns a DataFrame with bookie rankings.
    """
    if df.empty:
        return pd.DataFrame()

    sportsbooks = get_sportsbooks(df)
    if not sportsbooks:
        return pd.DataFrame()

    # Get latest snapshot per game
    game_col = "Date of Game (ET)" if "Date of Game (ET)" in df.columns else "Date of Game"
    latest = df.sort_values("Timestamp Pulled").groupby(["Home Team", "Away Team", game_col]).last().reset_index()

    # If we have results, calculate accuracy
    if results_df is not None and not results_df.empty:
        return _calculate_bookie_accuracy(latest, results_df, sportsbooks, game_col)

    # Otherwise calculate variance from average
    return _calculate_bookie_variance(latest, sportsbooks)


def _calculate_bookie_accuracy(latest: pd.DataFrame, results_df: pd.DataFrame,
                                sportsbooks: List[str], game_col: str) -> pd.DataFrame:
    """Calculate prediction accuracy for each bookie."""
    # Build results lookup
    results_dict = {}
    for _, row in results_df.iterrows():
        if pd.notna(row.get("Date")):
            date_key = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
            home = row.get("Home Team", "")
            away = row.get("Away Team", "")
            if pd.notna(row.get("Home Score")) and pd.notna(row.get("Away Score")):
                winner = home if row["Home Score"] > row["Away Score"] else away
                results_dict[(date_key, home, away)] = winner

    bookie_stats = {book: {"correct": 0, "total": 0} for book in sportsbooks}
    bookie_stats["Market Average"] = {"correct": 0, "total": 0}

    for _, row in latest.iterrows():
        game_date = str(row.get(game_col, ""))[:10]
        home = row["Home Team"]
        away = row["Away Team"]
        result_key = (game_date, home, away)

        if result_key not in results_dict:
            continue

        actual_winner = results_dict[result_key]

        # Market average prediction
        avg_home = row.get("Avg Home H2H Odds")
        avg_away = row.get("Avg Away H2H Odds")
        if pd.notna(avg_home) and pd.notna(avg_away):
            avg_predicted = home if avg_home < avg_away else away
            bookie_stats["Market Average"]["total"] += 1
            if avg_predicted == actual_winner:
                bookie_stats["Market Average"]["correct"] += 1

        # Each bookie's prediction
        for book in sportsbooks:
            home_col = f"Home {book} H2H Odds"
            away_col = f"Away {book} H2H Odds"
            book_home = row.get(home_col)
            book_away = row.get(away_col)

            if pd.notna(book_home) and pd.notna(book_away):
                predicted = home if book_home < book_away else away
                bookie_stats[book]["total"] += 1
                if predicted == actual_winner:
                    bookie_stats[book]["correct"] += 1

    # Build results dataframe
    results = []
    for book, stats in bookie_stats.items():
        if stats["total"] > 0:
            pct = round(100 * stats["correct"] / stats["total"], 1)
            results.append({
                "Bookmaker": book,
                "Correct Predictions": stats["correct"],
                "Total Games": stats["total"],
                "Accuracy %": pct,
                "Metric": "Prediction Accuracy"
            })

    result_df = pd.DataFrame(results).sort_values("Accuracy %", ascending=False)
    return result_df


def _calculate_bookie_variance(latest: pd.DataFrame, sportsbooks: List[str]) -> pd.DataFrame:
    """Calculate how much each bookie deviates from market average."""
    bookie_diffs = {book: [] for book in sportsbooks}

    for _, row in latest.iterrows():
        avg_home = row.get("Avg Home H2H Odds")
        avg_away = row.get("Avg Away H2H Odds")

        if pd.isna(avg_home) or pd.isna(avg_away):
            continue

        for book in sportsbooks:
            home_col = f"Home {book} H2H Odds"
            away_col = f"Away {book} H2H Odds"
            book_home = row.get(home_col)
            book_away = row.get(away_col)

            if pd.notna(book_home) and pd.notna(book_away):
                diff = abs(book_home - avg_home) + abs(book_away - avg_away)
                bookie_diffs[book].append(diff)

    # Build results
    results = []
    for book, diffs in bookie_diffs.items():
        if diffs:
            avg_diff = round(np.mean(diffs), 2)
            results.append({
                "Bookmaker": book,
                "Avg Deviation from Market": avg_diff,
                "Games Analyzed": len(diffs),
                "Metric": "Variance from Average"
            })

    result_df = pd.DataFrame(results).sort_values("Avg Deviation from Market")
    return result_df


def get_team_odds_by_bookie(team_df: pd.DataFrame, team_name: str = "Charlotte Hornets") -> pd.DataFrame:
    """
    Get odds from each bookmaker for team games.
    Returns a DataFrame with game info and each bookie's odds for the team.
    """
    if team_df.empty:
        return pd.DataFrame()

    sportsbooks = get_sportsbooks(team_df)
    game_col = "Date of Game (ET)" if "Date of Game (ET)" in team_df.columns else "Date of Game"

    records = []
    for _, row in team_df.iterrows():
        is_home = row["Home Team"] == team_name
        opponent = row["Away Team"] if is_home else row["Home Team"]

        game_record = {
            "Date": str(row.get(game_col, ""))[:10],
            "Opponent": opponent,
            "Location": "Home" if is_home else "Away",
            "Avg Odds": row["Avg Home H2H Odds"] if is_home else row["Avg Away H2H Odds"],
        }

        # Add each bookie's odds
        for book in sportsbooks:
            if is_home:
                odds_col = f"Home {book} H2H Odds"
            else:
                odds_col = f"Away {book} H2H Odds"
            game_record[book] = row.get(odds_col)

        records.append(game_record)

    return pd.DataFrame(records)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def get_sportsbooks(df: pd.DataFrame) -> List[str]:
    """Extract list of sportsbook names from column headers."""
    sportsbooks = []
    for col in df.columns:
        if "H2H Odds" in col and col.startswith("Home "):
            book = col.replace("Home ", "").replace(" H2H Odds", "")
            if book not in ["Avg"] and book not in sportsbooks:
                sportsbooks.append(book)
    return sportsbooks


def calculate_bookmaker_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate which bookmakers deviate most from average odds."""
    if df.empty:
        return pd.DataFrame()

    sportsbooks = get_sportsbooks(df)
    if not sportsbooks:
        return pd.DataFrame()

    # Get latest snapshot per game
    latest = df.sort_values("Timestamp Pulled").groupby(["Home Team", "Away Team"]).last().reset_index()

    variances = []
    for book in sportsbooks:
        home_col = f"Home {book} H2H Odds"
        away_col = f"Away {book} H2H Odds"

        if home_col not in latest.columns:
            continue

        for _, row in latest.iterrows():
            book_home = row.get(home_col)
            book_away = row.get(away_col)
            avg_home = row.get("Avg Home H2H Odds")
            avg_away = row.get("Avg Away H2H Odds")

            if pd.notna(book_home) and pd.notna(avg_home):
                diff_home = book_home - avg_home
                diff_away = book_away - avg_away if pd.notna(book_away) and pd.notna(avg_away) else 0

                variances.append({
                    "Sportsbook": book,
                    "Game": f"{row['Away Team'][:3]} @ {row['Home Team'][:3]}",
                    "Home Team": row["Home Team"],
                    "Away Team": row["Away Team"],
                    "Book Home Odds": book_home,
                    "Avg Home Odds": avg_home,
                    "Home Diff": diff_home,
                    "Book Away Odds": book_away,
                    "Avg Away Odds": avg_away,
                    "Away Diff": diff_away,
                    "Total Diff": abs(diff_home) + abs(diff_away),
                })

    return pd.DataFrame(variances)


def calculate_fanduel_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare FanDuel odds to average for all games."""
    if df.empty:
        return pd.DataFrame()

    latest = df.sort_values("Timestamp Pulled").groupby(["Home Team", "Away Team"]).last().reset_index()

    comparisons = []
    for _, row in latest.iterrows():
        fd_home = row.get("Home FanDuel H2H Odds")
        fd_away = row.get("Away FanDuel H2H Odds")
        avg_home = row.get("Avg Home H2H Odds")
        avg_away = row.get("Avg Away H2H Odds")

        if pd.notna(fd_home) and pd.notna(avg_home):
            comparisons.append({
                "Game": f"{row['Away Team'][:3]} @ {row['Home Team'][:3]}",
                "Home Team": row["Home Team"],
                "Away Team": row["Away Team"],
                "FanDuel Home": fd_home,
                "Avg Home": avg_home,
                "Home Diff": fd_home - avg_home,
                "FanDuel Away": fd_away if pd.notna(fd_away) else None,
                "Avg Away": avg_away if pd.notna(avg_away) else None,
                "Away Diff": (fd_away - avg_away) if pd.notna(fd_away) and pd.notna(avg_away) else None,
            })

    return pd.DataFrame(comparisons)


def calculate_team_season_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average odds for each team across the season."""
    if df.empty:
        return pd.DataFrame()

    team_odds = defaultdict(list)

    for _, row in df.iterrows():
        home = row.get("Home Team")
        away = row.get("Away Team")
        home_odds = row.get("Avg Home H2H Odds")
        away_odds = row.get("Avg Away H2H Odds")
        timestamp = row.get("Timestamp Pulled")

        if pd.notna(home) and pd.notna(home_odds):
            team_odds[home].append({"odds": home_odds, "timestamp": timestamp, "is_home": True})
        if pd.notna(away) and pd.notna(away_odds):
            team_odds[away].append({"odds": away_odds, "timestamp": timestamp, "is_home": False})

    # Calculate averages
    summary = []
    for team, odds_list in team_odds.items():
        odds_values = [o["odds"] for o in odds_list]
        avg = np.mean(odds_values)
        summary.append({
            "Team": team,
            "Avg Odds": avg,
            "Games": len(odds_list),
            "History": odds_list,
        })

    return pd.DataFrame(summary).sort_values("Avg Odds")


def calculate_bookmaker_accuracy(odds_df: pd.DataFrame, results_df: pd.DataFrame,
                                  days_back: int = 1) -> Dict[str, Any]:
    """
    Calculate bookmaker accuracy by comparing predicted favorites to actual winners.

    Returns accuracy stats for: yesterday, last 7 days, last 30 days, all time
    """
    if odds_df.empty or results_df.empty:
        return {}

    sportsbooks = get_sportsbooks(odds_df)
    if not sportsbooks:
        return {}

    # Map game dates to results
    results_dict = {}
    for _, row in results_df.iterrows():
        if pd.notna(row.get("Date")):
            date_key = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
            home = row.get("Home Team", "")
            away = row.get("Away Team", "")

            if pd.notna(row.get("Home Score")) and pd.notna(row.get("Away Score")):
                winner = home if row["Home Score"] > row["Away Score"] else away
                results_dict[(date_key, home, away)] = {
                    "winner": winner,
                    "home_score": row["Home Score"],
                    "away_score": row["Away Score"],
                }

    # Get pre-game odds (last snapshot before game)
    game_col = "Date of Game (ET)" if "Date of Game (ET)" in odds_df.columns else "Date of Game"

    accuracy = {book: {"correct": 0, "total": 0, "by_period": {}} for book in sportsbooks + ["Average"]}
    underdog_wins = []

    # Group odds by game and get final pre-game snapshot
    for (home, away, game_date), group in odds_df.groupby(["Home Team", "Away Team", game_col]):
        if not isinstance(game_date, str):
            game_date = str(game_date)
        date_key = game_date[:10]

        # Check if we have results for this game
        result_key = (date_key, home, away)
        if result_key not in results_dict:
            continue

        result = results_dict[result_key]
        actual_winner = result["winner"]

        # Get final pre-game odds
        final_odds = group.sort_values("Timestamp Pulled").iloc[-1]
        avg_home_odds = final_odds.get("Avg Home H2H Odds")
        avg_away_odds = final_odds.get("Avg Away H2H Odds")

        if pd.isna(avg_home_odds) or pd.isna(avg_away_odds):
            continue

        # Determine favorite (lower odds = favorite for negative, higher for positive)
        def is_favorite(odds):
            """More negative = bigger favorite, positive = underdog"""
            return odds < 0 and abs(odds) > 100

        avg_predicted = home if avg_home_odds < avg_away_odds else away

        # Check if underdog won
        underdog = away if avg_predicted == home else home
        if actual_winner == underdog:
            underdog_wins.append({
                "Date": date_key,
                "Game": f"{away} @ {home}",
                "Underdog": underdog,
                "Underdog Odds": avg_away_odds if underdog == away else avg_home_odds,
                "Final Score": f"{result['away_score']}-{result['home_score']}" if underdog == away else f"{result['home_score']}-{result['away_score']}",
            })

        # Check each sportsbook's accuracy
        for book in sportsbooks:
            home_col = f"Home {book} H2H Odds"
            away_col = f"Away {book} H2H Odds"

            book_home = final_odds.get(home_col)
            book_away = final_odds.get(away_col)

            if pd.notna(book_home) and pd.notna(book_away):
                predicted = home if book_home < book_away else away
                if predicted == actual_winner:
                    accuracy[book]["correct"] += 1
                accuracy[book]["total"] += 1

        # Average accuracy
        if avg_predicted == actual_winner:
            accuracy["Average"]["correct"] += 1
        accuracy["Average"]["total"] += 1

    # Calculate percentages
    for book in accuracy:
        if accuracy[book]["total"] > 0:
            accuracy[book]["pct"] = round(100 * accuracy[book]["correct"] / accuracy[book]["total"], 1)
        else:
            accuracy[book]["pct"] = 0

    return {
        "accuracy": accuracy,
        "underdog_wins": underdog_wins,
    }


def calculate_odds_movement(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate odds movement for each game/team over time."""
    if df.empty:
        return df

    game_col = "Date of Game (ET)" if "Date of Game (ET)" in df.columns else "Date of Game"

    movements = []
    for (home, away, game_date), group in df.groupby(["Home Team", "Away Team", game_col]):
        if len(group) < 2:
            continue

        group = group.sort_values("Timestamp Pulled")
        first = group.iloc[0]
        last = group.iloc[-1]

        movements.append({
            "Home Team": home,
            "Away Team": away,
            "Game Date": game_date,
            "Initial Home H2H": first.get("Avg Home H2H Odds"),
            "Final Home H2H": last.get("Avg Home H2H Odds"),
            "H2H Movement": _safe_subtract(last.get("Avg Home H2H Odds"), first.get("Avg Home H2H Odds")),
            "Initial Home Spread": first.get("Avg Home Spread Odds"),
            "Final Home Spread": last.get("Avg Home Spread Odds"),
            "Spread Movement": _safe_subtract(last.get("Avg Home Spread Odds"), first.get("Avg Home Spread Odds")),
            "Initial Total": first.get("Avg Total Points"),
            "Final Total": last.get("Avg Total Points"),
            "Total Movement": _safe_subtract(last.get("Avg Total Points"), first.get("Avg Total Points")),
            "Data Points": len(group),
        })

    return pd.DataFrame(movements)


def _safe_subtract(a, b) -> Optional[float]:
    """Safely subtract two values that might be None."""
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    return round(a - b, 2)


def calculate_nba_future_games_analysis(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze upcoming NBA games for the next 48 hours.

    Returns dictionary with:
    - biggest_underdogs: Teams with highest positive odds
    - biggest_favorites: Teams with most negative odds
    - fanduel_over_favored: Games where FanDuel odds are better than avg for home
    - fanduel_under_favored: Games where FanDuel odds are worse than avg for home
    """
    if df.empty:
        return {}

    # Get latest odds for each game
    latest = df.sort_values("Timestamp Pulled").groupby(["Home Team", "Away Team"]).last().reset_index()

    # Biggest underdogs (highest positive odds or least negative)
    underdogs = []
    for _, row in latest.iterrows():
        home_odds = row.get("Avg Home H2H Odds")
        away_odds = row.get("Avg Away H2H Odds")

        if pd.notna(home_odds):
            underdogs.append({
                "Team": row["Home Team"],
                "Opponent": row["Away Team"],
                "Location": "Home",
                "Odds": home_odds,
                "Game": f"{row['Away Team'][:3]} @ {row['Home Team'][:3]}"
            })
        if pd.notna(away_odds):
            underdogs.append({
                "Team": row["Away Team"],
                "Opponent": row["Home Team"],
                "Location": "Away",
                "Odds": away_odds,
                "Game": f"{row['Away Team'][:3]} @ {row['Home Team'][:3]}"
            })

    underdogs_df = pd.DataFrame(underdogs).sort_values("Odds", ascending=False)

    # Biggest favorites (most negative odds)
    favorites_df = pd.DataFrame(underdogs).sort_values("Odds", ascending=True)

    # FanDuel comparison
    fd_comparisons = []
    for _, row in latest.iterrows():
        fd_home = row.get("Home FanDuel H2H Odds")
        avg_home = row.get("Avg Home H2H Odds")
        fd_away = row.get("Away FanDuel H2H Odds")
        avg_away = row.get("Avg Away H2H Odds")

        if pd.notna(fd_home) and pd.notna(avg_home):
            home_diff = fd_home - avg_home
            fd_comparisons.append({
                "Game": f"{row['Away Team'][:3]} @ {row['Home Team'][:3]}",
                "Team": row["Home Team"],
                "FanDuel": fd_home,
                "Average": avg_home,
                "Difference": home_diff
            })

        if pd.notna(fd_away) and pd.notna(avg_away):
            away_diff = fd_away - avg_away
            fd_comparisons.append({
                "Game": f"{row['Away Team'][:3]} @ {row['Home Team'][:3]}",
                "Team": row["Away Team"],
                "FanDuel": fd_away,
                "Average": avg_away,
                "Difference": away_diff
            })

    fd_df = pd.DataFrame(fd_comparisons)

    # Over-favored: FanDuel offers better odds (more positive difference)
    fd_over = fd_df.sort_values("Difference", ascending=False)
    # Under-favored: FanDuel offers worse odds (more negative difference)
    fd_under = fd_df.sort_values("Difference", ascending=True)

    return {
        "biggest_underdogs": underdogs_df,
        "biggest_favorites": favorites_df,
        "fanduel_over_favored": fd_over,
        "fanduel_under_favored": fd_under
    }


def calculate_moneyline_accuracy_table(odds_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate moneyline accuracy for all bookmakers.
    Returns table with last week, past month, and full season stats.
    """
    if odds_df.empty or results_df.empty:
        return pd.DataFrame()

    # Get all sportsbooks
    sportsbooks = get_sportsbooks(odds_df)

    # Map game dates to results
    results_dict = {}
    for _, row in results_df.iterrows():
        if pd.notna(row.get("Date")):
            date_key = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
            home = row.get("Home Team", "")
            away = row.get("Away Team", "")

            if pd.notna(row.get("Home Score")) and pd.notna(row.get("Away Score")):
                winner = home if row["Home Score"] > row["Away Score"] else away
                results_dict[(date_key, home, away)] = {
                    "winner": winner,
                    "home_score": row["Home Score"],
                    "away_score": row["Away Score"],
                }

    game_col = "Date of Game (ET)" if "Date of Game (ET)" in odds_df.columns else "Date of Game"

    # Track accuracy for different time periods
    # Initialize stats for Average + all sportsbooks
    stats = {"Avg of all": {
        "last_week": {"correct": 0, "total": 0},
        "past_month": {"correct": 0, "total": 0},
        "full_season": {"correct": 0, "total": 0}
    }}

    for book in sportsbooks:
        stats[book] = {
            "last_week": {"correct": 0, "total": 0},
            "past_month": {"correct": 0, "total": 0},
            "full_season": {"correct": 0, "total": 0}
        }

    now = datetime.now()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    # Group odds by game and get final pre-game snapshot
    for (home, away, game_date), group in odds_df.groupby(["Home Team", "Away Team", game_col]):
        if not isinstance(game_date, str):
            game_date = str(game_date)
        date_key = game_date[:10]

        # Check if we have results for this game
        result_key = (date_key, home, away)
        if result_key not in results_dict:
            continue

        result = results_dict[result_key]
        actual_winner = result["winner"]

        # Get final pre-game odds
        final_odds = group.sort_values("Timestamp Pulled").iloc[-1]

        # Determine time period
        game_datetime = pd.to_datetime(game_date, errors='coerce')
        is_last_week = game_datetime >= week_ago if pd.notna(game_datetime) else False
        is_past_month = game_datetime >= month_ago if pd.notna(game_datetime) else False

        # Check Average prediction
        avg_home_odds = final_odds.get("Avg Home H2H Odds")
        avg_away_odds = final_odds.get("Avg Away H2H Odds")

        if pd.notna(avg_home_odds) and pd.notna(avg_away_odds):
            avg_predicted = home if avg_home_odds < avg_away_odds else away
            correct = avg_predicted == actual_winner

            stats["Avg of all"]["full_season"]["total"] += 1
            if correct:
                stats["Avg of all"]["full_season"]["correct"] += 1

            if is_past_month:
                stats["Avg of all"]["past_month"]["total"] += 1
                if correct:
                    stats["Avg of all"]["past_month"]["correct"] += 1

            if is_last_week:
                stats["Avg of all"]["last_week"]["total"] += 1
                if correct:
                    stats["Avg of all"]["last_week"]["correct"] += 1

        # Check each sportsbook's prediction
        for book in sportsbooks:
            home_col = f"Home {book} H2H Odds"
            away_col = f"Away {book} H2H Odds"

            book_home_odds = final_odds.get(home_col)
            book_away_odds = final_odds.get(away_col)

            if pd.notna(book_home_odds) and pd.notna(book_away_odds):
                book_predicted = home if book_home_odds < book_away_odds else away
                correct = book_predicted == actual_winner

                stats[book]["full_season"]["total"] += 1
                if correct:
                    stats[book]["full_season"]["correct"] += 1

                if is_past_month:
                    stats[book]["past_month"]["total"] += 1
                    if correct:
                        stats[book]["past_month"]["correct"] += 1

                if is_last_week:
                    stats[book]["last_week"]["total"] += 1
                    if correct:
                        stats[book]["last_week"]["correct"] += 1

    # Build table
    table_data = []
    for bookie, periods in stats.items():
        row = {"Bookie": bookie}

        # Last week
        if periods["last_week"]["total"] > 0:
            pct = round(100 * periods["last_week"]["correct"] / periods["last_week"]["total"])
            row["Last week money line accuracy"] = f"{pct}%"
        else:
            row["Last week money line accuracy"] = ""

        # Past month
        if periods["past_month"]["total"] > 0:
            pct = round(100 * periods["past_month"]["correct"] / periods["past_month"]["total"])
            row["past month money line accuracy"] = f"{pct}%"
        else:
            row["past month money line accuracy"] = ""

        # Full season
        if periods["full_season"]["total"] > 0:
            pct = round(100 * periods["full_season"]["correct"] / periods["full_season"]["total"])
            row["Full Season money Line Accuracy"] = f"{pct}%"
        else:
            row["Full Season money Line Accuracy"] = ""

        table_data.append(row)

    return pd.DataFrame(table_data)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_bookmaker_variance(variance_df: pd.DataFrame, output_dir: str, sport: str) -> Optional[str]:
    """Plot which bookmakers have biggest differences from average."""
    if variance_df.empty:
        return None

    # Aggregate by sportsbook
    book_summary = variance_df.groupby("Sportsbook").agg({
        "Total Diff": "mean",
        "Home Diff": "mean",
        "Away Diff": "mean",
    }).reset_index().sort_values("Total Diff", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(book_summary))
    bars = ax.bar(x, book_summary["Total Diff"], color='#3498db', alpha=0.8)

    ax.set_xlabel("Sportsbook")
    ax.set_ylabel("Avg Absolute Difference from Market Average")
    ax.set_title(f"{sport.upper()} - Bookmaker Variance from Average Odds")
    ax.set_xticks(x)
    ax.set_xticklabels(book_summary["Sportsbook"], rotation=45, ha='right')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight top variance
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{sport}_bookmaker_variance.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def plot_fanduel_comparison(fd_df: pd.DataFrame, output_dir: str, sport: str) -> Optional[str]:
    """Plot FanDuel odds vs average for all games."""
    if fd_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, max(6, len(fd_df) * 0.6)))

    games = fd_df["Game"].tolist()
    y_pos = range(len(games))

    # Plot home team differences
    home_diff = fd_df["Home Diff"].fillna(0)
    colors = ['#27ae60' if d > 0 else '#e74c3c' for d in home_diff]

    bars = ax.barh(y_pos, home_diff, color=colors, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(games)
    ax.set_xlabel("FanDuel vs Average (+ = FanDuel offers better odds for home team)")
    ax.set_title(f"{sport.upper()} - FanDuel vs Market Average (Moneyline)")
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, val in zip(bars, home_diff):
        if val != 0:
            ax.text(val + (1 if val >= 0 else -1), bar.get_y() + bar.get_height()/2,
                    f'{val:+.0f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{sport}_fanduel_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def plot_season_team_odds(season_df: pd.DataFrame, output_dir: str, sport: str) -> Optional[str]:
    """Plot season MONEYLINE odds history for top 5 and bottom 5 teams."""
    if season_df.empty or len(season_df) < 10:
        return None

    # Get top 5 (most favored - lowest/most negative average odds)
    top_5 = season_df.head(5)
    # Get bottom 5 (biggest underdogs - highest average odds)
    bottom_5 = season_df.tail(5)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Colors for teams
    colors_top = plt.cm.Greens(np.linspace(0.4, 0.9, 5))
    colors_bottom = plt.cm.Reds(np.linspace(0.4, 0.9, 5))

    # Plot top 5 (favorites)
    ax1 = axes[0]
    for i, (_, row) in enumerate(top_5.iterrows()):
        history = row["History"]
        if history:
            dates = [h["timestamp"] for h in history]
            # Only plot moneyline odds (H2H)
            odds = [h["odds"] for h in history]
            ax1.plot(dates, odds, label=f"{row['Team']} ({row['Avg Odds']:.0f})",
                    color=colors_top[i], linewidth=2, alpha=0.8)

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Moneyline Odds")
    ax1.set_title("Top 5 Teams (Most Favored) - Moneyline Only")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.axhline(y=-100, color='gray', linestyle='--', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Plot bottom 5 (underdogs)
    ax2 = axes[1]
    for i, (_, row) in enumerate(bottom_5.iterrows()):
        history = row["History"]
        if history:
            dates = [h["timestamp"] for h in history]
            # Only plot moneyline odds (H2H)
            odds = [h["odds"] for h in history]
            ax2.plot(dates, odds, label=f"{row['Team']} ({row['Avg Odds']:.0f})",
                    color=colors_bottom[i], linewidth=2, alpha=0.8)

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Moneyline Odds")
    ax2.set_title("Bottom 5 Teams (Biggest Underdogs) - Moneyline Only")
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.suptitle(f"{sport.upper()} Season Moneyline Odds by Team", fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{sport}_season_team_odds.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def plot_bookmaker_accuracy(accuracy_data: Dict, output_dir: str, sport: str) -> Optional[str]:
    """Plot bookmaker accuracy comparison."""
    if not accuracy_data or "accuracy" not in accuracy_data:
        return None

    accuracy = accuracy_data["accuracy"]

    # Filter to books with data
    books = [(k, v) for k, v in accuracy.items() if v.get("total", 0) > 0]
    if not books:
        return None

    # Sort by accuracy
    books.sort(key=lambda x: x[1].get("pct", 0), reverse=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    names = [b[0] for b in books]
    pcts = [b[1].get("pct", 0) for b in books]
    totals = [b[1].get("total", 0) for b in books]

    x = range(len(names))
    colors = ['#27ae60' if p >= 50 else '#e74c3c' for p in pcts]
    bars = ax.bar(x, pcts, color=colors, alpha=0.8)

    ax.set_xlabel("Sportsbook")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{sport.upper()} - Bookmaker Prediction Accuracy (Favorites vs Actual Winners)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    # Add labels
    for bar, pct, total in zip(bars, pcts, totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%\n({total})', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{sport}_bookmaker_accuracy.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def plot_daily_summary(df: pd.DataFrame, output_dir: str, sport: str) -> Optional[str]:
    """Create a summary visualization of all games for the day."""
    if df.empty:
        return None

    latest = df.sort_values("Timestamp Pulled").groupby(["Home Team", "Away Team"]).last().reset_index()

    if latest.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(latest) * 0.5)))

    # Plot 1: Spread comparison
    ax1 = axes[0]
    games = [f"{r['Away Team'][:3]} @ {r['Home Team'][:3]}" for _, r in latest.iterrows()]
    home_spreads = latest["Avg Home Spread Odds"].fillna(0)
    away_spreads = latest["Avg Away Spread Odds"].fillna(0)

    y_pos = range(len(games))
    ax1.barh(y_pos, home_spreads, height=0.4, label='Home', alpha=0.8, color='#2ecc71')
    ax1.barh([y + 0.4 for y in y_pos], away_spreads, height=0.4, label='Away', alpha=0.8, color='#e74c3c')

    ax1.set_yticks([y + 0.2 for y in y_pos])
    ax1.set_yticklabels(games)
    ax1.set_xlabel("Spread Odds")
    ax1.set_title("Spread Odds by Game")
    ax1.legend()
    ax1.axvline(x=-110, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='x')

    # Plot 2: Moneyline comparison
    ax2 = axes[1]
    home_h2h = latest["Avg Home H2H Odds"].fillna(0)
    away_h2h = latest["Avg Away H2H Odds"].fillna(0)

    ax2.barh(y_pos, home_h2h, height=0.4, label='Home', alpha=0.8, color='#2ecc71')
    ax2.barh([y + 0.4 for y in y_pos], away_h2h, height=0.4, label='Away', alpha=0.8, color='#e74c3c')

    ax2.set_yticks([y + 0.2 for y in y_pos])
    ax2.set_yticklabels(games)
    ax2.set_xlabel("Moneyline Odds")
    ax2.set_title("Moneyline Odds by Game")
    ax2.legend()
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.suptitle(f"{sport.upper()} Daily Odds Summary - {datetime.now().strftime('%Y-%m-%d')}", fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{sport}_daily_summary.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def plot_odds_movement(movements_df: pd.DataFrame, output_dir: str, sport: str) -> Optional[str]:
    """Plot odds movement for upcoming games."""
    if movements_df.empty:
        return None

    movements = movements_df[movements_df["H2H Movement"].notna()].copy()
    if movements.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, max(6, len(movements) * 0.6)))

    games = [f"{r['Away Team'][:3]} @ {r['Home Team'][:3]}" for _, r in movements.iterrows()]
    h2h_movement = movements["H2H Movement"].fillna(0)

    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in h2h_movement]

    y_pos = range(len(games))
    bars = ax.barh(y_pos, h2h_movement, color=colors, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(games)
    ax.set_xlabel("Moneyline Movement (+ = odds improved for home)")
    ax.set_title(f"{sport.upper()} Odds Movement - Home Team Moneyline Changes")
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    for bar, val in zip(bars, h2h_movement):
        if val != 0:
            ax.text(val + (2 if val >= 0 else -2), bar.get_y() + bar.get_height()/2,
                    f'{val:+.0f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{sport}_odds_movement.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def plot_nba_future_games(analysis: Dict, output_dir: str) -> Dict[str, str]:
    """
    Create 4 visualizations for upcoming NBA games (next 48 hours).

    Returns dict with paths to:
    - biggest_underdogs
    - biggest_favorites
    - fanduel_over_favored
    - fanduel_under_favored
    """
    paths = {}

    if not analysis:
        return paths

    # 1. Biggest Underdogs
    underdogs_df = analysis.get("biggest_underdogs")
    if underdogs_df is not None and not underdogs_df.empty:
        top_underdogs = underdogs_df.head(10)
        fig, ax = plt.subplots(figsize=(12, max(6, len(top_underdogs) * 0.5)))

        labels = [f"{r['Team'][:10]} vs {r['Opponent'][:10]}" for _, r in top_underdogs.iterrows()]
        odds = top_underdogs["Odds"].values
        y_pos = range(len(labels))

        bars = ax.barh(y_pos, odds, color='#e74c3c', alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Moneyline Odds")
        ax.set_title("NBA - Biggest Underdogs (Next 48 Hours)")
        ax.grid(True, alpha=0.3, axis='x')

        for bar, val in zip(bars, odds):
            ax.text(val + 10, bar.get_y() + bar.get_height()/2,
                    f'{val:+.0f}' if val >= 0 else f'{val:.0f}',
                    va='center', ha='left', fontsize=9)

        plt.tight_layout()
        path = os.path.join(output_dir, "nba_future_biggest_underdogs.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths["biggest_underdogs"] = path

    # 2. Biggest Favorites
    favorites_df = analysis.get("biggest_favorites")
    if favorites_df is not None and not favorites_df.empty:
        top_favorites = favorites_df.head(10)
        fig, ax = plt.subplots(figsize=(12, max(6, len(top_favorites) * 0.5)))

        labels = [f"{r['Team'][:10]} vs {r['Opponent'][:10]}" for _, r in top_favorites.iterrows()]
        odds = top_favorites["Odds"].values
        y_pos = range(len(labels))

        bars = ax.barh(y_pos, odds, color='#2ecc71', alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Moneyline Odds")
        ax.set_title("NBA - Biggest Favorites (Next 48 Hours)")
        ax.grid(True, alpha=0.3, axis='x')

        for bar, val in zip(bars, odds):
            ax.text(val - 10, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}',
                    va='center', ha='right', fontsize=9)

        plt.tight_layout()
        path = os.path.join(output_dir, "nba_future_biggest_favorites.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths["biggest_favorites"] = path

    # 3. FanDuel Over-Favored (better odds than avg)
    fd_over_df = analysis.get("fanduel_over_favored")
    if fd_over_df is not None and not fd_over_df.empty:
        top_over = fd_over_df[fd_over_df["Difference"] > 0].head(10)
        if not top_over.empty:
            fig, ax = plt.subplots(figsize=(12, max(6, len(top_over) * 0.5)))

            labels = [f"{r['Team'][:10]} ({r['Game']})" for _, r in top_over.iterrows()]
            diffs = top_over["Difference"].values
            y_pos = range(len(labels))

            bars = ax.barh(y_pos, diffs, color='#27ae60', alpha=0.8)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel("FanDuel vs Average Difference")
            ax.set_title("NBA - FanDuel Offers Better Odds (Next 48 Hours)")
            ax.grid(True, alpha=0.3, axis='x')

            for bar, val in zip(bars, diffs):
                ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                        f'{val:+.0f}',
                        va='center', ha='left', fontsize=9)

            plt.tight_layout()
            path = os.path.join(output_dir, "nba_future_fanduel_over_favored.png")
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            paths["fanduel_over_favored"] = path

    # 4. FanDuel Under-Favored (worse odds than avg)
    fd_under_df = analysis.get("fanduel_under_favored")
    if fd_under_df is not None and not fd_under_df.empty:
        top_under = fd_under_df[fd_under_df["Difference"] < 0].head(10)
        if not top_under.empty:
            fig, ax = plt.subplots(figsize=(12, max(6, len(top_under) * 0.5)))

            labels = [f"{r['Team'][:10]} ({r['Game']})" for _, r in top_under.iterrows()]
            diffs = top_under["Difference"].values
            y_pos = range(len(labels))

            bars = ax.barh(y_pos, diffs, color='#c0392b', alpha=0.8)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel("FanDuel vs Average Difference")
            ax.set_title("NBA - FanDuel Offers Worse Odds (Next 48 Hours)")
            ax.grid(True, alpha=0.3, axis='x')

            for bar, val in zip(bars, diffs):
                ax.text(val - 1, bar.get_y() + bar.get_height()/2,
                        f'{val:.0f}',
                        va='center', ha='right', fontsize=9)

            plt.tight_layout()
            path = os.path.join(output_dir, "nba_future_fanduel_under_favored.png")
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            paths["fanduel_under_favored"] = path

    return paths


# ============================================================================
# HTML REPORT GENERATION
# ============================================================================

def generate_html_report(hornets_data: dict, bookie_performance: pd.DataFrame,
                         output_dir: str, date_str: str) -> str:
    """Generate a Hornets-focused HTML report with bookie performance analysis."""

    team_name = hornets_data.get("team_name", "Charlotte Hornets")
    team_games = hornets_data.get("games", pd.DataFrame())
    team_stats = hornets_data.get("stats", {})
    odds_by_bookie = hornets_data.get("odds_by_bookie", pd.DataFrame())

    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>{team_name} Weekly Odds Report - {date_str}</title>",
        "<style>",
        """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 1100px;
            margin: 40px auto;
            padding: 20px;
            background: #ffffff;
            color: #333;
        }
        h1 {
            color: #1d428a;
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 5px;
            border-bottom: 4px solid #00788c;
            padding-bottom: 10px;
        }
        h2 {
            color: #1d428a;
            font-size: 22px;
            margin-top: 40px;
            margin-bottom: 15px;
            border-left: 4px solid #00788c;
            padding-left: 12px;
        }
        .timestamp {
            color: #666;
            font-size: 14px;
            margin-bottom: 30px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }
        .stat-value {
            font-size: 28px;
            font-weight: 700;
            color: #1d428a;
        }
        .stat-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-top: 5px;
        }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            font-size: 14px;
        }
        th, td {
            border: 1px solid #e0e0e0;
            padding: 10px 12px;
            text-align: left;
        }
        th {
            background: #1d428a;
            color: white;
            font-weight: 600;
            font-size: 13px;
        }
        td {
            color: #333;
        }
        tr:nth-child(even) {
            background: #f8f9fa;
        }
        tr:hover {
            background: #e8f4f8;
        }
        .favorite {
            color: #28a745;
            font-weight: 600;
        }
        .underdog {
            color: #dc3545;
            font-weight: 600;
        }
        .best-bookie {
            background: #d4edda !important;
        }
        .worst-bookie {
            background: #f8d7da !important;
        }
        .section-note {
            font-size: 13px;
            color: #666;
            font-style: italic;
            margin-bottom: 15px;
        }
        .no-data {
            color: #999;
            font-style: italic;
            padding: 20px;
            text-align: center;
        }
        """,
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{team_name} Weekly Odds Report</h1>",
        f"<p class='timestamp'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC | Week ending: {date_str}</p>",
    ]

    # =========================================================================
    # SECTION 1: Hornets Games This Week
    # =========================================================================
    html_parts.append("<h2>Games This Week</h2>")

    if not team_games.empty:
        html_parts.append("<table>")
        html_parts.append("<tr><th>Date</th><th>Opponent</th><th>Location</th><th>Team Odds</th><th>Opponent Odds</th><th>Status</th></tr>")

        for _, row in team_games.iterrows():
            game_col = "Date of Game (ET)" if "Date of Game (ET)" in row else "Date of Game"
            date = str(row.get(game_col, ""))[:10]
            opponent = row.get("Opponent", "")
            location = row.get("Team Location", "")
            team_odds = row.get("Team Odds", 0)
            opp_odds = row.get("Opponent Odds", 0)
            is_fav = row.get("Is Favorite", False)

            status_class = "favorite" if is_fav else "underdog"
            status_text = "Favorite" if is_fav else "Underdog"

            team_odds_str = f"{team_odds:+.0f}" if team_odds >= 0 else f"{team_odds:.0f}"
            opp_odds_str = f"{opp_odds:+.0f}" if opp_odds >= 0 else f"{opp_odds:.0f}"

            html_parts.append(f"<tr>")
            html_parts.append(f"<td>{date}</td>")
            html_parts.append(f"<td>{opponent}</td>")
            html_parts.append(f"<td>{location}</td>")
            html_parts.append(f"<td>{team_odds_str}</td>")
            html_parts.append(f"<td>{opp_odds_str}</td>")
            html_parts.append(f"<td class='{status_class}'>{status_text}</td>")
            html_parts.append(f"</tr>")

        html_parts.append("</table>")
    else:
        html_parts.append("<p class='no-data'>No games found this week.</p>")

    # =========================================================================
    # SECTION 2: Hornets Odds Statistics
    # =========================================================================
    html_parts.append("<h2>Odds Statistics</h2>")

    if team_stats:
        html_parts.append("<div class='stats-grid'>")

        # Total games
        html_parts.append(f"""
        <div class='stat-card'>
            <div class='stat-value'>{team_stats.get('total_games', 0)}</div>
            <div class='stat-label'>Total Games</div>
        </div>
        """)

        # Home/Away split
        html_parts.append(f"""
        <div class='stat-card'>
            <div class='stat-value'>{team_stats.get('home_games', 0)} / {team_stats.get('away_games', 0)}</div>
            <div class='stat-label'>Home / Away</div>
        </div>
        """)

        # Favorite/Underdog split
        html_parts.append(f"""
        <div class='stat-card'>
            <div class='stat-value'>{team_stats.get('games_as_favorite', 0)} / {team_stats.get('games_as_underdog', 0)}</div>
            <div class='stat-label'>Favorite / Underdog</div>
        </div>
        """)

        # Average odds
        avg_odds = team_stats.get('avg_odds', 0)
        odds_class = 'negative' if avg_odds < 0 else 'positive'
        avg_odds_str = f"{avg_odds:+.0f}" if avg_odds >= 0 else f"{avg_odds:.0f}"
        html_parts.append(f"""
        <div class='stat-card'>
            <div class='stat-value {odds_class}'>{avg_odds_str}</div>
            <div class='stat-label'>Avg Odds</div>
        </div>
        """)

        # Avg home odds
        if 'avg_home_odds' in team_stats:
            home_odds = team_stats['avg_home_odds']
            home_class = 'negative' if home_odds < 0 else 'positive'
            home_str = f"{home_odds:+.0f}" if home_odds >= 0 else f"{home_odds:.0f}"
            html_parts.append(f"""
            <div class='stat-card'>
                <div class='stat-value {home_class}'>{home_str}</div>
                <div class='stat-label'>Avg Home Odds</div>
            </div>
            """)

        # Avg away odds
        if 'avg_away_odds' in team_stats:
            away_odds = team_stats['avg_away_odds']
            away_class = 'negative' if away_odds < 0 else 'positive'
            away_str = f"{away_odds:+.0f}" if away_odds >= 0 else f"{away_odds:.0f}"
            html_parts.append(f"""
            <div class='stat-card'>
                <div class='stat-value {away_class}'>{away_str}</div>
                <div class='stat-label'>Avg Away Odds</div>
            </div>
            """)

        html_parts.append("</div>")
    else:
        html_parts.append("<p class='no-data'>No statistics available.</p>")

    # =========================================================================
    # SECTION 3: Odds by Bookmaker for Hornets Games
    # =========================================================================
    html_parts.append("<h2>Odds by Bookmaker</h2>")
    html_parts.append("<p class='section-note'>Moneyline odds for each Hornets game from different sportsbooks</p>")

    if not odds_by_bookie.empty:
        html_parts.append("<table>")
        html_parts.append("<tr>")
        for col in odds_by_bookie.columns:
            html_parts.append(f"<th>{col}</th>")
        html_parts.append("</tr>")

        for _, row in odds_by_bookie.iterrows():
            html_parts.append("<tr>")
            for col in odds_by_bookie.columns:
                val = row[col]
                if pd.isna(val):
                    html_parts.append("<td>-</td>")
                elif isinstance(val, (int, float)) and col not in ["Date", "Opponent", "Location"]:
                    val_str = f"{val:+.0f}" if val >= 0 else f"{val:.0f}"
                    html_parts.append(f"<td>{val_str}</td>")
                else:
                    html_parts.append(f"<td>{val}</td>")
            html_parts.append("</tr>")

        html_parts.append("</table>")
    else:
        html_parts.append("<p class='no-data'>No bookmaker data available.</p>")

    # =========================================================================
    # SECTION 4: Overall Bookie Performance
    # =========================================================================
    html_parts.append("<h2>Bookmaker Performance Rankings</h2>")

    if not bookie_performance.empty:
        metric = bookie_performance["Metric"].iloc[0] if "Metric" in bookie_performance.columns else "Performance"

        if "Accuracy" in metric:
            html_parts.append("<p class='section-note'>Based on correct winner predictions. Higher accuracy is better.</p>")
            sort_col = "Accuracy %"
            best_is_high = True
        else:
            html_parts.append("<p class='section-note'>Based on deviation from market average odds. Lower variance means closer to consensus.</p>")
            sort_col = "Avg Deviation from Market"
            best_is_high = False

        html_parts.append("<table>")
        html_parts.append("<tr>")
        for col in bookie_performance.columns:
            if col != "Metric":
                html_parts.append(f"<th>{col}</th>")
        html_parts.append("</tr>")

        # Determine best and worst
        if len(bookie_performance) > 0:
            if best_is_high:
                best_idx = bookie_performance[sort_col].idxmax()
                worst_idx = bookie_performance[sort_col].idxmin()
            else:
                best_idx = bookie_performance[sort_col].idxmin()
                worst_idx = bookie_performance[sort_col].idxmax()
        else:
            best_idx = worst_idx = None

        for idx, row in bookie_performance.iterrows():
            row_class = ""
            if idx == best_idx:
                row_class = "best-bookie"
            elif idx == worst_idx:
                row_class = "worst-bookie"

            html_parts.append(f"<tr class='{row_class}'>")
            for col in bookie_performance.columns:
                if col != "Metric":
                    val = row[col]
                    if col == sort_col and best_is_high:
                        html_parts.append(f"<td><strong>{val}%</strong></td>")
                    elif col == sort_col:
                        html_parts.append(f"<td><strong>{val}</strong></td>")
                    else:
                        html_parts.append(f"<td>{val}</td>")
            html_parts.append("</tr>")

        html_parts.append("</table>")

        # Summary of best/worst
        if best_idx is not None and worst_idx is not None:
            best_bookie = bookie_performance.loc[best_idx, "Bookmaker"]
            worst_bookie = bookie_performance.loc[worst_idx, "Bookmaker"]
            html_parts.append(f"<p><strong>Best Performing:</strong> {best_bookie} (highlighted in green)</p>")
            html_parts.append(f"<p><strong>Worst Performing:</strong> {worst_bookie} (highlighted in red)</p>")
    else:
        html_parts.append("<p class='no-data'>No bookmaker performance data available.</p>")

    html_parts.extend([
        "</body>",
        "</html>"
    ])

    output_path = os.path.join(output_dir, f"daily_report_{date_str}.html")
    with open(output_path, 'w') as f:
        f.write("\n".join(html_parts))

    return output_path


# ============================================================================
# MAIN REPORT GENERATION
# ============================================================================

def generate_report(data_dir: str = "data", output_dir: str = "reports", days: int = 7,
                    team_name: str = "Charlotte Hornets") -> dict:
    """Generate Hornets-focused weekly report with bookie performance analysis."""
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  {team_name} Weekly Odds Report")
    print(f"  Week ending: {date_str}")
    print(f"{'='*60}")

    # Load actual results for accuracy tracking
    results_df = load_actual_results(data_dir)
    has_results = not results_df.empty
    if has_results:
        print(f"\nLoaded {len(results_df)} game results for accuracy tracking")
    else:
        print("\nNo actual game results found - using variance-based bookie analysis")

    # Load NBA data (this report focuses on basketball)
    print(f"\nLoading NBA data from last {days} days...")
    df = load_historical_data(data_dir, "nba", days)

    if df.empty:
        print("  No NBA data found!")
        return {"html_report": None, "error": "No data found"}

    print(f"  Loaded {len(df)} total records")

    # =========================================================================
    # HORNETS ANALYSIS
    # =========================================================================
    print(f"\n--- {team_name} Analysis ---")

    # Get all Hornets games
    hornets_games = get_team_games(df, team_name)
    print(f"  Found {len(hornets_games)} {team_name} games this week")

    # Calculate Hornets statistics
    hornets_stats = calculate_team_odds_stats(hornets_games, team_name)
    if hornets_stats:
        print(f"  Average odds: {hornets_stats.get('avg_odds', 'N/A')}")
        print(f"  Games as favorite: {hornets_stats.get('games_as_favorite', 0)}")
        print(f"  Games as underdog: {hornets_stats.get('games_as_underdog', 0)}")

    # Get odds breakdown by bookmaker for Hornets games
    hornets_odds_by_bookie = get_team_odds_by_bookie(hornets_games, team_name)
    print(f"  Odds data from {len(hornets_odds_by_bookie.columns) - 4} bookmakers")

    # =========================================================================
    # OVERALL BOOKIE PERFORMANCE
    # =========================================================================
    print(f"\n--- Bookmaker Performance Analysis ---")

    # Load season data for more comprehensive bookie analysis
    season_df = load_season_data(data_dir, "nba")
    print(f"  Analyzing {len(season_df)} season records")

    # Calculate bookie performance (accuracy if we have results, variance otherwise)
    bookie_performance = calculate_bookie_performance(season_df, results_df if has_results else None)

    if not bookie_performance.empty:
        metric_type = bookie_performance["Metric"].iloc[0] if "Metric" in bookie_performance.columns else "Performance"
        print(f"  Metric: {metric_type}")

        if "Accuracy" in metric_type:
            best = bookie_performance.loc[bookie_performance["Accuracy %"].idxmax()]
            worst = bookie_performance.loc[bookie_performance["Accuracy %"].idxmin()]
            print(f"  Best: {best['Bookmaker']} ({best['Accuracy %']}%)")
            print(f"  Worst: {worst['Bookmaker']} ({worst['Accuracy %']}%)")
        else:
            best = bookie_performance.loc[bookie_performance["Avg Deviation from Market"].idxmin()]
            worst = bookie_performance.loc[bookie_performance["Avg Deviation from Market"].idxmax()]
            print(f"  Closest to market: {best['Bookmaker']} ({best['Avg Deviation from Market']} avg deviation)")
            print(f"  Most different: {worst['Bookmaker']} ({worst['Avg Deviation from Market']} avg deviation)")

    # =========================================================================
    # GENERATE HTML REPORT
    # =========================================================================
    hornets_data = {
        "team_name": team_name,
        "games": hornets_games,
        "stats": hornets_stats,
        "odds_by_bookie": hornets_odds_by_bookie,
    }

    html_path = generate_html_report(hornets_data, bookie_performance, output_dir, date_str)
    print(f"\n{'='*60}")
    print(f"  Report generated: {html_path}")
    print(f"{'='*60}")

    return {
        "html_report": html_path,
        "hornets_data": hornets_data,
        "bookie_performance": bookie_performance,
        "date": date_str,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Hornets-focused weekly odds report")
    parser.add_argument("--days", type=int, default=7, help="Days of history to analyze")
    parser.add_argument("--output", default="reports", help="Output directory")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--team", default="Charlotte Hornets", help="Team to focus on")

    args = parser.parse_args()

    result = generate_report(
        data_dir=args.data_dir,
        output_dir=args.output,
        days=args.days,
        team_name=args.team
    )

    print(f"\nReport generation complete!")
    if result.get("html_report"):
        print(f"HTML Report: {result['html_report']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
