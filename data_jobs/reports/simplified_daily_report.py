#!/usr/bin/env python3
"""
Simplified Daily Report Generator
==================================
Generates a mobile-friendly HTML report with 4 key observations:
1. Hornets game odds & performance (next game + past month vs bookies)
2. Bookie performance rankings (league-wide)
3. ML predictions for next 48 hours (classifier + regressor)
4. Historical prediction performance tracking
"""

import os
import sys
import argparse
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
NBA_DATA_DIR = os.path.join(DATA_DIR, "nba")
TEAM_STATS_DIR = os.path.join(NBA_DATA_DIR, "team_stats")
ACTUAL_GAMES_DIR = os.path.join(NBA_DATA_DIR, "actual_games")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")
MODELS_DIR = os.path.join(DATA_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CHARTS_DIR = os.path.join(REPORTS_DIR, "charts")

for d in [PREDICTIONS_DIR, MODELS_DIR, REPORTS_DIR, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Email configuration - set these via environment variables or config file
EMAIL_CONFIG = {
    "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    "smtp_port": int(os.getenv("SMTP_PORT", "587")),
    "sender_email": os.getenv("SENDER_EMAIL", ""),
    "sender_password": os.getenv("SENDER_PASSWORD", ""),
    "recipient_email": os.getenv("RECIPIENT_EMAIL", ""),
}


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_team_stats() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all team statistics files."""
    advanced = pd.DataFrame()
    per_100_team = pd.DataFrame()
    per_100_opp = pd.DataFrame()
    
    adv_path = os.path.join(TEAM_STATS_DIR, "advanced_stats_25-26.csv")
    if os.path.exists(adv_path):
        advanced = pd.read_csv(adv_path, header=1)
        advanced = advanced[advanced['Team'].notna() & (advanced['Team'] != 'League Average')]
    
    per100_path = os.path.join(TEAM_STATS_DIR, "per_100_team_stats_25-26.csv")
    if os.path.exists(per100_path):
        per_100_team = pd.read_csv(per100_path)
        per_100_team = per_100_team[per_100_team['Team'].notna()]
    
    opp_path = os.path.join(TEAM_STATS_DIR, "per_100_opposing_team_stats_25-26")
    if os.path.exists(opp_path):
        per_100_opp = pd.read_csv(opp_path)
        per_100_opp = per_100_opp[per_100_opp['Team'].notna()]
    
    return advanced, per_100_team, per_100_opp


def load_odds_data(days: int = 30) -> pd.DataFrame:
    """Load historical odds data for the specified number of days."""
    pattern_dir = NBA_DATA_DIR
    prefix = "nba_odds_api_data_"
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


def load_actual_results() -> pd.DataFrame:
    """Load actual game results from all monthly CSV files."""
    all_results = []
    
    if os.path.exists(ACTUAL_GAMES_DIR):
        for filename in os.listdir(ACTUAL_GAMES_DIR):
            if filename.endswith(".csv"):
                filepath = os.path.join(ACTUAL_GAMES_DIR, filename)
                try:
                    df = pd.read_csv(filepath)
                    if 'Visitor/Neutral' in df.columns:
                        # Get column positions - PTS appears twice (visitor then home)
                        cols = df.columns.tolist()
                        visitor_idx = cols.index('Visitor/Neutral')
                        home_idx = cols.index('Home/Neutral')
                        
                        # Away score is column after Visitor/Neutral, Home score after Home/Neutral
                        away_score_idx = visitor_idx + 1
                        home_score_idx = home_idx + 1
                        
                        df['Away Team'] = df['Visitor/Neutral']
                        df['Home Team'] = df['Home/Neutral']
                        df['Away Score'] = pd.to_numeric(df.iloc[:, away_score_idx], errors='coerce')
                        df['Home Score'] = pd.to_numeric(df.iloc[:, home_score_idx], errors='coerce')
                        df['Game Date'] = df['Date']
                    all_results.append(df)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    continue
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        if 'Home Score' in combined.columns:
            combined = combined[combined['Home Score'].notna()]
        return combined
    return pd.DataFrame()


def load_latest_odds() -> pd.DataFrame:
    """Load the latest odds snapshot."""
    latest_path = os.path.join(NBA_DATA_DIR, "nba_odds_api_data_latest.csv")
    if os.path.exists(latest_path):
        return pd.read_csv(latest_path)
    return pd.DataFrame()


def load_predictions_history() -> pd.DataFrame:
    """Load all historical predictions."""
    all_preds = []
    if os.path.exists(PREDICTIONS_DIR):
        for filename in os.listdir(PREDICTIONS_DIR):
            if filename.startswith("predictions_") and filename.endswith(".csv"):
                filepath = os.path.join(PREDICTIONS_DIR, filename)
                try:
                    df = pd.read_csv(filepath)
                    all_preds.append(df)
                except:
                    continue
    if all_preds:
        return pd.concat(all_preds, ignore_index=True)
    return pd.DataFrame()


# ============================================================================
# HORNETS ANALYSIS
# ============================================================================

def get_hornets_next_game(odds_df: pd.DataFrame) -> Optional[Dict]:
    """Get the next upcoming Hornets game from odds data."""
    if odds_df.empty:
        return None
    
    team_name = "Charlotte Hornets"
    now = datetime.now()
    
    hornets_mask = (odds_df["Home Team"] == team_name) | (odds_df["Away Team"] == team_name)
    hornets_df = odds_df[hornets_mask].copy()
    
    if hornets_df.empty:
        return None
    
    game_col = "Date of Game (ET)" if "Date of Game (ET)" in hornets_df.columns else "Date of Game"
    hornets_df["Game DateTime"] = pd.to_datetime(hornets_df[game_col])
    
    future_games = hornets_df[hornets_df["Game DateTime"] > now]
    if future_games.empty:
        return None
    
    next_game = future_games.sort_values("Game DateTime").iloc[0]
    is_home = next_game["Home Team"] == team_name
    opponent = next_game["Away Team"] if is_home else next_game["Home Team"]
    
    return {
        "date": next_game[game_col],
        "opponent": opponent,
        "is_home": is_home,
        "location": "Home" if is_home else "Away",
        "team_h2h_odds": next_game["Avg Home H2H Odds"] if is_home else next_game["Avg Away H2H Odds"],
        "opponent_h2h_odds": next_game["Avg Away H2H Odds"] if is_home else next_game["Avg Home H2H Odds"],
    }


def calculate_hornets_performance(odds_df: pd.DataFrame, results_df: pd.DataFrame) -> Dict:
    """Calculate Hornets performance vs bookie predictions over the past month."""
    team_name = "Charlotte Hornets"
    
    if odds_df.empty or results_df.empty:
        return {}
    
    hornets_results = results_df[
        (results_df["Home Team"] == team_name) | (results_df["Away Team"] == team_name)
    ].copy()
    
    if hornets_results.empty:
        return {}
    
    hornets_odds = odds_df[
        (odds_df["Home Team"] == team_name) | (odds_df["Away Team"] == team_name)
    ].copy()
    
    if hornets_odds.empty:
        return {}
    
    game_col = "Date of Game (ET)" if "Date of Game (ET)" in hornets_odds.columns else "Date of Game"
    latest_odds = hornets_odds.sort_values("Timestamp Pulled").groupby(
        ["Home Team", "Away Team", game_col]
    ).last().reset_index()
    
    games_analyzed = 0
    wins_as_favorite = 0
    wins_as_underdog = 0
    losses_as_favorite = 0
    losses_as_underdog = 0
    
    for _, result in hornets_results.iterrows():
        home_team = result["Home Team"]
        away_team = result["Away Team"]
        
        odds_match = latest_odds[
            (latest_odds["Home Team"] == home_team) & 
            (latest_odds["Away Team"] == away_team)
        ]
        
        if odds_match.empty:
            continue
        
        odds_row = odds_match.iloc[0]
        is_home = home_team == team_name
        
        try:
            home_score = float(result["Home Score"])
            away_score = float(result["Away Score"])
        except (ValueError, TypeError):
            continue
        
        team_score = home_score if is_home else away_score
        opp_score = away_score if is_home else home_score
        team_won = team_score > opp_score
        
        team_odds = odds_row["Avg Home H2H Odds"] if is_home else odds_row["Avg Away H2H Odds"]
        opp_odds = odds_row["Avg Away H2H Odds"] if is_home else odds_row["Avg Home H2H Odds"]
        
        if pd.isna(team_odds) or pd.isna(opp_odds):
            continue
        
        is_favorite = team_odds < opp_odds
        games_analyzed += 1
        
        if team_won:
            if is_favorite:
                wins_as_favorite += 1
            else:
                wins_as_underdog += 1
        else:
            if is_favorite:
                losses_as_favorite += 1
            else:
                losses_as_underdog += 1
    
    total_wins = wins_as_favorite + wins_as_underdog
    total_losses = losses_as_favorite + losses_as_underdog
    
    return {
        "games_analyzed": games_analyzed,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "wins_as_favorite": wins_as_favorite,
        "wins_as_underdog": wins_as_underdog,
        "losses_as_favorite": losses_as_favorite,
        "losses_as_underdog": losses_as_underdog,
        "win_rate": round(total_wins / games_analyzed * 100, 1) if games_analyzed > 0 else 0,
        "favorite_win_rate": round(wins_as_favorite / (wins_as_favorite + losses_as_favorite) * 100, 1) if (wins_as_favorite + losses_as_favorite) > 0 else 0,
        "underdog_win_rate": round(wins_as_underdog / (wins_as_underdog + losses_as_underdog) * 100, 1) if (wins_as_underdog + losses_as_underdog) > 0 else 0,
    }


def calculate_league_avg_vs_bookies(odds_df: pd.DataFrame, results_df: pd.DataFrame) -> Dict:
    """Calculate league-wide average performance vs bookie predictions."""
    if odds_df.empty or results_df.empty:
        return {}
    
    game_col = "Date of Game (ET)" if "Date of Game (ET)" in odds_df.columns else "Date of Game"
    latest_odds = odds_df.sort_values("Timestamp Pulled").groupby(
        ["Home Team", "Away Team", game_col]
    ).last().reset_index()
    
    total_games = 0
    favorite_wins = 0
    
    for _, result in results_df.iterrows():
        home_team = result.get("Home Team")
        away_team = result.get("Away Team")
        
        if pd.isna(home_team) or pd.isna(away_team):
            continue
        
        odds_match = latest_odds[
            (latest_odds["Home Team"] == home_team) & 
            (latest_odds["Away Team"] == away_team)
        ]
        
        if odds_match.empty:
            continue
        
        odds_row = odds_match.iloc[0]
        
        try:
            home_score = float(result["Home Score"])
            away_score = float(result["Away Score"])
        except (ValueError, TypeError):
            continue
        
        home_odds = odds_row.get("Avg Home H2H Odds")
        away_odds = odds_row.get("Avg Away H2H Odds")
        
        if pd.isna(home_odds) or pd.isna(away_odds):
            continue
        
        home_won = home_score > away_score
        home_favorite = home_odds < away_odds
        
        total_games += 1
        if (home_won and home_favorite) or (not home_won and not home_favorite):
            favorite_wins += 1
    
    return {
        "total_games": total_games,
        "favorite_wins": favorite_wins,
        "underdog_wins": total_games - favorite_wins,
        "favorite_win_rate": round(favorite_wins / total_games * 100, 1) if total_games > 0 else 0,
    }


# ============================================================================
# BOOKIE PERFORMANCE ANALYSIS
# ============================================================================

def get_sportsbooks(df: pd.DataFrame) -> List[str]:
    """Extract list of sportsbooks from dataframe columns."""
    sportsbooks = []
    for col in df.columns:
        # Format is "Home FanDuel H2H Odds" or "Away FanDuel H2H Odds"
        if col.startswith("Home ") and col.endswith(" H2H Odds"):
            book = col.replace("Home ", "").replace(" H2H Odds", "")
            if book not in sportsbooks and book != "Avg":
                sportsbooks.append(book)
        elif col.startswith("Away ") and col.endswith(" H2H Odds"):
            book = col.replace("Away ", "").replace(" H2H Odds", "")
            if book not in sportsbooks and book != "Avg":
                sportsbooks.append(book)
    return sportsbooks


def calculate_bookie_accuracy(odds_df: pd.DataFrame, results_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """Calculate prediction accuracy for each bookmaker, plus notable calls.
    
    Returns:
        Tuple of (bookie_rankings DataFrame, list of notable game dicts)
    """
    if odds_df.empty or results_df.empty:
        return pd.DataFrame(), []
    
    sportsbooks = get_sportsbooks(odds_df)
    if not sportsbooks:
        return pd.DataFrame(), []
    
    game_col = "Date of Game (ET)" if "Date of Game (ET)" in odds_df.columns else "Date of Game"
    latest_odds = odds_df.sort_values("Timestamp Pulled").groupby(
        ["Home Team", "Away Team", game_col]
    ).last().reset_index()
    
    bookie_stats = {book: {"correct": 0, "total": 0, "deviation_sum": 0} for book in sportsbooks}
    # Track per-game bookie predictions for notable calls
    game_details = []
    
    for _, result in results_df.iterrows():
        home_team = result.get("Home Team")
        away_team = result.get("Away Team")
        
        if pd.isna(home_team) or pd.isna(away_team):
            continue
        
        odds_match = latest_odds[
            (latest_odds["Home Team"] == home_team) & 
            (latest_odds["Away Team"] == away_team)
        ]
        
        if odds_match.empty:
            continue
        
        odds_row = odds_match.iloc[0]
        
        try:
            home_score = float(result["Home Score"])
            away_score = float(result["Away Score"])
        except (ValueError, TypeError):
            continue
        
        home_won = home_score > away_score
        actual_winner = home_team if home_won else away_team
        avg_home = odds_row.get("Avg Home H2H Odds")
        avg_away = odds_row.get("Avg Away H2H Odds")
        
        if pd.isna(avg_home) or pd.isna(avg_away):
            continue
        
        # Market consensus: who did the average predict?
        market_predicted_home = avg_home < avg_away
        market_correct = market_predicted_home == home_won
        
        game_bookie_calls = {}
        
        for book in sportsbooks:
            home_col = f"Home {book} H2H Odds"
            away_col = f"Away {book} H2H Odds"
            
            if home_col not in odds_row or away_col not in odds_row:
                continue
            
            book_home = odds_row.get(home_col)
            book_away = odds_row.get(away_col)
            
            if pd.isna(book_home) or pd.isna(book_away):
                continue
            
            book_predicted_home = book_home < book_away
            book_correct = book_predicted_home == home_won
            
            bookie_stats[book]["total"] += 1
            if book_correct:
                bookie_stats[book]["correct"] += 1
            
            deviation = abs(book_home - avg_home) + abs(book_away - avg_away)
            bookie_stats[book]["deviation_sum"] += deviation
            
            game_bookie_calls[book] = {
                "predicted_home": book_predicted_home,
                "correct": book_correct,
                "home_odds": book_home,
                "away_odds": book_away,
            }
        
        if game_bookie_calls:
            correct_books = [b for b, c in game_bookie_calls.items() if c["correct"]]
            wrong_books = [b for b, c in game_bookie_calls.items() if not c["correct"]]
            total_books = len(game_bookie_calls)
            
            game_details.append({
                "home_team": home_team,
                "away_team": away_team,
                "home_score": int(home_score),
                "away_score": int(away_score),
                "actual_winner": actual_winner,
                "market_correct": market_correct,
                "correct_books": correct_books,
                "wrong_books": wrong_books,
                "num_correct": len(correct_books),
                "num_wrong": len(wrong_books),
                "total_books": total_books,
                "score_margin": abs(home_score - away_score),
                "avg_home_odds": avg_home,
                "avg_away_odds": avg_away,
            })
    
    # Build rankings
    results = []
    for book, stats in bookie_stats.items():
        if stats["total"] > 0:
            results.append({
                "Bookmaker": book,
                "Games Analyzed": stats["total"],
                "Correct Predictions": stats["correct"],
                "Accuracy %": round(stats["correct"] / stats["total"] * 100, 1),
                "Avg Deviation": round(stats["deviation_sum"] / stats["total"], 1),
            })
    
    rankings_df = pd.DataFrame()
    if results:
        rankings_df = pd.DataFrame(results)
        rankings_df = rankings_df.sort_values("Accuracy %", ascending=False).reset_index(drop=True)
    
    # Find notable calls
    notable_calls = []
    
    # 1. Biggest upsets (market was wrong, large margin)
    upsets = [g for g in game_details if not g["market_correct"]]
    upsets.sort(key=lambda x: x["score_margin"], reverse=True)
    for g in upsets[:3]:
        notable_calls.append({
            "type": "upset",
            "label": "Biggest Upset",
            "game": f"{g['away_team']} @ {g['home_team']}",
            "score": f"{g['away_score']}-{g['home_score']}",
            "winner": g["actual_winner"],
            "margin": int(g["score_margin"]),
            "detail": f"Market had {g['home_team'] if g['avg_home_odds'] < g['avg_away_odds'] else g['away_team']} winning",
        })
    
    # 2. Best unique calls (only 1-2 bookies got it right, rest were wrong)
    for g in game_details:
        if 0 < g["num_correct"] <= 2 and g["num_wrong"] >= 4:
            for book in g["correct_books"]:
                notable_calls.append({
                    "type": "unique_call",
                    "label": "Lone Correct Call",
                    "game": f"{g['away_team']} @ {g['home_team']}",
                    "score": f"{g['away_score']}-{g['home_score']}",
                    "winner": g["actual_winner"],
                    "bookie": book,
                    "detail": f"Only {len(g['correct_books'])} of {g['total_books']} bookies got it right",
                })
    
    # 3. Biggest misses (everyone was wrong, large margin upset)
    all_wrong = [g for g in game_details if g["num_correct"] == 0 and g["total_books"] > 0]
    all_wrong.sort(key=lambda x: x["score_margin"], reverse=True)
    for g in all_wrong[:3]:
        notable_calls.append({
            "type": "all_wrong",
            "label": "Everyone Wrong",
            "game": f"{g['away_team']} @ {g['home_team']}",
            "score": f"{g['away_score']}-{g['home_score']}",
            "winner": g["actual_winner"],
            "margin": int(g["score_margin"]),
            "detail": f"All {g['total_books']} bookies picked wrong",
        })
    
    return rankings_df, notable_calls


# ============================================================================
# ML PREDICTIVE MODELS
# ============================================================================

def prepare_ml_features(advanced_stats: pd.DataFrame, per_100_team: pd.DataFrame, 
                        per_100_opp: pd.DataFrame) -> pd.DataFrame:
    """Prepare feature dataframe for ML models by merging team stats."""
    if advanced_stats.empty:
        return pd.DataFrame()
    
    adv_features = ['Team', 'W', 'L', 'MOV', 'SRS', 'ORtg', 'DRtg', 'NRtg', 'Pace', 'TS%']
    adv_cols = [c for c in adv_features if c in advanced_stats.columns]
    features = advanced_stats[adv_cols].copy()
    
    if not per_100_team.empty:
        per100_features = ['Team', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']
        per100_cols = [c for c in per100_features if c in per_100_team.columns]
        if per100_cols:
            per100_subset = per_100_team[per100_cols].copy()
            per100_subset.columns = ['Team'] + [f'Off_{c}' for c in per100_cols[1:]]
            features = features.merge(per100_subset, on='Team', how='left')
    
    if not per_100_opp.empty:
        opp_features = ['Team', 'FG%', '3P%', 'PTS']
        opp_cols = [c for c in opp_features if c in per_100_opp.columns]
        if opp_cols:
            opp_subset = per_100_opp[opp_cols].copy()
            opp_subset.columns = ['Team'] + [f'Def_{c}' for c in opp_cols[1:]]
            features = features.merge(opp_subset, on='Team', how='left')
    
    return features


def build_training_data(results_df: pd.DataFrame, team_features: pd.DataFrame):
    """Build training dataset from historical results and team features."""
    from sklearn.preprocessing import StandardScaler
    
    if results_df.empty or team_features.empty:
        return pd.DataFrame(), pd.Series(), pd.Series(), pd.Series()
    
    X_data = []
    y_winner = []
    y_home_score = []
    y_away_score = []
    
    for _, game in results_df.iterrows():
        home_team = game.get("Home Team")
        away_team = game.get("Away Team")
        
        if pd.isna(home_team) or pd.isna(away_team):
            continue
        
        try:
            home_score = float(game["Home Score"])
            away_score = float(game["Away Score"])
        except (ValueError, TypeError):
            continue
        
        home_features = team_features[team_features["Team"] == home_team]
        away_features = team_features[team_features["Team"] == away_team]
        
        if home_features.empty or away_features.empty:
            continue
        
        home_row = home_features.iloc[0]
        away_row = away_features.iloc[0]
        
        feature_cols = [c for c in team_features.columns if c != 'Team']
        
        game_features = {}
        for col in feature_cols:
            try:
                home_val = float(home_row[col]) if pd.notna(home_row[col]) else 0
                away_val = float(away_row[col]) if pd.notna(away_row[col]) else 0
                game_features[f"home_{col}"] = home_val
                game_features[f"away_{col}"] = away_val
                game_features[f"diff_{col}"] = home_val - away_val
            except (ValueError, TypeError):
                continue
        
        if game_features:
            X_data.append(game_features)
            y_winner.append(1 if home_score > away_score else 0)
            y_home_score.append(home_score)
            y_away_score.append(away_score)
    
    if X_data:
        X = pd.DataFrame(X_data)
        return X, pd.Series(y_winner), pd.Series(y_home_score), pd.Series(y_away_score)
    
    return pd.DataFrame(), pd.Series(), pd.Series(), pd.Series()


def train_models(X: pd.DataFrame, y_winner: pd.Series, y_home_score: pd.Series, 
                 y_away_score: pd.Series) -> Dict:
    """Train classifier and regressor models, return best ones."""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
    
    if X.empty or len(y_winner) < 10:
        return {}
    
    X = X.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_win_train, y_win_test = train_test_split(
        X_scaled, y_winner, test_size=0.2, random_state=42
    )
    _, _, y_home_train, y_home_test = train_test_split(
        X_scaled, y_home_score, test_size=0.2, random_state=42
    )
    _, _, y_away_train, y_away_test = train_test_split(
        X_scaled, y_away_score, test_size=0.2, random_state=42
    )
    
    results = {
        "classifiers": {},
        "regressors_home": {},
        "regressors_away": {},
        "best_classifier": None,
        "best_regressor_home": None,
        "best_regressor_away": None,
        "scaler": scaler,
        "feature_names": X.columns.tolist(),
    }
    
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    
    best_clf_acc = 0
    for name, clf in classifiers.items():
        try:
            clf.fit(X_train, y_win_train)
            train_acc = accuracy_score(y_win_train, clf.predict(X_train))
            test_acc = accuracy_score(y_win_test, clf.predict(X_test))
            cv_scores = cross_val_score(clf, X_scaled, y_winner, cv=5)
            
            results["classifiers"][name] = {
                "model": clf,
                "train_accuracy": round(train_acc * 100, 1),
                "test_accuracy": round(test_acc * 100, 1),
                "cv_mean": round(cv_scores.mean() * 100, 1),
                "cv_std": round(cv_scores.std() * 100, 1),
            }
            
            if test_acc > best_clf_acc:
                best_clf_acc = test_acc
                results["best_classifier"] = name
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    regressors = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    
    best_home_mae = float('inf')
    best_away_mae = float('inf')
    
    for name, reg in regressors.items():
        try:
            reg_home = type(reg)(**reg.get_params())
            reg_home.fit(X_train, y_home_train)
            home_pred = reg_home.predict(X_test)
            home_mae = mean_absolute_error(y_home_test, home_pred)
            home_rmse = np.sqrt(mean_squared_error(y_home_test, home_pred))
            
            results["regressors_home"][name] = {
                "model": reg_home,
                "mae": round(home_mae, 1),
                "rmse": round(home_rmse, 1),
            }
            
            if home_mae < best_home_mae:
                best_home_mae = home_mae
                results["best_regressor_home"] = name
            
            reg_away = type(reg)(**reg.get_params())
            reg_away.fit(X_train, y_away_train)
            away_pred = reg_away.predict(X_test)
            away_mae = mean_absolute_error(y_away_test, away_pred)
            away_rmse = np.sqrt(mean_squared_error(y_away_test, away_pred))
            
            results["regressors_away"][name] = {
                "model": reg_away,
                "mae": round(away_mae, 1),
                "rmse": round(away_rmse, 1),
            }
            
            if away_mae < best_away_mae:
                best_away_mae = away_mae
                results["best_regressor_away"] = name
                
        except Exception as e:
            print(f"Error training {name} regressor: {e}")
    
    return results


def make_predictions(models: Dict, team_features: pd.DataFrame, 
                     upcoming_games: pd.DataFrame) -> pd.DataFrame:
    """Make predictions for upcoming games."""
    if not models or team_features.empty or upcoming_games.empty:
        return pd.DataFrame()
    
    predictions = []
    scaler = models.get("scaler")
    feature_names = models.get("feature_names", [])
    
    best_clf_name = models.get("best_classifier")
    best_home_reg_name = models.get("best_regressor_home")
    best_away_reg_name = models.get("best_regressor_away")
    
    if not best_clf_name:
        return pd.DataFrame()
    
    best_clf = models["classifiers"][best_clf_name]["model"]
    best_home_reg = models["regressors_home"].get(best_home_reg_name, {}).get("model") if best_home_reg_name else None
    best_away_reg = models["regressors_away"].get(best_away_reg_name, {}).get("model") if best_away_reg_name else None
    
    game_col = "Date of Game (ET)" if "Date of Game (ET)" in upcoming_games.columns else "Date of Game"
    
    for _, game in upcoming_games.iterrows():
        home_team = game["Home Team"]
        away_team = game["Away Team"]
        
        home_features = team_features[team_features["Team"] == home_team]
        away_features = team_features[team_features["Team"] == away_team]
        
        if home_features.empty or away_features.empty:
            continue
        
        home_row = home_features.iloc[0]
        away_row = away_features.iloc[0]
        
        feature_cols = [c for c in team_features.columns if c != 'Team']
        game_features = {}
        
        for col in feature_cols:
            try:
                home_val = float(home_row[col]) if pd.notna(home_row[col]) else 0
                away_val = float(away_row[col]) if pd.notna(away_row[col]) else 0
                game_features[f"home_{col}"] = home_val
                game_features[f"away_{col}"] = away_val
                game_features[f"diff_{col}"] = home_val - away_val
            except (ValueError, TypeError):
                continue
        
        if not game_features:
            continue
        
        X_game = pd.DataFrame([game_features])
        for col in feature_names:
            if col not in X_game.columns:
                X_game[col] = 0
        X_game = X_game[feature_names]
        X_scaled = scaler.transform(X_game)
        
        win_prob = best_clf.predict_proba(X_scaled)[0]
        predicted_winner = home_team if win_prob[1] > 0.5 else away_team
        confidence = max(win_prob) * 100
        
        pred_home_score = best_home_reg.predict(X_scaled)[0] if best_home_reg else None
        pred_away_score = best_away_reg.predict(X_scaled)[0] if best_away_reg else None
        
        predicted_spread = None
        if pred_home_score and pred_away_score:
            predicted_spread = round(pred_home_score - pred_away_score, 1)
        
        predictions.append({
            "Game Date": game[game_col],
            "Home Team": home_team,
            "Away Team": away_team,
            "Predicted Winner": predicted_winner,
            "Win Confidence %": round(confidence, 1),
            "Pred Home Score": round(pred_home_score, 0) if pred_home_score else None,
            "Pred Away Score": round(pred_away_score, 0) if pred_away_score else None,
            "Pred Spread": predicted_spread,
            "Model (Clf)": best_clf_name,
            "Model (Reg)": best_home_reg_name,
            "Prediction Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
    
    return pd.DataFrame(predictions)


def save_predictions(predictions: pd.DataFrame, date_str: str):
    """Save predictions to monthly categorized CSV files."""
    if predictions.empty:
        return
    
    month_str = date_str[:7]
    filepath = os.path.join(PREDICTIONS_DIR, f"predictions_{month_str}.csv")
    
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath)
        combined = pd.concat([existing, predictions], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["Game Date", "Home Team", "Away Team"], 
            keep="last"
        )
        combined.to_csv(filepath, index=False)
    else:
        predictions.to_csv(filepath, index=False)
    
    print(f"Predictions saved to {filepath}")


def save_models(models: Dict, date_str: str):
    """Save trained models to disk."""
    if not models:
        return
    
    filepath = os.path.join(MODELS_DIR, f"models_{date_str}.pkl")
    
    save_data = {
        "scaler": models.get("scaler"),
        "feature_names": models.get("feature_names"),
        "best_classifier": models.get("best_classifier"),
        "best_regressor_home": models.get("best_regressor_home"),
        "best_regressor_away": models.get("best_regressor_away"),
        "classifiers": {k: v["model"] for k, v in models.get("classifiers", {}).items()},
        "regressors_home": {k: v["model"] for k, v in models.get("regressors_home", {}).items()},
        "regressors_away": {k: v["model"] for k, v in models.get("regressors_away", {}).items()},
        "classifier_metrics": {k: {kk: vv for kk, vv in v.items() if kk != "model"} 
                              for k, v in models.get("classifiers", {}).items()},
        "regressor_home_metrics": {k: {kk: vv for kk, vv in v.items() if kk != "model"} 
                                   for k, v in models.get("regressors_home", {}).items()},
        "date": date_str,
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Models saved to {filepath}")


def evaluate_past_predictions(predictions_df: pd.DataFrame, results_df: pd.DataFrame) -> Dict:
    """Evaluate how past predictions performed."""
    if predictions_df.empty or results_df.empty:
        return {}
    
    total = 0
    correct = 0
    spread_errors = []
    monthly_stats = {}
    
    for _, pred in predictions_df.iterrows():
        home_team = pred.get("Home Team")
        away_team = pred.get("Away Team")
        predicted_winner = pred.get("Predicted Winner")
        pred_spread = pred.get("Pred Spread")
        
        if pd.isna(home_team) or pd.isna(away_team) or pd.isna(predicted_winner):
            continue
        
        result_match = results_df[
            (results_df["Home Team"] == home_team) & 
            (results_df["Away Team"] == away_team)
        ]
        
        if result_match.empty:
            continue
        
        result = result_match.iloc[0]
        
        try:
            home_score = float(result["Home Score"])
            away_score = float(result["Away Score"])
        except (ValueError, TypeError):
            continue
        
        actual_winner = home_team if home_score > away_score else away_team
        actual_spread = home_score - away_score
        
        total += 1
        if predicted_winner == actual_winner:
            correct += 1
        
        if pred_spread is not None:
            spread_errors.append(abs(pred_spread - actual_spread))
        
        pred_date = pred.get("Prediction Date", "")
        if pred_date:
            month = pred_date[:7]
            if month not in monthly_stats:
                monthly_stats[month] = {"total": 0, "correct": 0}
            monthly_stats[month]["total"] += 1
            if predicted_winner == actual_winner:
                monthly_stats[month]["correct"] += 1
    
    return {
        "total_evaluated": total,
        "correct_predictions": correct,
        "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
        "avg_spread_error": round(np.mean(spread_errors), 1) if spread_errors else None,
        "monthly_stats": {
            k: {**v, "accuracy": round(v["correct"] / v["total"] * 100, 1) if v["total"] > 0 else 0} 
            for k, v in monthly_stats.items()
        },
    }


# ============================================================================
# CHART GENERATION
# ============================================================================

def create_bookie_performance_chart(bookie_performance: pd.DataFrame, notable_calls: List[Dict], date_str: str) -> Optional[str]:
    """Create a detailed bookie performance visual with table + notable calls."""
    if bookie_performance.empty:
        return None
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)
    
    # --- Top left: Tabular performance heatmap ---
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('off')
    
    top_bookies = bookie_performance.head(10)
    table_data = []
    for idx, row in top_bookies.iterrows():
        table_data.append([
            f"#{idx+1}",
            row['Bookmaker'],
            f"{row['Accuracy %']}%",
            f"{row['Correct Predictions']}/{row['Games Analyzed']}",
            f"{row['Avg Deviation']:.0f}",
        ])
    
    col_labels = ['Rank', 'Bookmaker', 'Accuracy', 'Record', 'Avg Dev']
    table = ax_table.table(cellText=table_data, colLabels=col_labels,
                           cellLoc='center', loc='center', colWidths=[0.08, 0.25, 0.15, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    
    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#1d428a')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Color accuracy cells by value
    for i in range(len(table_data)):
        acc = top_bookies.iloc[i]['Accuracy %']
        if acc >= 66:
            color = '#d4edda'
        elif acc >= 63:
            color = '#fff3cd'
        else:
            color = '#f8d7da'
        table[i+1, 2].set_facecolor(color)
        # Highlight #1
        if i == 0:
            for j in range(len(col_labels)):
                if j != 2:
                    table[i+1, j].set_facecolor('#e8f5e9')
    
    ax_table.set_title('Bookmaker Performance Rankings', fontsize=14, fontweight='bold', pad=20)
    
    # --- Bottom left: Notable calls - Unique/Best calls ---
    ax_unique = fig.add_subplot(gs[1, 0])
    ax_unique.axis('off')
    
    unique_calls = [c for c in notable_calls if c['type'] == 'unique_call'][:5]
    if unique_calls:
        unique_text = ""
        for c in unique_calls:
            unique_text += f"\u2705 {c['bookie']} called {c['winner']}\n"
            unique_text += f"   {c['game']} ({c['score']})\n"
            unique_text += f"   {c['detail']}\n\n"
        ax_unique.text(0.05, 0.95, unique_text.strip(), transform=ax_unique.transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f5e9', alpha=0.8))
        ax_unique.set_title('Best Unique Calls', fontsize=12, fontweight='bold', color='#28a745')
    else:
        ax_unique.text(0.5, 0.5, 'No standout unique calls\nthis period', 
                      transform=ax_unique.transAxes, ha='center', va='center',
                      fontsize=11, color='#666')
        ax_unique.set_title('Best Unique Calls', fontsize=12, fontweight='bold', color='#28a745')
    
    # --- Bottom right: Biggest misses ---
    ax_miss = fig.add_subplot(gs[1, 1])
    ax_miss.axis('off')
    
    all_wrong = [c for c in notable_calls if c['type'] == 'all_wrong'][:3]
    upsets = [c for c in notable_calls if c['type'] == 'upset'][:3]
    misses = all_wrong if all_wrong else upsets
    
    if misses:
        miss_text = ""
        for c in misses:
            emoji = "\u274c" if c['type'] == 'all_wrong' else "\u26a0\ufe0f"
            miss_text += f"{emoji} {c['game']} ({c['score']})\n"
            miss_text += f"   Winner: {c['winner']} (+{c.get('margin', '?')} pts)\n"
            miss_text += f"   {c['detail']}\n\n"
        ax_miss.text(0.05, 0.95, miss_text.strip(), transform=ax_miss.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8d7da', alpha=0.8))
        ax_miss.set_title('Biggest Misses', fontsize=12, fontweight='bold', color='#dc3545')
    else:
        ax_miss.text(0.5, 0.5, 'No major misses\nthis period',
                    transform=ax_miss.transAxes, ha='center', va='center',
                    fontsize=11, color='#666')
        ax_miss.set_title('Biggest Misses', fontsize=12, fontweight='bold', color='#dc3545')
    
    plt.suptitle(f'Bookmaker Analysis — Past 30 Days', fontsize=16, fontweight='bold', y=0.98)
    
    output_path = os.path.join(CHARTS_DIR, f"bookie_performance_{date_str}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_hornets_performance_chart(hornets_data: Dict, results_df: pd.DataFrame, date_str: str) -> Optional[str]:
    """Create a detailed Hornets performance chart with game-by-game results."""
    perf = hornets_data.get("performance", {})
    league_avg = hornets_data.get("league_avg", {})
    
    if not perf or perf.get('games_analyzed', 0) == 0:
        return None
    
    # Get recent Hornets game results for the timeline
    team_name = "Charlotte Hornets"
    hornets_games = results_df[
        (results_df["Home Team"] == team_name) | (results_df["Away Team"] == team_name)
    ].copy()
    
    if hornets_games.empty:
        return None
    
    # Calculate per-game data
    game_data = []
    for _, row in hornets_games.iterrows():
        try:
            hs = float(row["Home Score"])
            aws = float(row["Away Score"])
        except (ValueError, TypeError):
            continue
        is_home = row["Home Team"] == team_name
        team_score = hs if is_home else aws
        opp_score = aws if is_home else hs
        opponent = row["Away Team"] if is_home else row["Home Team"]
        won = team_score > opp_score
        margin = team_score - opp_score
        game_data.append({
            "opponent": opponent,
            "team_score": int(team_score),
            "opp_score": int(opp_score),
            "won": won,
            "margin": margin,
            "is_home": is_home,
            "date": row.get("Game Date", row.get("Date", "")),
        })
    
    if not game_data:
        return None
    
    # Take last 15 games max
    game_data = game_data[-15:]
    
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.25)
    
    # --- Left: Game-by-game margin chart ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    margins = [g['margin'] for g in game_data]
    colors = ['#28a745' if m > 0 else '#dc3545' for m in margins]
    x_labels = [f"{'vs' if g['is_home'] else '@'} {g['opponent'][:3]}" for g in game_data]
    
    bars = ax1.bar(range(len(margins)), margins, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    
    # Add score labels on bars
    for i, (bar, g) in enumerate(zip(bars, game_data)):
        label = f"{g['team_score']}-{g['opp_score']}"
        y_pos = bar.get_height() if bar.get_height() > 0 else bar.get_height()
        va = 'bottom' if bar.get_height() > 0 else 'top'
        ax1.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                ha='center', va=va, fontsize=7, fontweight='bold', color='#333')
    
    ax1.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax1.axhline(y=0, color='black', linewidth=0.8)
    ax1.set_ylabel('Point Margin', fontweight='bold')
    ax1.set_title('Hornets Game-by-Game Results', fontweight='bold', fontsize=13)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add W-L record annotation
    wins = sum(1 for m in margins if m > 0)
    losses = len(margins) - wins
    ax1.text(0.02, 0.98, f"Record: {wins}-{losses}", transform=ax1.transAxes,
            fontsize=12, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', alpha=0.9))
    
    # --- Right: Summary stats ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    # Build summary text block
    total_w = perf.get('total_wins', 0)
    total_l = perf.get('total_losses', 0)
    fav_w = perf.get('wins_as_favorite', 0)
    fav_l = perf.get('losses_as_favorite', 0)
    dog_w = perf.get('wins_as_underdog', 0)
    dog_l = perf.get('losses_as_underdog', 0)
    fav_rate = perf.get('favorite_win_rate', 0)
    dog_rate = perf.get('underdog_win_rate', 0)
    league_fav = league_avg.get('favorite_win_rate', 0)
    
    avg_margin = np.mean(margins) if margins else 0
    
    summary_lines = [
        ("Overall Record", f"{total_w}-{total_l}"),
        ("Win Rate", f"{perf.get('win_rate', 0)}%"),
        ("Avg Margin", f"{avg_margin:+.1f} pts"),
        ("", ""),
        ("As Favorite", f"{fav_w}-{fav_l} ({fav_rate}%)"),
        ("As Underdog", f"{dog_w}-{dog_l} ({dog_rate}%)"),
        ("", ""),
        ("NBA Avg (Fav Win)", f"{league_fav}%"),
        ("Hornets vs Avg", f"{fav_rate - league_fav:+.1f}%"),
    ]
    
    y_start = 0.92
    for i, (label, value) in enumerate(summary_lines):
        if not label:
            y_start -= 0.04
            continue
        color = '#333'
        if 'vs Avg' in label:
            diff = fav_rate - league_fav
            color = '#28a745' if diff > 0 else '#dc3545' if diff < 0 else '#666'
        
        ax2.text(0.05, y_start - i * 0.09, label, transform=ax2.transAxes,
                fontsize=11, fontweight='bold', color='#555')
        ax2.text(0.95, y_start - i * 0.09, value, transform=ax2.transAxes,
                fontsize=12, fontweight='bold', color=color, ha='right')
    
    ax2.set_title('Performance Summary', fontweight='bold', fontsize=13)
    # Add border
    ax2.patch.set_edgecolor('#1d428a')
    ax2.patch.set_linewidth(2)
    ax2.patch.set_visible(True)
    ax2.patch.set_facecolor('#f8f9fa')
    
    plt.suptitle('Charlotte Hornets — Past Month', fontsize=16, fontweight='bold', y=1.0,
                color='#1d428a')
    
    output_path = os.path.join(CHARTS_DIR, f"hornets_performance_{date_str}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_ml_models_chart(model_results: Dict, date_str: str) -> Optional[str]:
    """Create a chart comparing ML model performance."""
    if not model_results or not model_results.get("classifiers"):
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Chart 1: Classifier accuracy
    clf_names = []
    clf_accs = []
    best_clf = model_results.get("best_classifier")
    
    for name, metrics in model_results.get("classifiers", {}).items():
        clf_names.append(name.replace(" ", "\n"))
        clf_accs.append(metrics.get("test_accuracy", 0))
    
    colors_clf = ['#28a745' if name.replace("\n", " ") == best_clf else '#1d428a' for name in clf_names]
    bars1 = ax1.bar(clf_names, clf_accs, color=colors_clf, alpha=0.8)
    
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Winner Prediction Models', fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars1, clf_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val}%', 
                ha='center', fontsize=9, fontweight='bold')
    
    # Chart 2: Regressor MAE
    reg_names = []
    reg_maes = []
    best_reg = model_results.get("best_regressor_home")
    
    for name, metrics in model_results.get("regressors_home", {}).items():
        reg_names.append(name.replace(" ", "\n"))
        reg_maes.append(metrics.get("mae", 0))
    
    colors_reg = ['#28a745' if name.replace("\n", " ") == best_reg else '#1d428a' for name in reg_names]
    bars2 = ax2.bar(reg_names, reg_maes, color=colors_reg, alpha=0.8)
    
    ax2.set_ylabel('Mean Absolute Error (points)', fontweight='bold')
    ax2.set_title('Score Prediction Models', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, reg_maes):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val}', 
                ha='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('ML Model Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(CHARTS_DIR, f"ml_models_{date_str}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


# ============================================================================
# EMAIL SENDING
# ============================================================================

def send_email_report(html_content: str, chart_paths: List[str], date_str: str) -> bool:
    """Send the daily report via email with chart attachments."""
    
    sender = EMAIL_CONFIG["sender_email"]
    recipient = EMAIL_CONFIG["recipient_email"]
    password = EMAIL_CONFIG["sender_password"]
    
    if not sender or not recipient or not password:
        print("\n⚠️  Email not configured. Set environment variables:")
        print("   - SENDER_EMAIL")
        print("   - SENDER_PASSWORD")
        print("   - RECIPIENT_EMAIL")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart('related')
        msg['Subject'] = f'🏀 NBA Daily Report - {date_str}'
        msg['From'] = sender
        msg['To'] = recipient
        
        # Attach HTML body
        msg_alternative = MIMEMultipart('alternative')
        msg.attach(msg_alternative)
        
        html_part = MIMEText(html_content, 'html')
        msg_alternative.attach(html_part)
        
        # Attach charts as inline images and as attachments
        for i, chart_path in enumerate(chart_paths):
            if chart_path and os.path.exists(chart_path):
                with open(chart_path, 'rb') as f:
                    img_data = f.read()
                
                # Inline image
                img = MIMEImage(img_data)
                img.add_header('Content-ID', f'<chart{i}>')
                img.add_header('Content-Disposition', 'inline', filename=os.path.basename(chart_path))
                msg.attach(img)
                
                # Also attach as downloadable
                attachment = MIMEImage(img_data)
                attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(chart_path))
                msg.attach(attachment)
        
        # Send email
        print(f"\n📧 Sending email to {recipient}...")
        
        with smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        
        print("✅ Email sent successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return False


# ============================================================================
# HTML REPORT GENERATION
# ============================================================================

def generate_html_report(hornets_data: Dict, bookie_performance: pd.DataFrame,
                         model_results: Dict, predictions: pd.DataFrame,
                         prediction_eval: Dict, date_str: str,
                         notable_calls: List[Dict] = None) -> str:
    """Generate mobile-friendly HTML report."""
    if notable_calls is None:
        notable_calls = []
    
    next_game = hornets_data.get("next_game")
    perf = hornets_data.get("performance", {})
    league_avg = hornets_data.get("league_avg", {})
    
    # Calculate vs league comparison
    hornets_fav_rate = perf.get("favorite_win_rate", 0)
    league_fav_rate = league_avg.get("favorite_win_rate", 0)
    vs_league = hornets_fav_rate - league_fav_rate
    vs_league_class = "positive" if vs_league > 0 else "negative" if vs_league < 0 else "neutral"
    
    # Build next game section
    next_game_html = ""
    if next_game:
        location_emoji = "🏠" if next_game["is_home"] else "✈️"
        odds_val = next_game.get('team_h2h_odds', 0)
        odds_str = f"+{odds_val:.0f}" if odds_val >= 0 else f"{odds_val:.0f}"
        next_game_html = f'''
        <div class="game-card">
            <div class="date">{location_emoji} {next_game['location']} - {next_game['date']}</div>
            <div class="teams">Hornets vs {next_game['opponent']}</div>
            <div class="odds">Current Odds: <strong>{odds_str}</strong></div>
        </div>'''
    else:
        next_game_html = '<div class="card"><p>No upcoming Hornets game found.</p></div>'
    
    # Build performance section
    perf_html = ""
    if perf:
        perf_html = f'''
        <div class="card">
            <h3>Past Month Performance</h3>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-value">{perf.get('total_wins', 0)}-{perf.get('total_losses', 0)}</div>
                    <div class="stat-label">Record</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{perf.get('win_rate', 0)}%</div>
                    <div class="stat-label">Win Rate</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{perf.get('wins_as_underdog', 0)}</div>
                    <div class="stat-label">Upset Wins</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{perf.get('losses_as_favorite', 0)}</div>
                    <div class="stat-label">Upset Losses</div>
                </div>
            </div>
            <h3>vs Bookie Predictions</h3>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-value">{perf.get('favorite_win_rate', 0)}%</div>
                    <div class="stat-label">Win % as Favorite</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{perf.get('underdog_win_rate', 0)}%</div>
                    <div class="stat-label">Win % as Underdog</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value {vs_league_class}">{vs_league:+.1f}%</div>
                    <div class="stat-label">vs NBA Avg</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{league_avg.get('favorite_win_rate', 0)}%</div>
                    <div class="stat-label">NBA Fav Win Rate</div>
                </div>
            </div>
        </div>'''
    
    # Build bookie table
    bookie_html = ""
    if not bookie_performance.empty:
        rows = ""
        for idx, row in bookie_performance.iterrows():
            hl = 'class="highlight"' if idx == 0 else ""
            rows += f'''<tr {hl}>
                <td>{idx + 1}</td>
                <td>{row['Bookmaker']}</td>
                <td><strong>{row['Accuracy %']}%</strong></td>
                <td>{row['Games Analyzed']}</td>
                <td>{row['Avg Deviation']}</td>
            </tr>'''
        # Build notable calls HTML
        notable_html = ""
        unique_calls = [c for c in notable_calls if c['type'] == 'unique_call'][:5]
        all_wrong_calls = [c for c in notable_calls if c['type'] == 'all_wrong'][:3]
        upset_calls = [c for c in notable_calls if c['type'] == 'upset'][:3]
        
        if unique_calls:
            unique_items = ""
            for c in unique_calls:
                unique_items += f'<div class="notable-item good"><strong>{c["bookie"]}</strong> correctly called {c["winner"]}<br><span style="font-size:11px;">{c["game"]} ({c["score"]}) — {c["detail"]}</span></div>'
            notable_html += f'<h3>Best Unique Calls</h3>{unique_items}'
        
        miss_calls = all_wrong_calls if all_wrong_calls else upset_calls
        if miss_calls:
            miss_items = ""
            for c in miss_calls:
                miss_items += f'<div class="notable-item bad"><strong>{c["game"]}</strong> ({c["score"]})<br><span style="font-size:11px;">{c["winner"]} won by {c.get("margin", "?")} pts — {c["detail"]}</span></div>'
            notable_html += f'<h3>Biggest Misses</h3>{miss_items}'
        
        bookie_html = f'''
        <div class="card">
            <table>
                <tr><th>Rank</th><th>Bookmaker</th><th>Accuracy</th><th>Games</th><th>Deviation</th></tr>
                {rows}
            </table>
            <p style="font-size:11px;color:#666;margin-top:10px;">* Accuracy = correct winner predictions</p>
        </div>
        {f'<div class="card">{notable_html}</div>' if notable_html else ''}'''
    else:
        bookie_html = '<div class="card"><p>No bookmaker data available.</p></div>'
    
    # Build model performance section
    model_html = ""
    if model_results and model_results.get("classifiers"):
        clf_rows = ""
        best_clf = model_results.get("best_classifier")
        for name, m in model_results.get("classifiers", {}).items():
            hl = 'class="highlight"' if name == best_clf else ""
            clf_rows += f'''<tr {hl}>
                <td>{name}</td>
                <td>{m.get('test_accuracy', 0)}%</td>
                <td>{m.get('cv_mean', 0)}%</td>
                <td>±{m.get('cv_std', 0)}%</td>
            </tr>'''
        
        reg_rows = ""
        best_reg = model_results.get("best_regressor_home")
        for name, m in model_results.get("regressors_home", {}).items():
            hl = 'class="highlight"' if name == best_reg else ""
            reg_rows += f'''<tr {hl}>
                <td>{name}</td>
                <td>{m.get('mae', 0)} pts</td>
                <td>{m.get('rmse', 0)} pts</td>
            </tr>'''
        
        model_html = f'''
        <div class="card">
            <h3>Model Performance</h3>
            <p style="font-size:12px;margin-bottom:10px;"><strong>Winner Prediction:</strong></p>
            <table>
                <tr><th>Model</th><th>Test Acc</th><th>CV Mean</th><th>CV Std</th></tr>
                {clf_rows}
            </table>
            <p style="font-size:12px;margin:15px 0 10px 0;"><strong>Score Prediction:</strong></p>
            <table>
                <tr><th>Model</th><th>MAE</th><th>RMSE</th></tr>
                {reg_rows}
            </table>
        </div>'''
    
    # Build predictions section
    pred_html = ""
    if not predictions.empty:
        pred_cards = ""
        for _, p in predictions.iterrows():
            winner = p.get("Predicted Winner", "")
            conf = p.get("Win Confidence %", 0)
            home = p.get("Home Team", "")
            away = p.get("Away Team", "")
            ph = p.get("Pred Home Score")
            pa = p.get("Pred Away Score")
            spread = p.get("Pred Spread")
            
            score_str = f"Predicted: {int(pa)} - {int(ph)}" if ph and pa else ""
            spread_str = f"Spread: {spread:+.1f}" if spread else ""
            
            pred_cards += f'''
            <div class="prediction-card">
                <div class="matchup">{away} @ {home}</div>
                <div class="details">
                    <span class="winner">🏆 {winner}</span> ({conf:.0f}%)
                    {f'<br>{score_str}' if score_str else ''}
                    {f' | {spread_str}' if spread_str else ''}
                </div>
            </div>'''
        pred_html = f'<div class="card"><h3>Game Predictions</h3>{pred_cards}</div>'
    else:
        pred_html = '<div class="card"><p>No upcoming games for predictions.</p></div>'
    
    # Build history section
    hist_html = ""
    if prediction_eval and prediction_eval.get("total_evaluated", 0) > 0:
        monthly_rows = ""
        for month, stats in sorted(prediction_eval.get("monthly_stats", {}).items(), reverse=True):
            monthly_rows += f'''<tr>
                <td>{month}</td>
                <td>{stats.get('correct', 0)}</td>
                <td>{stats.get('total', 0)}</td>
                <td>{stats.get('accuracy', 0)}%</td>
            </tr>'''
        
        hist_html = f'''
        <div class="card">
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-value">{prediction_eval.get('accuracy', 0)}%</div>
                    <div class="stat-label">Overall Accuracy</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{prediction_eval.get('correct_predictions', 0)}/{prediction_eval.get('total_evaluated', 0)}</div>
                    <div class="stat-label">Correct/Total</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{prediction_eval.get('avg_spread_error', 'N/A')}</div>
                    <div class="stat-label">Avg Spread Error</div>
                </div>
            </div>
            <h3>Monthly Breakdown</h3>
            <table>
                <tr><th>Month</th><th>Correct</th><th>Total</th><th>Accuracy</th></tr>
                {monthly_rows}
            </table>
        </div>'''
    else:
        hist_html = '<div class="card"><p>No historical predictions to evaluate yet.</p></div>'
    
    # Assemble full HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Daily Report - {date_str}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; padding: 10px; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        h1 {{ color: #1d428a; font-size: 24px; text-align: center; padding: 15px 0; border-bottom: 3px solid #00788c; margin-bottom: 20px; }}
        h2 {{ color: white; font-size: 18px; margin: 25px 0 15px 0; padding: 10px; background: linear-gradient(90deg, #1d428a 0%, #00788c 100%); border-radius: 5px; }}
        h3 {{ color: #1d428a; font-size: 16px; margin: 15px 0 10px 0; border-left: 4px solid #00788c; padding-left: 10px; }}
        .card {{ background: white; border-radius: 10px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 10px 0; }}
        .stat-box {{ background: #f8f9fa; border-radius: 8px; padding: 12px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: 700; color: #1d428a; }}
        .stat-label {{ font-size: 11px; color: #666; text-transform: uppercase; margin-top: 5px; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        .neutral {{ color: #6c757d; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 10px 0; }}
        th, td {{ padding: 8px 6px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
        th {{ background: #1d428a; color: white; font-weight: 600; font-size: 12px; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        .highlight {{ background: #d4edda !important; }}
        .game-card {{ background: linear-gradient(135deg, #1d428a 0%, #00788c 100%); color: white; border-radius: 10px; padding: 20px; text-align: center; margin-bottom: 15px; }}
        .game-card .teams {{ font-size: 20px; font-weight: 700; margin-bottom: 10px; }}
        .game-card .date {{ font-size: 14px; opacity: 0.9; }}
        .game-card .odds {{ font-size: 16px; margin-top: 10px; }}
        .prediction-card {{ background: white; border-radius: 8px; padding: 12px; margin-bottom: 10px; border-left: 4px solid #1d428a; }}
        .prediction-card .matchup {{ font-weight: 600; font-size: 14px; }}
        .prediction-card .details {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .prediction-card .winner {{ color: #28a745; font-weight: 700; }}
        .notable-item {{ border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; font-size: 13px; line-height: 1.5; }}
        .notable-item.good {{ background: #d4edda; border-left: 4px solid #28a745; }}
        .notable-item.bad {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        .timestamp {{ text-align: center; color: #666; font-size: 12px; margin-bottom: 20px; }}
        @media (max-width: 480px) {{ .stat-grid {{ grid-template-columns: repeat(2, 1fr); }} table {{ font-size: 11px; }} th, td {{ padding: 6px 4px; }} }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏀 NBA Daily Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Data as of: {date_str}</p>
        
        <h2>🐝 Charlotte Hornets</h2>
        {next_game_html}
        {perf_html}
        
        <h2>📊 Bookmaker Rankings</h2>
        {bookie_html}
        
        <h2>🤖 ML Predictions (Next 48 Hours)</h2>
        {model_html}
        {pred_html}
        
        <h2>📈 Prediction History</h2>
        {hist_html}
    </div>
</body>
</html>'''
    
    return html


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_report(days: int = 30) -> Dict:
    """Generate the simplified daily report with all 4 observations."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\n{'='*60}")
    print(f"  NBA Daily Report Generator")
    print(f"  Date: {date_str}")
    print(f"{'='*60}")
    
    # Load all data
    print("\n[1/6] Loading data...")
    odds_df = load_odds_data(days)
    results_df = load_actual_results()
    advanced_stats, per_100_team, per_100_opp = load_team_stats()
    latest_odds = load_latest_odds()
    predictions_history = load_predictions_history()
    
    print(f"  - Odds records: {len(odds_df)}")
    print(f"  - Game results: {len(results_df)}")
    print(f"  - Team stats loaded: {len(advanced_stats)} teams")
    
    # Hornets Analysis
    print("\n[2/6] Analyzing Hornets performance...")
    next_game = get_hornets_next_game(latest_odds if not latest_odds.empty else odds_df)
    hornets_perf = calculate_hornets_performance(odds_df, results_df)
    league_avg = calculate_league_avg_vs_bookies(odds_df, results_df)
    
    hornets_data = {
        "next_game": next_game,
        "performance": hornets_perf,
        "league_avg": league_avg,
    }
    
    if next_game:
        print(f"  - Next game: vs {next_game['opponent']} ({next_game['location']})")
    if hornets_perf:
        print(f"  - Past month: {hornets_perf.get('total_wins', 0)}-{hornets_perf.get('total_losses', 0)}")
    
    # Bookie Performance
    print("\n[3/6] Calculating bookie performance...")
    bookie_performance, notable_calls = calculate_bookie_accuracy(odds_df, results_df)
    if not bookie_performance.empty:
        best_bookie = bookie_performance.iloc[0]
        print(f"  - Best bookie: {best_bookie['Bookmaker']} ({best_bookie['Accuracy %']}%)")
    if notable_calls:
        print(f"  - Notable calls found: {len(notable_calls)}")
    
    # ML Models
    print("\n[4/6] Training ML models...")
    team_features = prepare_ml_features(advanced_stats, per_100_team, per_100_opp)
    X, y_winner, y_home, y_away = build_training_data(results_df, team_features)
    
    model_results = {}
    if not X.empty:
        print(f"  - Training samples: {len(X)}")
        model_results = train_models(X, y_winner, y_home, y_away)
        
        if model_results.get("best_classifier"):
            best_clf = model_results["best_classifier"]
            clf_acc = model_results["classifiers"][best_clf]["test_accuracy"]
            print(f"  - Best classifier: {best_clf} ({clf_acc}%)")
        
        if model_results.get("best_regressor_home"):
            best_reg = model_results["best_regressor_home"]
            reg_mae = model_results["regressors_home"][best_reg]["mae"]
            print(f"  - Best regressor: {best_reg} (MAE: {reg_mae})")
        
        # Save models
        save_models(model_results, date_str)
    
    # Make Predictions
    print("\n[5/6] Making predictions for upcoming games...")
    predictions = pd.DataFrame()
    
    # Get upcoming games from latest odds
    if not latest_odds.empty and model_results:
        game_col = "Date of Game (ET)" if "Date of Game (ET)" in latest_odds.columns else "Date of Game"
        latest_odds["Game DateTime"] = pd.to_datetime(latest_odds[game_col])
        now = datetime.now()
        cutoff = now + timedelta(hours=48)
        
        upcoming = latest_odds[
            (latest_odds["Game DateTime"] > now) & 
            (latest_odds["Game DateTime"] <= cutoff)
        ].drop_duplicates(subset=["Home Team", "Away Team"])
        
        if not upcoming.empty:
            predictions = make_predictions(model_results, team_features, upcoming)
            print(f"  - Predictions made: {len(predictions)}")
            
            # Save predictions
            if not predictions.empty:
                save_predictions(predictions, date_str)
    
    # Evaluate Past Predictions
    print("\n[6/6] Evaluating historical predictions...")
    prediction_eval = evaluate_past_predictions(predictions_history, results_df)
    if prediction_eval.get("total_evaluated", 0) > 0:
        print(f"  - Historical accuracy: {prediction_eval['accuracy']}% ({prediction_eval['correct_predictions']}/{prediction_eval['total_evaluated']})")
    else:
        print("  - No historical predictions to evaluate yet")
    
    # Generate HTML Report
    print("\n" + "="*60)
    print("Generating HTML report and charts...")
    
    html_content = generate_html_report(
        hornets_data, bookie_performance, model_results,
        predictions, prediction_eval, date_str,
        notable_calls=notable_calls
    )
    
    output_path = os.path.join(REPORTS_DIR, f"daily_report_{date_str}.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Report saved to: {output_path}")
    
    # Generate Charts
    print("\nGenerating visual charts...")
    chart_paths = []
    
    chart1 = create_hornets_performance_chart(hornets_data, results_df, date_str)
    if chart1:
        chart_paths.append(chart1)
        print(f"  ✓ Hornets performance chart")
    
    chart2 = create_bookie_performance_chart(bookie_performance, notable_calls, date_str)
    if chart2:
        chart_paths.append(chart2)
        print(f"  ✓ Bookie performance chart")
    
    chart3 = create_ml_models_chart(model_results, date_str)
    if chart3:
        chart_paths.append(chart3)
        print(f"  ✓ ML models chart")
    
    # Send Email
    email_sent = send_email_report(html_content, chart_paths, date_str)
    
    print("="*60)
    
    return {
        "report_path": output_path,
        "chart_paths": chart_paths,
        "email_sent": email_sent,
        "hornets_data": hornets_data,
        "bookie_performance": bookie_performance,
        "model_results": model_results,
        "predictions": predictions,
        "prediction_eval": prediction_eval,
        "date": date_str,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate simplified NBA daily report with email delivery")
    parser.add_argument("--days", type=int, default=30, help="Days of history to analyze")
    parser.add_argument("--no-email", action="store_true", help="Skip sending email (just generate report)")
    parser.add_argument("--email", help="Override recipient email address")
    args = parser.parse_args()
    
    # Override email if provided
    if args.email:
        EMAIL_CONFIG["recipient_email"] = args.email
    
    # Disable email if requested
    if args.no_email:
        EMAIL_CONFIG["recipient_email"] = ""
    
    result = generate_report(days=args.days)
    
    print(f"\n✅ Report generation complete!")
    print(f"   HTML Report: {result['report_path']}")
    if result.get('chart_paths'):
        print(f"   Charts: {len(result['chart_paths'])} generated")
    if result.get('email_sent'):
        print(f"   📧 Email delivered successfully!")


if __name__ == "__main__":
    main()
