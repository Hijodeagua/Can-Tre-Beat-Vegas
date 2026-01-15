#!/usr/bin/env python3
"""
Daily Odds Report Generator
Generates visualizations based on daily odds and week-to-week changes

Usage:
    python -m data_jobs.reports.generate_daily_report [--days 7] [--output reports/]
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CI use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_historical_data(data_dir: str, sport: str, days: int = 7) -> pd.DataFrame:
    """
    Load historical odds data for a sport.

    Args:
        data_dir: Base data directory
        sport: 'nfl' or 'nba'
        days: Number of days of history to load

    Returns:
        Combined DataFrame with all historical data
    """
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
                # Extract date from filename
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
        # Parse timestamp
        if "Timestamp Pulled" in combined.columns:
            combined["Timestamp Pulled"] = pd.to_datetime(combined["Timestamp Pulled"])
        return combined.sort_values("Timestamp Pulled")

    return pd.DataFrame()


def calculate_odds_movement(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate odds movement for each game/team over time"""
    if df.empty:
        return df

    # Group by game (home/away teams and game date)
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
    """Safely subtract two values that might be None"""
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    return round(a - b, 2)


def plot_team_odds_history(df: pd.DataFrame, team: str, output_dir: str, sport: str) -> Optional[str]:
    """
    Plot odds history for a specific team.

    Returns:
        Path to saved figure or None
    """
    if df.empty:
        return None

    # Filter for games involving this team
    team_games = df[(df["Home Team"] == team) | (df["Away Team"] == team)].copy()
    if team_games.empty:
        return None

    # Get H2H odds for team (home or away depending on matchup)
    team_games["Team Odds"] = team_games.apply(
        lambda r: r["Avg Home H2H Odds"] if r["Home Team"] == team else r["Avg Away H2H Odds"],
        axis=1
    )
    team_games["Is Home"] = team_games["Home Team"] == team

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot odds over time
    ax.plot(team_games["Timestamp Pulled"], team_games["Team Odds"],
            marker='o', linewidth=2, markersize=4)

    ax.set_xlabel("Date/Time")
    ax.set_ylabel("Moneyline Odds (American)")
    ax.set_title(f"{team} - Moneyline Odds History ({sport.upper()})")

    # Add reference line at +/- 100
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Even odds')
    ax.axhline(y=-100, color='gray', linestyle='--', alpha=0.5)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax.xaxis.set_major_locator(MaxNLocator(8))
    plt.xticks(rotation=45)

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save figure
    safe_team = team.replace(" ", "_").lower()
    output_path = os.path.join(output_dir, f"{sport}_{safe_team}_odds_history.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def plot_daily_summary(df: pd.DataFrame, output_dir: str, sport: str) -> Optional[str]:
    """
    Create a summary visualization of all games for the day.

    Returns:
        Path to saved figure or None
    """
    if df.empty:
        return None

    # Get latest snapshot
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
    """
    Plot odds movement for upcoming games.

    Returns:
        Path to saved figure or None
    """
    if movements_df.empty:
        return None

    # Filter to games with meaningful movement
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

    # Add value labels
    for bar, val in zip(bars, h2h_movement):
        if val != 0:
            ax.text(val + (2 if val >= 0 else -2), bar.get_y() + bar.get_height()/2,
                    f'{val:+.0f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{sport}_odds_movement.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def plot_sportsbook_comparison(df: pd.DataFrame, output_dir: str, sport: str) -> Optional[str]:
    """
    Compare odds across different sportsbooks.

    Returns:
        Path to saved figure or None
    """
    if df.empty:
        return None

    # Get latest data
    latest = df.sort_values("Timestamp Pulled").groupby(["Home Team", "Away Team"]).last().reset_index()

    # Find sportsbook columns
    sportsbooks = []
    for col in latest.columns:
        if "H2H Odds" in col and col.startswith("Home "):
            book = col.replace("Home ", "").replace(" H2H Odds", "")
            if book not in ["Avg"]:
                sportsbooks.append(book)

    if not sportsbooks or latest.empty:
        return None

    # Pick first game for comparison
    game = latest.iloc[0]
    home_team = game["Home Team"]
    away_team = game["Away Team"]

    home_odds = []
    away_odds = []
    valid_books = []

    for book in sportsbooks:
        h = game.get(f"Home {book} H2H Odds")
        a = game.get(f"Away {book} H2H Odds")
        if pd.notna(h) and pd.notna(a):
            home_odds.append(h)
            away_odds.append(a)
            valid_books.append(book)

    if not valid_books:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(valid_books))
    width = 0.35

    ax.bar([i - width/2 for i in x], home_odds, width, label=f'{home_team} (Home)', alpha=0.8, color='#2ecc71')
    ax.bar([i + width/2 for i in x], away_odds, width, label=f'{away_team} (Away)', alpha=0.8, color='#e74c3c')

    ax.set_xlabel('Sportsbook')
    ax.set_ylabel('Moneyline Odds')
    ax.set_title(f'{sport.upper()} Sportsbook Comparison: {away_team} @ {home_team}')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_books, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{sport}_sportsbook_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def generate_html_report(figures: dict, output_dir: str, date_str: str) -> str:
    """Generate an HTML report combining all visualizations"""

    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>Daily Odds Report - {date_str}</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }",
        "h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }",
        "h2 { color: #34495e; margin-top: 30px; }",
        ".sport-section { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "img { max-width: 100%; height: auto; margin: 10px 0; border-radius: 4px; }",
        ".timestamp { color: #7f8c8d; font-size: 0.9em; }",
        ".no-data { color: #95a5a6; font-style: italic; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Daily Odds Report</h1>",
        f"<p class='timestamp'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>",
    ]

    for sport, sport_figures in figures.items():
        html_parts.append(f"<div class='sport-section'>")
        html_parts.append(f"<h2>{sport.upper()} Odds Analysis</h2>")

        if not sport_figures:
            html_parts.append("<p class='no-data'>No data available for this sport.</p>")
        else:
            for fig_type, fig_path in sport_figures.items():
                if fig_path and os.path.exists(fig_path):
                    rel_path = os.path.basename(fig_path)
                    html_parts.append(f"<h3>{fig_type.replace('_', ' ').title()}</h3>")
                    html_parts.append(f"<img src='{rel_path}' alt='{fig_type}'>")

        html_parts.append("</div>")

    html_parts.extend([
        "</body>",
        "</html>"
    ])

    output_path = os.path.join(output_dir, f"daily_report_{date_str}.html")
    with open(output_path, 'w') as f:
        f.write("\n".join(html_parts))

    return output_path


def generate_report(data_dir: str = "data", output_dir: str = "reports", days: int = 7) -> dict:
    """
    Generate complete daily report with visualizations.

    Args:
        data_dir: Directory containing odds data
        output_dir: Directory for output reports
        days: Number of days of history to analyze

    Returns:
        Dictionary with report paths
    """
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")

    figures = {}

    for sport in ["nba", "nfl"]:
        print(f"\nProcessing {sport.upper()} data...")
        sport_figures = {}

        # Load data
        df = load_historical_data(data_dir, sport, days)

        if df.empty:
            print(f"  No {sport.upper()} data found")
            figures[sport] = {}
            continue

        print(f"  Loaded {len(df)} records")

        # Generate visualizations
        summary_path = plot_daily_summary(df, output_dir, sport)
        if summary_path:
            sport_figures["daily_summary"] = summary_path
            print(f"  Created daily summary")

        movements = calculate_odds_movement(df)
        if not movements.empty:
            movement_path = plot_odds_movement(movements, output_dir, sport)
            if movement_path:
                sport_figures["odds_movement"] = movement_path
                print(f"  Created odds movement chart")

        sportsbook_path = plot_sportsbook_comparison(df, output_dir, sport)
        if sportsbook_path:
            sport_figures["sportsbook_comparison"] = sportsbook_path
            print(f"  Created sportsbook comparison")

        # Get top teams by game frequency for individual charts
        teams = pd.concat([df["Home Team"], df["Away Team"]]).value_counts().head(3).index.tolist()
        for team in teams:
            team_path = plot_team_odds_history(df, team, output_dir, sport)
            if team_path:
                safe_team = team.replace(" ", "_").lower()
                sport_figures[f"team_{safe_team}"] = team_path

        figures[sport] = sport_figures

    # Generate HTML report
    html_path = generate_html_report(figures, output_dir, date_str)
    print(f"\nGenerated HTML report: {html_path}")

    return {
        "html_report": html_path,
        "figures": figures,
        "date": date_str,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate daily odds visualization report")
    parser.add_argument("--days", type=int, default=7, help="Days of history to analyze")
    parser.add_argument("--output", default="reports", help="Output directory")
    parser.add_argument("--data-dir", default="data", help="Data directory")

    args = parser.parse_args()

    result = generate_report(
        data_dir=args.data_dir,
        output_dir=args.output,
        days=args.days
    )

    print(f"\nReport generation complete!")
    print(f"HTML Report: {result['html_report']}")


if __name__ == "__main__":
    main()
