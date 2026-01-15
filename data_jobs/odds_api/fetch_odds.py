#!/usr/bin/env python3
"""
Unified Odds Fetcher - Fetches odds for all supported sports
Optimized for The Odds API free tier (500 requests/month)

Usage:
    python -m data_jobs.odds_api.fetch_odds [--sport nfl|nba|all] [--report]

Environment Variables:
    ODDS_API_KEY: Required API key for The Odds API
"""

import argparse
import sys
import os

# Add parent directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_jobs.odds_api.client import OddsAPIClient
from data_jobs.odds_api.processors import process_game_data, save_odds_data
from data_jobs.odds_api.config import SUPPORTED_SPORTS


def fetch_single_sport(client: OddsAPIClient, sport: str, base_dir: str = ".") -> dict:
    """
    Fetch and save odds for a single sport.

    Returns:
        Dict with status information
    """
    result = {
        "sport": sport,
        "success": False,
        "games_count": 0,
        "error": None,
        "files": [],
    }

    try:
        games = client.fetch_odds(sport)
        result["games_count"] = len(games)

        if games:
            df = process_game_data(games, sport)
            timestamped, latest = save_odds_data(df, sport, base_dir)
            result["files"] = [timestamped, latest]

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        print(f"Error fetching {sport}: {e}")

    return result


def fetch_all_sports(client: OddsAPIClient, base_dir: str = ".") -> list:
    """
    Fetch and save odds for all supported sports.

    Returns:
        List of result dicts for each sport
    """
    results = []

    for sport in SUPPORTED_SPORTS:
        print(f"\n{'='*50}")
        print(f"Fetching {sport.upper()} odds...")
        print(f"{'='*50}")

        result = fetch_single_sport(client, sport, base_dir)
        results.append(result)

    return results


def print_summary(results: list, client: OddsAPIClient):
    """Print a summary of the fetch operation"""
    print("\n" + "="*60)
    print("FETCH SUMMARY")
    print("="*60)

    total_games = 0
    for result in results:
        status = "OK" if result["success"] else "FAILED"
        print(f"{result['sport'].upper():8} [{status}] - {result['games_count']} games")
        if result["error"]:
            print(f"         Error: {result['error']}")
        total_games += result["games_count"]

    print(f"\nTotal games fetched: {total_games}")

    # Print quota status
    status = client.get_quota_status()
    if status["requests_remaining"] is not None:
        print(f"\nAPI Quota: {status['requests_remaining']} / {status['monthly_limit']} requests remaining")

        # Estimate daily budget
        remaining = status["requests_remaining"]
        # Assume we're mid-month, so ~15 days remaining on average
        daily_budget = remaining // 15 if remaining > 0 else 0
        print(f"Estimated daily budget: ~{daily_budget} requests")

    warning = client.check_quota_warning()
    if warning:
        print(f"\n{warning}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch sports betting odds from The Odds API"
    )
    parser.add_argument(
        "--sport",
        choices=["nfl", "nba", "all"],
        default="all",
        help="Sport to fetch (default: all)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Show API usage report",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory for output files (default: current directory)",
    )

    args = parser.parse_args()

    try:
        client = OddsAPIClient()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.report:
        print(client.get_usage_report())
        return

    if args.sport == "all":
        results = fetch_all_sports(client, args.base_dir)
    else:
        results = [fetch_single_sport(client, args.sport, args.base_dir)]

    print_summary(results, client)


if __name__ == "__main__":
    main()
