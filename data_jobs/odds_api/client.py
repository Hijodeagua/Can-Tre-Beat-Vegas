"""
Odds API Client with rate limiting and usage tracking
Optimized for free tier (500 requests/month)
"""

import os
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import (
    BASE_URL,
    DEFAULT_REGION,
    DEFAULT_MARKETS,
    DEFAULT_ODDS_FORMAT,
    DEFAULT_DATE_FORMAT,
    SUPPORTED_SPORTS,
    FREE_TIER_MONTHLY_LIMIT,
)


class OddsAPIClient:
    """
    Client for The Odds API with built-in rate limiting and usage tracking.
    Designed to work within free tier limits.
    """

    def __init__(self, api_key: Optional[str] = None, usage_file: str = "data/api_usage.json"):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Missing ODDS_API_KEY. Set env var ODDS_API_KEY or pass api_key parameter."
            )

        self.usage_file = Path(usage_file)
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)

        # Track API quota from response headers
        self.requests_remaining: Optional[int] = None
        self.requests_used: Optional[int] = None

    def _load_usage(self) -> dict:
        """Load usage tracking data from file"""
        if self.usage_file.exists():
            try:
                with open(self.usage_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"history": [], "last_remaining": None}

    def _save_usage(self, data: dict):
        """Save usage tracking data to file"""
        with open(self.usage_file, "w") as f:
            json.dump(data, f, indent=2)

    def _log_usage(self, sport: str, games_count: int):
        """Log API usage for tracking purposes"""
        usage = self._load_usage()
        usage["history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "sport": sport,
            "games_fetched": games_count,
            "requests_remaining": self.requests_remaining,
            "requests_used": self.requests_used,
        })

        # Keep only last 100 entries to prevent file bloat
        usage["history"] = usage["history"][-100:]
        usage["last_remaining"] = self.requests_remaining

        self._save_usage(usage)

    def get_quota_status(self) -> dict:
        """Get current API quota status"""
        usage = self._load_usage()
        return {
            "requests_remaining": usage.get("last_remaining"),
            "monthly_limit": FREE_TIER_MONTHLY_LIMIT,
            "recent_calls": len(usage.get("history", [])),
        }

    def check_quota_warning(self) -> Optional[str]:
        """Check if we're approaching quota limits and return warning message"""
        status = self.get_quota_status()
        remaining = status.get("requests_remaining")

        if remaining is not None:
            if remaining <= 10:
                return f"CRITICAL: Only {remaining} API requests remaining this month!"
            elif remaining <= 50:
                return f"WARNING: Only {remaining} API requests remaining this month."
            elif remaining <= 100:
                return f"Notice: {remaining} API requests remaining this month."

        return None

    def _request_with_retry(
        self,
        url: str,
        params: dict,
        max_attempts: int = 3,
        backoff_seconds: float = 2.0,
        timeout: int = 30,
    ) -> requests.Response:
        """
        GET with modest retry + exponential backoff (2s, 4s).

        Retries on connection errors, timeouts, HTTP 429, and HTTP 5xx.
        Does NOT retry other 4xx errors (bad key, quota exhausted, etc.).
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response

            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                body = e.response.text if e.response is not None else ""
                print(f"HTTP Error {status}: {body}")
                retryable = status is not None and (status == 429 or status >= 500)
                if not retryable or attempt >= max_attempts:
                    raise
                last_exc = e

            except (requests.ConnectionError, requests.Timeout) as e:
                print(f"Request failed: {e}")
                if attempt >= max_attempts:
                    raise
                last_exc = e

            wait = backoff_seconds * (2 ** (attempt - 1))
            print(f"Retrying in {wait:.0f}s (attempt {attempt}/{max_attempts} failed)...")
            time.sleep(wait)

        # Unreachable in practice, but keeps the type checker honest
        raise last_exc  # pragma: no cover

    def fetch_odds(
        self,
        sport: str,
        regions: str = DEFAULT_REGION,
        markets: str = DEFAULT_MARKETS,
        odds_format: str = DEFAULT_ODDS_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT,
    ) -> list:
        """
        Fetch odds for a specific sport.

        Args:
            sport: Sport key ('nfl' or 'nba')
            regions: Comma-separated regions (default: 'us')
            markets: Comma-separated markets (default: 'h2h,spreads,totals')
            odds_format: Odds format (default: 'american')
            date_format: Date format (default: 'iso')

        Returns:
            List of game data with odds
        """
        if sport.lower() not in SUPPORTED_SPORTS:
            raise ValueError(f"Unsupported sport: {sport}. Supported: {list(SUPPORTED_SPORTS.keys())}")

        sport_config = SUPPORTED_SPORTS[sport.lower()]
        api_sport_key = sport_config["api_key"]

        # Check quota before making request
        warning = self.check_quota_warning()
        if warning:
            print(warning)

        url = f"{BASE_URL}/sports/{api_sport_key}/odds/"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }

        response = self._request_with_retry(url, params)

        # Extract quota info from headers
        self.requests_remaining = int(response.headers.get("x-requests-remaining", -1))
        self.requests_used = int(response.headers.get("x-requests-used", -1))

        games = response.json()

        # Log usage
        self._log_usage(sport, len(games))

        print(f"Fetched {len(games)} {sport.upper()} games")
        print(f"API Quota: {self.requests_remaining} requests remaining (used {self.requests_used})")

        return games

    def fetch_all_sports(self) -> dict:
        """
        Fetch odds for all supported sports in a single run.
        More efficient than separate calls as it tracks quota across both.

        Returns:
            Dictionary with sport keys and their game data
        """
        results = {}
        errors = []

        for sport in SUPPORTED_SPORTS:
            try:
                games = self.fetch_odds(sport)
                results[sport] = games
            except Exception as e:
                errors.append(f"{sport}: {str(e)}")
                results[sport] = []

        if errors:
            print(f"Errors occurred: {errors}")

        return results

    def get_usage_report(self) -> str:
        """Generate a human-readable usage report"""
        usage = self._load_usage()
        history = usage.get("history", [])

        if not history:
            return "No API usage recorded yet."

        # Calculate statistics
        today = datetime.utcnow().date().isoformat()
        today_calls = [h for h in history if h["timestamp"].startswith(today)]

        lines = [
            "=== Odds API Usage Report ===",
            f"Total logged calls: {len(history)}",
            f"Calls today: {len(today_calls)}",
            f"Last known remaining: {usage.get('last_remaining', 'Unknown')}",
            "",
            "Recent calls:",
        ]

        for entry in history[-5:]:
            lines.append(
                f"  {entry['timestamp'][:16]} - {entry['sport'].upper()}: "
                f"{entry['games_fetched']} games (remaining: {entry['requests_remaining']})"
            )

        return "\n".join(lines)
