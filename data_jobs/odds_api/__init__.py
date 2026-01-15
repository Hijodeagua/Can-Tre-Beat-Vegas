"""
Odds API Module - Unified odds data fetching for NBA and NFL
Optimized for The Odds API free tier (500 requests/month)
"""

from .config import NFL_TEAMS, NBA_TEAMS, SUPPORTED_SPORTS
from .client import OddsAPIClient
from .processors import process_game_data

__all__ = [
    "OddsAPIClient",
    "NFL_TEAMS",
    "NBA_TEAMS",
    "SUPPORTED_SPORTS",
    "process_game_data",
]
