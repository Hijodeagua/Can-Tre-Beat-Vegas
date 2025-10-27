# nba_odds_pull.py
# Imports
import os
import sys
import requests
import pandas as pd
from datetime import datetime
import pytz
from geopy.distance import geodesic

# ===== API KEY CONFIG =====
# Prefer environment variable in prod/CI; fall back to a local hard-coded key for dev.
HARDCODED_API_KEY = "change in local"  # 👈 replace this with your real Odds API key

API_KEY = os.environ.get("ODDS_API_KEY") or HARDCODED_API_KEY

# Basic guard: force you to replace the placeholder before use
if not API_KEY or API_KEY == "PASTE_TEMP_DEV_KEY_HERE":
    raise RuntimeError(
        "Missing ODDS_API_KEY. Set env var ODDS_API_KEY or put a temp key in HARDCODED_API_KEY for local testing."
    )


# ----- Pull NBA odds -----
SPORT = "basketball_nba"
url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds/"
params = {
    "apiKey": API_KEY,
    "regions": "us",
    "markets": "h2h,spreads,totals",
    "oddsFormat": "american",
    "dateFormat": "iso",
}

resp = requests.get(url, params=params)
try:
    resp.raise_for_status()
except requests.HTTPError as e:
    print(f"Error {resp.status_code}: {resp.text}")
    raise

games = resp.json()
print(f"✅ Found {len(games)} NBA games.")

# ----- Team metadata (conference, division, arena lat/lon) -----
team_info = {
    # EAST — Atlantic
    "Boston Celtics": {"conf": "East", "div": "Atlantic", "loc": (42.3662, -71.0621)},
    "Brooklyn Nets": {"conf": "East", "div": "Atlantic", "loc": (40.6826, -73.9754)},
    "New York Knicks": {"conf": "East", "div": "Atlantic", "loc": (40.7505, -73.9934)},
    "Philadelphia 76ers": {"conf": "East", "div": "Atlantic", "loc": (39.9012, -75.1720)},
    "Toronto Raptors": {"conf": "East", "div": "Atlantic", "loc": (43.6435, -79.3791)},
    # EAST — Central
    "Chicago Bulls": {"conf": "East", "div": "Central", "loc": (41.8807, -87.6742)},
    "Cleveland Cavaliers": {"conf": "East", "div": "Central", "loc": (41.4966, -81.6881)},
    "Detroit Pistons": {"conf": "East", "div": "Central", "loc": (42.3410, -83.0550)},
    "Indiana Pacers": {"conf": "East", "div": "Central", "loc": (39.7639, -86.1555)},
    "Milwaukee Bucks": {"conf": "East", "div": "Central", "loc": (43.0451, -87.9172)},
    # EAST — Southeast
    "Atlanta Hawks": {"conf": "East", "div": "Southeast", "loc": (33.7573, -84.3963)},
    "Charlotte Hornets": {"conf": "East", "div": "Southeast", "loc": (35.2251, -80.8392)},
    "Miami Heat": {"conf": "East", "div": "Southeast", "loc": (25.7814, -80.1880)},
    "Orlando Magic": {"conf": "East", "div": "Southeast", "loc": (28.5392, -81.3839)},
    "Washington Wizards": {"conf": "East", "div": "Southeast", "loc": (38.8981, -77.0209)},

    # WEST — Northwest
    "Denver Nuggets": {"conf": "West", "div": "Northwest", "loc": (39.7487, -105.0077)},
    "Minnesota Timberwolves": {"conf": "West", "div": "Northwest", "loc": (44.9795, -93.2760)},
    "Oklahoma City Thunder": {"conf": "West", "div": "Northwest", "loc": (35.4634, -97.5151)},
    "Portland Trail Blazers": {"conf": "West", "div": "Northwest", "loc": (45.5316, -122.6668)},
    "Utah Jazz": {"conf": "West", "div": "Northwest", "loc": (40.7683, -111.9011)},
    # WEST — Pacific
    "Golden State Warriors": {"conf": "West", "div": "Pacific", "loc": (37.7680, -122.3877)},
    "Los Angeles Clippers": {"conf": "West", "div": "Pacific", "loc": (33.9345, -118.3391)},  # Intuit Dome
    "Los Angeles Lakers": {"conf": "West", "div": "Pacific", "loc": (34.0430, -118.2673)},   # Crypto.com Arena
    "Phoenix Suns": {"conf": "West", "div": "Pacific", "loc": (33.4457, -112.0712)},
    "Sacramento Kings": {"conf": "West", "div": "Pacific", "loc": (38.5802, -121.4997)},
    # WEST — Southwest
    "Dallas Mavericks": {"conf": "West", "div": "Southwest", "loc": (32.7905, -96.8104)},
    "Houston Rockets": {"conf": "West", "div": "Southwest", "loc": (29.7508, -95.3621)},
    "Memphis Grizzlies": {"conf": "West", "div": "Southwest", "loc": (35.1382, -90.0506)},
    "New Orleans Pelicans": {"conf": "West", "div": "Southwest", "loc": (29.9489, -90.0810)},
    "San Antonio Spurs": {"conf": "West", "div": "Southwest", "loc": (29.4271, -98.4375)},
}

# Build table
rows = []
eastern = pytz.timezone("US/Eastern")
timestamp_pulled = datetime.now(pytz.UTC).astimezone(eastern).strftime("%Y-%m-%d %H:%M:%S")

for game in games:
    # commence_time is ISO with Z; convert to aware datetime
    utc_time = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
    eastern_time = utc_time.astimezone(eastern)
    prime_time = eastern_time.hour >= 19  # 7pm ET or later

    home_team = game["home_team"]
    away_team = game["away_team"]

    # Skip if unexpected team names (e.g., preseason or feed quirks)
    if home_team not in team_info or away_team not in team_info:
        # You can print to debug if you want:
        # print("Skipping teams not in mapping:", home_team, away_team)
        continue

    # Divisional matchup check
    div_match = (
        team_info[home_team]["conf"] == team_info[away_team]["conf"]
        and team_info[home_team]["div"] == team_info[away_team]["div"]
    )

    # Travel distance in miles (between arenas)
    travel_distance = geodesic(
        team_info[home_team]["loc"], team_info[away_team]["loc"]
    ).miles

    # Lists to calculate averages
    home_spread_odds = []
    away_spread_odds = []
    home_h2h_odds = []
    away_h2h_odds = []
    over_odds = []
    under_odds = []

    # Collect all odds across sportsbooks (defensively)
    for bookmaker in game.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", []) or []

            if key == "spreads":
                for outcome in outcomes:
                    if outcome.get("name") == home_team:
                        home_spread_odds.append(outcome.get("price"))
                    elif outcome.get("name") == away_team:
                        away_spread_odds.append(outcome.get("price"))

            elif key == "h2h":
                for outcome in outcomes:
                    if outcome.get("name") == home_team:
                        home_h2h_odds.append(outcome.get("price"))
                    elif outcome.get("name") == away_team:
                        away_h2h_odds.append(outcome.get("price"))

            elif key == "totals":
                for outcome in outcomes:
                    n = str(outcome.get("name", "")).lower()
                    if n == "over":
                        over_odds.append(outcome.get("price"))
                    elif n == "under":
                        under_odds.append(outcome.get("price"))

    row = {
        "League": "NBA",
        "Timestamp Pulled": timestamp_pulled,
        "Date of Game (ET)": eastern_time.strftime("%Y-%m-%d %H:%M"),
        "Home Team": home_team,
        "Away Team": away_team,
        "Divisional Matchup": "Yes" if div_match else "No",
        "Travel Distance (mi)": round(travel_distance, 1),
        "Prime Time (ET >= 7p)": "Yes" if prime_time else "No",
        "Avg Home Spread Odds": round(sum(home_spread_odds) / len(home_spread_odds), 2) if home_spread_odds else None,
        "Avg Away Spread Odds": round(sum(away_spread_odds) / len(away_spread_odds), 2) if away_spread_odds else None,
        "Avg Home H2H Odds": round(sum(home_h2h_odds) / len(home_h2h_odds), 2) if home_h2h_odds else None,
        "Avg Away H2H Odds": round(sum(away_h2h_odds) / len(away_h2h_odds), 2) if away_h2h_odds else None,
        "Avg Over Odds": round(sum(over_odds) / len(over_odds), 2) if over_odds else None,
        "Avg Under Odds": round(sum(under_odds) / len(under_odds), 2) if under_odds else None,
        "Home Conf": team_info[home_team]["conf"],
        "Home Div": team_info[home_team]["div"],
        "Away Conf": team_info[away_team]["conf"],
        "Away Div": team_info[away_team]["div"],
    }

    # Per-book details (optional but nice)
    for book in game.get("bookmakers", []):
        book_name = book.get("title", "Unknown")
        home_h2h = away_h2h = home_spread = away_spread = over_total = under_total = None

        for market in book.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", []) or []

            if key == "h2h":
                for outcome in outcomes:
                    if outcome.get("name") == home_team:
                        home_h2h = outcome.get("price")
                    elif outcome.get("name") == away_team:
                        away_h2h = outcome.get("price")

            elif key == "spreads":
                for outcome in outcomes:
                    if outcome.get("name") == home_team:
                        home_spread = outcome.get("price")
                    elif outcome.get("name") == away_team:
                        away_spread = outcome.get("price")

            elif key == "totals":
                for outcome in outcomes:
                    n = str(outcome.get("name", "")).lower()
                    if n == "over":
                        over_total = outcome.get("price")
                    elif n == "under":
                        under_total = outcome.get("price")

        row[f"Home {book_name} Spread Odds"] = home_spread
        row[f"Away {book_name} Spread Odds"] = away_spread
        row[f"Home {book_name} H2H Odds"] = home_h2h
        row[f"Away {book_name} H2H Odds"] = away_h2h
        row[f"Home {book_name} O/U Odds"] = over_total
        row[f"Away {book_name} O/U Odds"] = under_total

    rows.append(row)

df = pd.DataFrame(rows)

# Save to repo data/nba folder
save_dir = "data/nba"
os.makedirs(save_dir, exist_ok=True)

# Add date + hour (UTC) stamp: YYYY-MM-DD-HHMM
stamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M")

# League-specific filenames to avoid clobbering your NFL files
file_timestamped = os.path.join(save_dir, f"nba_odds_api_data_{stamp}.csv")
file_latest = os.path.join(save_dir, "nba_odds_api_data_latest.csv")

df.to_csv(file_timestamped, index=False)
df.to_csv(file_latest, index=False)

print(f"Saved {len(df)} rows to {file_timestamped} and {file_latest}")


