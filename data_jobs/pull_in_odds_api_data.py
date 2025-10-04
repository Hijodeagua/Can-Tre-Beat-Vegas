# Imports
import os
import sys
import requests
import pandas as pd
from datetime import datetime
import pytz
from bs4 import BeautifulSoup  # if you don't use it, feel free to remove from imports and requirements
from geopy.distance import geodesic

# Read API key from environment
API_KEY = os.environ.get("ODDS_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing ODDS_API_KEY env var. Add it as a GitHub Actions secret.")

# ----- (optional) list available sports -----
def get_available_sports():
    url = "https://api.the-odds-api.com/v4/sports"
    params = {"apiKey": API_KEY}
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

def show_valid_options():
    print("\n🎯 Valid Options by Category:\n")
    print("🗺️ Regions:\n- us\n- us2\n- uk\n- au\n- eu")
    print("\n🎯 Markets:\n- h2h (moneyline)\n- spreads (point spread)\n- totals (over/under)\n- outrights (futures)")
    print("\n⏱️ Date Formats:\n- iso (default)\n- unix")
    print("\n📈 Odds Formats:\n- decimal (default)\n- american")
    print("\n🔗 Other Flags:\n- includeLinks\n- includeSids\n- includeBetLimits")
# show_valid_options()  # optional

# ----- Pull NFL odds -----
SPORT = "americanfootball_nfl"
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
    # Helpful diagnostics for common Odds API errors
    print(f"Error {resp.status_code}: {resp.text}")
    raise

games = resp.json()
print(f"✅ Found {len(games)} NFL games.")

# ----- Team metadata -----
team_info = {
    "Buffalo Bills": {"conf": "AFC", "div": "East", "loc": (42.7738, -78.7868)},
    "Miami Dolphins": {"conf": "AFC", "div": "East", "loc": (25.9580, -80.2389)},
    "New England Patriots": {"conf": "AFC", "div": "East", "loc": (42.0909, -71.2643)},
    "New York Jets": {"conf": "AFC", "div": "East", "loc": (40.8136, -74.0740)},
    "Baltimore Ravens": {"conf": "AFC", "div": "North", "loc": (39.2780, -76.6227)},
    "Cincinnati Bengals": {"conf": "AFC", "div": "North", "loc": (39.0954, -84.5161)},
    "Cleveland Browns": {"conf": "AFC", "div": "North", "loc": (41.5061, -81.6995)},
    "Pittsburgh Steelers": {"conf": "AFC", "div": "North", "loc": (40.4468, -80.0158)},
    "Houston Texans": {"conf": "AFC", "div": "South", "loc": (29.6847, -95.4107)},
    "Indianapolis Colts": {"conf": "AFC", "div": "South", "loc": (39.7601, -86.1639)},
    "Jacksonville Jaguars": {"conf": "AFC", "div": "South", "loc": (30.3239, -81.6373)},
    "Tennessee Titans": {"conf": "AFC", "div": "South", "loc": (36.1665, -86.7713)},
    "Denver Broncos": {"conf": "AFC", "div": "West", "loc": (39.7439, -105.0201)},
    "Kansas City Chiefs": {"conf": "AFC", "div": "West", "loc": (39.0490, -94.4840)},
    "Las Vegas Raiders": {"conf": "AFC", "div": "West", "loc": (36.0909, -115.1830)},
    "Los Angeles Chargers": {"conf": "AFC", "div": "West", "loc": (33.9535, -118.3391)},
    "Dallas Cowboys": {"conf": "NFC", "div": "East", "loc": (32.7473, -97.0945)},
    "New York Giants": {"conf": "NFC", "div": "East", "loc": (40.8136, -74.0740)},
    "Philadelphia Eagles": {"conf": "NFC", "div": "East", "loc": (39.9008, -75.1675)},
    "Washington Commanders": {"conf": "NFC", "div": "East", "loc": (38.9076, -76.8645)},
    "Chicago Bears": {"conf": "NFC", "div": "North", "loc": (41.8623, -87.6167)},
    "Detroit Lions": {"conf": "NFC", "div": "North", "loc": (42.3400, -83.0456)},
    "Green Bay Packers": {"conf": "NFC", "div": "North", "loc": (44.5013, -88.0622)},
    "Minnesota Vikings": {"conf": "NFC", "div": "North", "loc": (44.9738, -93.2581)},
    "Atlanta Falcons": {"conf": "NFC", "div": "South", "loc": (33.7554, -84.4008)},
    "Carolina Panthers": {"conf": "NFC", "div": "South", "loc": (35.2251, -80.8528)},
    "New Orleans Saints": {"conf": "NFC", "div": "South", "loc": (29.9509, -90.0815)},
    "Tampa Bay Buccaneers": {"conf": "NFC", "div": "South", "loc": (27.9759, -82.5033)},
    "Arizona Cardinals": {"conf": "NFC", "div": "West", "loc": (33.5276, -112.2626)},
    "Los Angeles Rams": {"conf": "NFC", "div": "West", "loc": (33.9535, -118.3391)},
    "San Francisco 49ers": {"conf": "NFC", "div": "West", "loc": (37.4030, -121.9700)},
    "Seattle Seahawks": {"conf": "NFC", "div": "West", "loc": (47.5952, -122.3316)},
}

# Build table
rows = []
eastern = pytz.timezone("US/Eastern")
timestamp_pulled = datetime.now(pytz.UTC).astimezone(eastern).strftime("%Y-%m-%d %H:%M:%S")

for game in games:
    # commence_time is ISO with Z; convert to aware datetime
    utc_time = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
    eastern_time = utc_time.astimezone(eastern)
    prime_time = eastern_time.hour >= 19

    home_team = game["home_team"]
    away_team = game["away_team"]

    # Team safety check
    if home_team not in team_info or away_team not in team_info:
        # Skip unexpected teams (e.g., preseason mishaps)
        continue

    # Divisional matchup check
    div_match = (
        team_info[home_team]["conf"] == team_info[away_team]["conf"]
        and team_info[home_team]["div"] == team_info[away_team]["div"]
    )

    # Travel distance in miles
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
                    if str(outcome.get("name", "")).lower() == "over":
                        over_odds.append(outcome.get("price"))
                    elif str(outcome.get("name", "")).lower() == "under":
                        under_odds.append(outcome.get("price"))

    row = {
        "Timestamp Pulled": timestamp_pulled,
        "Date of Game": eastern_time.strftime("%Y-%m-%d %H:%M"),
        "Home Team": home_team,
        "Away Team": away_team,
        "Divisional Matchup": "Yes" if div_match else "No",
        "Travel Distance (mi)": round(travel_distance, 1),
        "Prime Time": "Yes" if prime_time else "No",
        "Avg Home Spread Odds": round(sum(home_spread_odds) / len(home_spread_odds), 2) if home_spread_odds else None,
        "Avg Away Spread Odds": round(sum(away_spread_odds) / len(away_spread_odds), 2) if away_spread_odds else None,
        "Avg Home H2H Odds": round(sum(home_h2h_odds) / len(home_h2h_odds), 2) if home_h2h_odds else None,
        "Avg Away H2H Odds": round(sum(away_h2h_odds) / len(away_h2h_odds), 2) if away_h2h_odds else None,
        "Avg Over Odds": round(sum(over_odds) / len(over_odds), 2) if over_odds else None,
        "Avg Under Odds": round(sum(under_odds) / len(under_odds), 2) if under_odds else None,
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
                    if str(outcome.get("name", "")).lower() == "over":
                        over_total = outcome.get("price")
                    elif str(outcome.get("name", "")).lower() == "under":
                        under_total = outcome.get("price")

        row[f"Home {book_name} Spread Odds"] = home_spread
        row[f"Away {book_name} Spread Odds"] = away_spread
        row[f"Home {book_name} H2H Odds"] = home_h2h
        row[f"Away {book_name} H2H Odds"] = away_h2h
        row[f"Home {book_name} O/U Odds"] = over_total
        row[f"Away {book_name} O/U Odds"] = under_total

    rows.append(row)

df = pd.DataFrame(rows)

# Save to repo data folder
os.makedirs("data", exist_ok=True)
stamp = datetime.utcnow().strftime("%Y-%m-%d")
df.to_csv(f"data/odds_api_data_{stamp}.csv", index=False)
df.to_csv("data/odds_api_data_latest.csv", index=False)
print(f"Saved {len(df)} rows to data/odds_api_data_{stamp}.csv and data/odds_api_data_latest.csv")
