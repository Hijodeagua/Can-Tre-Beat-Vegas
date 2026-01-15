"""
Configuration and team metadata for Odds API
"""

# API Configuration
BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_REGION = "us"
DEFAULT_MARKETS = "h2h,spreads,totals"
DEFAULT_ODDS_FORMAT = "american"
DEFAULT_DATE_FORMAT = "iso"

# Free tier limits
FREE_TIER_MONTHLY_LIMIT = 500
FREE_TIER_DAILY_RECOMMENDED = 16  # ~500/31 days with buffer

# Supported sports configuration
SUPPORTED_SPORTS = {
    "nfl": {
        "api_key": "americanfootball_nfl",
        "name": "NFL",
        "data_dir": "data",
        "file_prefix": "odds_api_data",
    },
    "nba": {
        "api_key": "basketball_nba",
        "name": "NBA",
        "data_dir": "data/nba",
        "file_prefix": "nba_odds_api_data",
    },
}

# NFL Team metadata (conference, division, stadium coordinates)
NFL_TEAMS = {
    # AFC East
    "Buffalo Bills": {"conf": "AFC", "div": "East", "loc": (42.7738, -78.7868)},
    "Miami Dolphins": {"conf": "AFC", "div": "East", "loc": (25.9580, -80.2389)},
    "New England Patriots": {"conf": "AFC", "div": "East", "loc": (42.0909, -71.2643)},
    "New York Jets": {"conf": "AFC", "div": "East", "loc": (40.8136, -74.0740)},
    # AFC North
    "Baltimore Ravens": {"conf": "AFC", "div": "North", "loc": (39.2780, -76.6227)},
    "Cincinnati Bengals": {"conf": "AFC", "div": "North", "loc": (39.0954, -84.5161)},
    "Cleveland Browns": {"conf": "AFC", "div": "North", "loc": (41.5061, -81.6995)},
    "Pittsburgh Steelers": {"conf": "AFC", "div": "North", "loc": (40.4468, -80.0158)},
    # AFC South
    "Houston Texans": {"conf": "AFC", "div": "South", "loc": (29.6847, -95.4107)},
    "Indianapolis Colts": {"conf": "AFC", "div": "South", "loc": (39.7601, -86.1639)},
    "Jacksonville Jaguars": {"conf": "AFC", "div": "South", "loc": (30.3239, -81.6373)},
    "Tennessee Titans": {"conf": "AFC", "div": "South", "loc": (36.1665, -86.7713)},
    # AFC West
    "Denver Broncos": {"conf": "AFC", "div": "West", "loc": (39.7439, -105.0201)},
    "Kansas City Chiefs": {"conf": "AFC", "div": "West", "loc": (39.0490, -94.4840)},
    "Las Vegas Raiders": {"conf": "AFC", "div": "West", "loc": (36.0909, -115.1830)},
    "Los Angeles Chargers": {"conf": "AFC", "div": "West", "loc": (33.9535, -118.3391)},
    # NFC East
    "Dallas Cowboys": {"conf": "NFC", "div": "East", "loc": (32.7473, -97.0945)},
    "New York Giants": {"conf": "NFC", "div": "East", "loc": (40.8136, -74.0740)},
    "Philadelphia Eagles": {"conf": "NFC", "div": "East", "loc": (39.9008, -75.1675)},
    "Washington Commanders": {"conf": "NFC", "div": "East", "loc": (38.9076, -76.8645)},
    # NFC North
    "Chicago Bears": {"conf": "NFC", "div": "North", "loc": (41.8623, -87.6167)},
    "Detroit Lions": {"conf": "NFC", "div": "North", "loc": (42.3400, -83.0456)},
    "Green Bay Packers": {"conf": "NFC", "div": "North", "loc": (44.5013, -88.0622)},
    "Minnesota Vikings": {"conf": "NFC", "div": "North", "loc": (44.9738, -93.2581)},
    # NFC South
    "Atlanta Falcons": {"conf": "NFC", "div": "South", "loc": (33.7554, -84.4008)},
    "Carolina Panthers": {"conf": "NFC", "div": "South", "loc": (35.2251, -80.8528)},
    "New Orleans Saints": {"conf": "NFC", "div": "South", "loc": (29.9509, -90.0815)},
    "Tampa Bay Buccaneers": {"conf": "NFC", "div": "South", "loc": (27.9759, -82.5033)},
    # NFC West
    "Arizona Cardinals": {"conf": "NFC", "div": "West", "loc": (33.5276, -112.2626)},
    "Los Angeles Rams": {"conf": "NFC", "div": "West", "loc": (33.9535, -118.3391)},
    "San Francisco 49ers": {"conf": "NFC", "div": "West", "loc": (37.4030, -121.9700)},
    "Seattle Seahawks": {"conf": "NFC", "div": "West", "loc": (47.5952, -122.3316)},
}

# NBA Team metadata (conference, division, arena coordinates)
NBA_TEAMS = {
    # EAST - Atlantic
    "Boston Celtics": {"conf": "East", "div": "Atlantic", "loc": (42.3662, -71.0621)},
    "Brooklyn Nets": {"conf": "East", "div": "Atlantic", "loc": (40.6826, -73.9754)},
    "New York Knicks": {"conf": "East", "div": "Atlantic", "loc": (40.7505, -73.9934)},
    "Philadelphia 76ers": {"conf": "East", "div": "Atlantic", "loc": (39.9012, -75.1720)},
    "Toronto Raptors": {"conf": "East", "div": "Atlantic", "loc": (43.6435, -79.3791)},
    # EAST - Central
    "Chicago Bulls": {"conf": "East", "div": "Central", "loc": (41.8807, -87.6742)},
    "Cleveland Cavaliers": {"conf": "East", "div": "Central", "loc": (41.4966, -81.6881)},
    "Detroit Pistons": {"conf": "East", "div": "Central", "loc": (42.3410, -83.0550)},
    "Indiana Pacers": {"conf": "East", "div": "Central", "loc": (39.7639, -86.1555)},
    "Milwaukee Bucks": {"conf": "East", "div": "Central", "loc": (43.0451, -87.9172)},
    # EAST - Southeast
    "Atlanta Hawks": {"conf": "East", "div": "Southeast", "loc": (33.7573, -84.3963)},
    "Charlotte Hornets": {"conf": "East", "div": "Southeast", "loc": (35.2251, -80.8392)},
    "Miami Heat": {"conf": "East", "div": "Southeast", "loc": (25.7814, -80.1880)},
    "Orlando Magic": {"conf": "East", "div": "Southeast", "loc": (28.5392, -81.3839)},
    "Washington Wizards": {"conf": "East", "div": "Southeast", "loc": (38.8981, -77.0209)},
    # WEST - Northwest
    "Denver Nuggets": {"conf": "West", "div": "Northwest", "loc": (39.7487, -105.0077)},
    "Minnesota Timberwolves": {"conf": "West", "div": "Northwest", "loc": (44.9795, -93.2760)},
    "Oklahoma City Thunder": {"conf": "West", "div": "Northwest", "loc": (35.4634, -97.5151)},
    "Portland Trail Blazers": {"conf": "West", "div": "Northwest", "loc": (45.5316, -122.6668)},
    "Utah Jazz": {"conf": "West", "div": "Northwest", "loc": (40.7683, -111.9011)},
    # WEST - Pacific
    "Golden State Warriors": {"conf": "West", "div": "Pacific", "loc": (37.7680, -122.3877)},
    "Los Angeles Clippers": {"conf": "West", "div": "Pacific", "loc": (33.9345, -118.3391)},
    "Los Angeles Lakers": {"conf": "West", "div": "Pacific", "loc": (34.0430, -118.2673)},
    "Phoenix Suns": {"conf": "West", "div": "Pacific", "loc": (33.4457, -112.0712)},
    "Sacramento Kings": {"conf": "West", "div": "Pacific", "loc": (38.5802, -121.4997)},
    # WEST - Southwest
    "Dallas Mavericks": {"conf": "West", "div": "Southwest", "loc": (32.7905, -96.8104)},
    "Houston Rockets": {"conf": "West", "div": "Southwest", "loc": (29.7508, -95.3621)},
    "Memphis Grizzlies": {"conf": "West", "div": "Southwest", "loc": (35.1382, -90.0506)},
    "New Orleans Pelicans": {"conf": "West", "div": "Southwest", "loc": (29.9489, -90.0810)},
    "San Antonio Spurs": {"conf": "West", "div": "Southwest", "loc": (29.4271, -98.4375)},
}


def get_team_info(sport: str) -> dict:
    """Get team metadata for a specific sport"""
    if sport.lower() == "nfl":
        return NFL_TEAMS
    elif sport.lower() == "nba":
        return NBA_TEAMS
    else:
        raise ValueError(f"Unsupported sport: {sport}")
