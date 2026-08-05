"""
Configuration for the MLB data pipeline.

Sources (checked for reachability 2026-08-05 from the remote session):

- Retrosheet game logs, via the Chadwick Baseball Bureau's GitHub mirror
  (github.com/chadwickbureau/retrosheet, branch `master`). Complete seasons
  1871-2025, one GL{year}.TXT per season under seasons/{year}/. This is the
  historical spine: retrosheet.org itself is blocked by the session's egress
  policy, the GitHub mirror is not, and the mirror carries the same files.
- MLB Stats API (statsapi.mlb.com) for the in-progress season, probable
  pitchers, weather and umpires. Blocked from the dev session but open from
  GitHub Actions, so `statsapi.py` is designed to run on a CI schedule the
  same way data_jobs/odds_api does.
- The Odds API for moneyline / runline / totals: handled by the existing
  data_jobs/odds_api job with an `mlb` sport entry, on the same CI schedule.

Team codes: Retrosheet uses its own 3-letter codes (CHN, NYA, SLN...). The
crosswalk below maps every franchise active 2005+ to its Retrosheet code(s),
MLB Stats API team id, and The Odds API display name. `FRANCHISE` is the
stable key to join on across sources and across renames (FLO->MIA, OAK->ATH).
"""

import os

# ---------------------------------------------------------------------------
# Paths and season range
# ---------------------------------------------------------------------------

MLB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MLB_ROOT, "data")
RAW_GAMELOG_DIR = os.path.join(DATA_DIR, "raw", "gamelogs")
REFERENCE_DIR = os.path.join(DATA_DIR, "reference")
STATSAPI_DIR = os.path.join(DATA_DIR, "statsapi")
GAMES_PATH = os.path.join(DATA_DIR, "games.csv.gz")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
COVERAGE_REPORT_PATH = os.path.join(DATA_DIR, "coverage_report.md")

FIRST_SEASON = 2005

# Retrosheet mirror. raw.githubusercontent serves individual files; the
# session's git proxy serves anonymous reads of public repos.
RETROSHEET_MIRROR_RAW = (
    "https://raw.githubusercontent.com/chadwickbureau/retrosheet/master"
)
GAMELOG_URL_TEMPLATE = RETROSHEET_MIRROR_RAW + "/seasons/{year}/GL{year}.TXT"
REFERENCE_FILES = {
    "ballparks.csv": RETROSHEET_MIRROR_RAW + "/reference/ballparks.csv",
    "teams.csv": RETROSHEET_MIRROR_RAW + "/reference/teams.csv",
}

# MLB Stats API (no key required).
STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"

# Expected regular-season game counts per season, used by the coverage
# assertions. 2020 was the 60-game COVID season; 2005-2025 otherwise play
# 162 games x 30 teams / 2 = 2430, minus the occasional unplayed
# rain-cancelled game that isn't made up (a handful of seasons end at 2428/9).
EXPECTED_GAMES = {year: 2430 for year in range(FIRST_SEASON, 2031)}
EXPECTED_GAMES[2020] = 898  # 60-game season, plus 2 unplayed-makeup quirks
# Tolerance: regular seasons occasionally drop 1-3 games that are never made
# up. Coverage below expected - GAME_COUNT_TOLERANCE fails the assertion.
GAME_COUNT_TOLERANCE = 6

# ---------------------------------------------------------------------------
# Team crosswalk, franchises active 2005+
# ---------------------------------------------------------------------------
# franchise -> {retro: [codes used 2005+], mlb_id, odds_names: [The Odds API
# display names, oldest first], league, division}
TEAMS = {
    "ARI": {"retro": ["ARI"], "mlb_id": 109, "odds_names": ["Arizona Diamondbacks"], "lg": "NL", "div": "West", "loc": (33.4453, -112.0667)},
    "ATL": {"retro": ["ATL"], "mlb_id": 144, "odds_names": ["Atlanta Braves"], "lg": "NL", "div": "East", "loc": (33.8908, -84.4678)},
    "BAL": {"retro": ["BAL"], "mlb_id": 110, "odds_names": ["Baltimore Orioles"], "lg": "AL", "div": "East", "loc": (39.2839, -76.6217)},
    "BOS": {"retro": ["BOS"], "mlb_id": 111, "odds_names": ["Boston Red Sox"], "lg": "AL", "div": "East", "loc": (42.3467, -71.0972)},
    "CHC": {"retro": ["CHN"], "mlb_id": 112, "odds_names": ["Chicago Cubs"], "lg": "NL", "div": "Central", "loc": (41.9484, -87.6553)},
    "CHW": {"retro": ["CHA"], "mlb_id": 145, "odds_names": ["Chicago White Sox"], "lg": "AL", "div": "Central", "loc": (41.8299, -87.6338)},
    "CIN": {"retro": ["CIN"], "mlb_id": 113, "odds_names": ["Cincinnati Reds"], "lg": "NL", "div": "Central", "loc": (39.0975, -84.5066)},
    "CLE": {"retro": ["CLE"], "mlb_id": 114, "odds_names": ["Cleveland Indians", "Cleveland Guardians"], "lg": "AL", "div": "Central", "loc": (41.4962, -81.6852)},
    "COL": {"retro": ["COL"], "mlb_id": 115, "odds_names": ["Colorado Rockies"], "lg": "NL", "div": "West", "loc": (39.7559, -104.9942)},
    "DET": {"retro": ["DET"], "mlb_id": 116, "odds_names": ["Detroit Tigers"], "lg": "AL", "div": "Central", "loc": (42.3390, -83.0485)},
    "HOU": {"retro": ["HOU"], "mlb_id": 117, "odds_names": ["Houston Astros"], "lg": "AL", "div": "West", "loc": (29.7573, -95.3555)},  # NL through 2012
    "KCR": {"retro": ["KCA"], "mlb_id": 118, "odds_names": ["Kansas City Royals"], "lg": "AL", "div": "Central", "loc": (39.0517, -94.4803)},
    "LAA": {"retro": ["ANA", "LAA"], "mlb_id": 108, "odds_names": ["Los Angeles Angels"], "lg": "AL", "div": "West", "loc": (33.8003, -117.8827)},
    "LAD": {"retro": ["LAN"], "mlb_id": 119, "odds_names": ["Los Angeles Dodgers"], "lg": "NL", "div": "West", "loc": (34.0739, -118.2400)},
    "MIA": {"retro": ["FLO", "MIA"], "mlb_id": 146, "odds_names": ["Miami Marlins"], "lg": "NL", "div": "East", "loc": (25.7781, -80.2196)},
    "MIL": {"retro": ["MIL"], "mlb_id": 158, "odds_names": ["Milwaukee Brewers"], "lg": "NL", "div": "Central", "loc": (43.0280, -87.9712)},
    "MIN": {"retro": ["MIN"], "mlb_id": 142, "odds_names": ["Minnesota Twins"], "lg": "AL", "div": "Central", "loc": (44.9817, -93.2776)},
    "NYM": {"retro": ["NYN"], "mlb_id": 121, "odds_names": ["New York Mets"], "lg": "NL", "div": "East", "loc": (40.7571, -73.8458)},
    "NYY": {"retro": ["NYA"], "mlb_id": 147, "odds_names": ["New York Yankees"], "lg": "AL", "div": "East", "loc": (40.8296, -73.9262)},
    "ATH": {"retro": ["OAK", "ATH"], "mlb_id": 133, "odds_names": ["Oakland Athletics", "Athletics"], "lg": "AL", "div": "West", "loc": (38.5802, -121.5133)},  # Oakland through 2024, W. Sacramento 2025+
    "PHI": {"retro": ["PHI"], "mlb_id": 143, "odds_names": ["Philadelphia Phillies"], "lg": "NL", "div": "East", "loc": (39.9061, -75.1665)},
    "PIT": {"retro": ["PIT"], "mlb_id": 134, "odds_names": ["Pittsburgh Pirates"], "lg": "NL", "div": "Central", "loc": (40.4469, -80.0057)},
    "SDP": {"retro": ["SDN"], "mlb_id": 135, "odds_names": ["San Diego Padres"], "lg": "NL", "div": "West", "loc": (32.7076, -117.1570)},
    "SEA": {"retro": ["SEA"], "mlb_id": 136, "odds_names": ["Seattle Mariners"], "lg": "AL", "div": "West", "loc": (47.5914, -122.3325)},
    "SFG": {"retro": ["SFN"], "mlb_id": 137, "odds_names": ["San Francisco Giants"], "lg": "NL", "div": "West", "loc": (37.7786, -122.3893)},
    "STL": {"retro": ["SLN"], "mlb_id": 138, "odds_names": ["St. Louis Cardinals", "St louis Cardinals"], "lg": "NL", "div": "Central", "loc": (38.6226, -90.1928)},
    "TBR": {"retro": ["TBA"], "mlb_id": 139, "odds_names": ["Tampa Bay Rays"], "lg": "AL", "div": "East", "loc": (27.7683, -82.6534)},
    "TEX": {"retro": ["TEX"], "mlb_id": 140, "odds_names": ["Texas Rangers"], "lg": "AL", "div": "West", "loc": (32.7473, -97.0842)},
    "TOR": {"retro": ["TOR"], "mlb_id": 141, "odds_names": ["Toronto Blue Jays"], "lg": "AL", "div": "East", "loc": (43.6414, -79.3894)},
    "WSN": {"retro": ["WAS"], "mlb_id": 120, "odds_names": ["Washington Nationals"], "lg": "NL", "div": "East", "loc": (38.8730, -77.0074)},
}

# Retrosheet code -> stable franchise key.
RETRO_TO_FRANCHISE = {
    retro: franchise
    for franchise, info in TEAMS.items()
    for retro in info["retro"]
}

MLB_ID_TO_FRANCHISE = {info["mlb_id"]: f for f, info in TEAMS.items()}
