"""Shared constants for the daily MLB pipeline: paths, team metadata,
league/division structure, and simulation defaults."""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

GAMES_CSV = REPO / "data" / "mlb" / "games_2009_2026.csv"
SCHEDULE_CSV = REPO / "data" / "mlb" / "schedule_2026_remaining.csv"
PREDICTIONS_DIR = REPO / "data" / "mlb" / "predictions"
GRADES_CSV = PREDICTIONS_DIR / "grades.csv"
REPORTS_DIR = REPO / "reports" / "mlb_daily"
SITE_DATA_DIR = REPO / "web" / "public" / "data" / "mlb"
SITE_HISTORY_DIR = SITE_DATA_DIR / "history"

CURRENT_SEASON = 2026

# Simulation defaults. Season sims are the expensive knob: 2000 sims over the
# ~750-game August remainder runs in a few seconds vectorized, cheap enough
# for a daily cadence.
SEASON_SIMS = 2000
GAME_SIMS = 10000

# Franchise codes are canonical current-day BRef codes (matches games CSV
# away_fr/home_fr and mlb/elo.py).
DIVISIONS = {
    "AL East": ["NYY", "BOS", "TBR", "TOR", "BAL"],
    "AL Central": ["CLE", "MIN", "DET", "CHW", "KCR"],
    "AL West": ["HOU", "SEA", "TEX", "LAA", "ATH"],
    "NL East": ["ATL", "PHI", "NYM", "MIA", "WSN"],
    "NL Central": ["MIL", "CHC", "STL", "CIN", "PIT"],
    "NL West": ["LAD", "SDP", "ARI", "SFG", "COL"],
}

TEAM_DIVISION = {t: d for d, teams in DIVISIONS.items() for t in teams}
TEAM_LEAGUE = {t: d.split(" ")[0] for d, teams in DIVISIONS.items() for t in teams}
ALL_TEAMS = sorted(TEAM_DIVISION)

TEAM_NAMES = {
    "ARI": "Diamondbacks", "ATL": "Braves", "BAL": "Orioles", "BOS": "Red Sox",
    "CHC": "Cubs", "CHW": "White Sox", "CIN": "Reds", "CLE": "Guardians",
    "COL": "Rockies", "DET": "Tigers", "HOU": "Astros", "KCR": "Royals",
    "LAA": "Angels", "LAD": "Dodgers", "MIA": "Marlins", "MIL": "Brewers",
    "MIN": "Twins", "NYM": "Mets", "NYY": "Yankees", "ATH": "Athletics",
    "PHI": "Phillies", "PIT": "Pirates", "SDP": "Padres", "SEA": "Mariners",
    "SFG": "Giants", "STL": "Cardinals", "TBR": "Rays", "TEX": "Rangers",
    "TOR": "Blue Jays", "WSN": "Nationals",
}
