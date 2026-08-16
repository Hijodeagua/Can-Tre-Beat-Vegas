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

STARTS_CSV = REPO / "data" / "mlb" / "pitcher_starts.csv"

# --- Model versions ---------------------------------------------------------
# The active model is the graded record; the shadow model (if any) is
# predicted, persisted, and graded in its own bucket so the live record is
# never contaminated. Cutover = swap ACTIVE_MODEL/SHADOW_MODEL.
MODEL_V1 = "v1-team-elo"
MODEL_V2 = "v2-sp"
# Cut over 2026-08-15: skipped the 7-day shadow soak (v1's live record was
# exploratory/noise-dominated at n=94 anyway - see docs/SP-AUDIT.md) and
# promoted v2-sp straight to active. v1's ledger freezes wherever it last
# graded and stays visible on the model card as the pre-change record
# (predictions_dir() keeps it in the original flat bucket permanently, so
# nothing about this flip touches it). Set SHADOW_MODEL = MODEL_V1 instead
# of None below to keep running v1 in parallel for an ongoing comparison.
ACTIVE_MODEL = MODEL_V2
SHADOW_MODEL = None   # set to a model name to run it in parallel

# v2 starting-pitcher adjustment knobs (tuned on 2012-2021 only; see
# research/SP-BACKTEST.md - 538's published C=4.7/half-life 10 was not
# distinguishable from the current model out of sample, this pair was).
SP_C = 3.0
SP_HALF_LIFE = 20.0
USE_PITCHER_ADJ = True
USE_REST = True
USE_TRAVEL = True

# Always-pick-home reference forecast for the paired grading baseline: the
# long-run MLB home win rate over 2012-2025 (0.5335 across 32,484 games),
# frozen here so the baseline never peeks at the games it is scoring.
ALWAYS_HOME_P = 0.534


def predictions_dir(model_version: str) -> Path:
    """Prediction/ledger bucket for a model version. v1 keeps the original
    flat layout so the pre-change record stays where it always lived."""
    if model_version == MODEL_V1:
        return PREDICTIONS_DIR
    return PREDICTIONS_DIR / model_version

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
