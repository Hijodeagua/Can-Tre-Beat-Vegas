"""Paths and defaults for the club-soccer daily pipeline."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CLUBS_DIR = REPO_ROOT / "soccer" / "clubs"
ARTIFACTS = CLUBS_DIR / "model" / "artifacts"

# Persisted slates + graded ledger, mirroring data/mlb/predictions/.
PREDICTIONS_DIR = REPO_ROOT / "data" / "soccer_clubs" / "predictions"
GRADES_CSV = PREDICTIONS_DIR / "grades.csv"

# Site JSON, next to the MLB card's data.
SITE_DIR = REPO_ROOT / "web" / "public" / "data" / "soccer"
SITE_LATEST = SITE_DIR / "latest.json"
SITE_HISTORY = SITE_DIR / "history"

# Matches dated [D, D + SLATE_WINDOW_DAYS) make the slate for a run dated D.
SLATE_WINDOW_DAYS = 2

# Rest-of-season Monte Carlo.
SEASON_SIMS = 1000
MAX_GOALS = 8            # Poisson scoreline grid is [0, MAX_GOALS] per side
MIN_LAMBDA = 0.2
RELEGATION_SPOTS = 3     # bottom-3 = drop zone (incl. any playoff spot)
UCL_SPOTS = 4

# League goal rates are estimated from this many most recent completed
# seasons (COVID-era home rates argue against reaching further back).
GOAL_RATE_SEASONS = 2
