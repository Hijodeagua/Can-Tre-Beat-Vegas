"""Paths and defaults for the college-football daily pipeline — the CFB
sibling of `mlb/daily/config.py` and `soccer/clubs/daily/config.py`."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CFB_DIR = REPO_ROOT / "CFB"
ARTIFACTS = CFB_DIR / "model" / "artifacts"

DATA_DIR = REPO_ROOT / "data" / "college_football"
GAMES_CSV = DATA_DIR / "games.csv"

# Persisted slates + graded ledger, mirroring data/mlb/predictions/ and
# data/soccer_clubs/predictions/.
PREDICTIONS_DIR = DATA_DIR / "predictions"
GRADES_CSV = PREDICTIONS_DIR / "grades.csv"

# Site JSON, next to the MLB and soccer cards' data.
SITE_DIR = REPO_ROOT / "web" / "public" / "data" / "cfb"
SITE_LATEST = SITE_DIR / "latest.json"
SITE_HISTORY = SITE_DIR / "history"

# Email HTML + manifest + send ledger, mirroring reports/soccer/.
EMAIL_REPORTS_DIR = REPO_ROOT / "reports" / "cfb"

# Games dated [D, D + SLATE_WINDOW_DAYS) (US/Eastern kickoff date) make the
# slate for a run dated D. Two days, like soccer: a Friday run previews
# Friday + Saturday, Saturday's run re-predicts Saturday from the same
# ratings (grading is idempotent by game, so overlap is harmless).
SLATE_WINDOW_DAYS = 2

# Twice-weekly update email, same weekdays as the soccer email (Mon=0):
# Monday wraps the weekend's games, Thursday previews the coming ones.
EMAIL_WEEKDAYS = (0, 3)
EMAIL_FIXTURE_DAYS = 7      # "games this week" horizon for the email
ROLLING_WINDOWS = (7, 30)   # rolling performance windows, in days

# Rest-of-season Monte Carlo. Vectorized across sims (like the MLB sim),
# so 10k replays of ~800 remaining games take a few seconds.
SEASON_SIMS = 10000
BOWL_ELIGIBLE_WINS = 6
TOP_N = 25

# Always-pick-home reference forecast for the paired grading baseline (the
# same device as mlb/daily's ALWAYS_HOME_P): the home win rate over every
# non-neutral FBS-involved game 2015-2025 (0.632 across 8,820 games),
# frozen here so the baseline never peeks at the games it is scoring.
# Neutral-site games get a coin flip.
ALWAYS_HOME_P = 0.632
NEUTRAL_HOME_P = 0.5

# Score model: the margin map is fit on this many most recent completed
# seasons of the engine's own history; scoring rates (points per game) on
# fewer, since the scoring environment drifts.
MARGIN_FIT_SEASONS = 10
TOTAL_SEASONS = 2
