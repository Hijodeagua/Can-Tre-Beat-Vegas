"""Paths and defaults for the NFL daily pipeline — the NFL sibling of
`CFB/daily/config.py`, `mlb/daily/config.py` and
`soccer/clubs/daily/config.py`."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NFL_DIR = REPO_ROOT / "NFL"
ARTIFACTS = NFL_DIR / "elo" / "artifacts"

# The spine is the nflverse schedule the LightGBM model already reads;
# `NFL/model/schedule.py --refresh` re-downloads it.
GAMES_CSV = REPO_ROOT / "data" / "schedules" / "nflverse_games.csv"

# Persisted slates + graded ledger, mirroring data/college_football/predictions/.
DATA_DIR = REPO_ROOT / "data" / "nfl"
PREDICTIONS_DIR = DATA_DIR / "predictions"
GRADES_CSV = PREDICTIONS_DIR / "grades.csv"

# Site JSON, next to the MLB, soccer and CFB cards' data.
SITE_DIR = REPO_ROOT / "web" / "public" / "data" / "nfl"
SITE_LATEST = SITE_DIR / "latest.json"
SITE_HISTORY = SITE_DIR / "history"

# Email HTML + manifest + send ledger, mirroring reports/cfb/.
EMAIL_REPORTS_DIR = REPO_ROOT / "reports" / "nfl"

# The slate for a run dated D is the next NFL week: every unplayed game
# dated on or after D in the earliest such week. An NFL week runs
# Thursday-Monday, so a Tuesday run previews the whole week, a Sunday run
# re-predicts what's left of it, and grading (idempotent by game id,
# earliest persisted prediction wins) locks each pick the first morning
# it appears on a slate.
EMAIL_WEEKDAYS = (1, 3)     # Tue = the week is final after MNF; Thu = before TNF
ROLLING_WINDOWS = (7, 30)   # rolling performance windows, in days

# Rest-of-season Monte Carlo, vectorized across sims like the CFB sim:
# 10k replays of a 272-game season plus the bracket take a few seconds.
SEASON_SIMS = 10000

# Always-pick-home reference forecast for the paired grading baseline: the
# home win rate over every non-neutral game 2015-2025 (0.549 across 2,967
# games, ties counted as half), frozen here so the baseline never peeks
# at the games it is scoring. Neutral sites get a coin flip.
ALWAYS_HOME_P = 0.55
NEUTRAL_HOME_P = 0.5

# Score model: the margin map is fit on this many most recent completed
# seasons of the engine's own history; scoring rates on fewer, since the
# scoring environment drifts.
MARGIN_FIT_SEASONS = 10
TOTAL_SEASONS = 2
