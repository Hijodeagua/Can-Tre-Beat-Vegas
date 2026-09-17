"""
Rolling goal-form features from the result log itself
(`soccer/clubs/data/results.csv` — every score the Elo replay already
reads).

One candidate feature: `goals_net_diff` — the home side's rolling goal
difference (goals for − goals against, mean of its last WINDOW league
matches) minus the away side's.

Why it is worth trying at all, and why it might well fail: unlike the xG
and shots layers, this carries no information the result log doesn't
already have, and the Elo engine is *built* from those results with a
margin-of-victory multiplier. So the honest prior is that most of this
signal is already inside `elo_gap`, and what is left is the part Elo
throws away — the ln-damping and the 4/5-goal cap, plus the fact that Elo
weights a result by opponent strength while raw goal difference does not.
Whether that residue predicts anything is a measurement, not an opinion:
see `python -m soccer.clubs.model.eval_goals_form`.

What it does have over both chance-creation feeds is coverage and
liveness: every league and every season in results.csv, updated by the
same daily fetch that drives the ratings. No upstream to go stale.

Same degrade-gracefully contract as the rest: a club short of
MIN_MATCHES, or whose last covered match is older than MAX_AGE_DAYS,
contributes 0 rather than half a differential.
"""

from pathlib import Path

import pandas as pd

from soccer.clubs.model.form import MatchValues, RollingForm, attach, replay

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_CSV = DATA_DIR / "results.csv"

WINDOW = 10          # rolling window, league matches — same as the other feeds
MIN_MATCHES = 5      # form is 0 until a club has this many results
MAX_AGE_DAYS = 130   # spans a summer break; a longer gap voids the form

GOAL_FEATURES = ["goals_net_diff"]


def goals_available() -> bool:
    return RESULTS_CSV.exists()


def load_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS_CSV)


# Window, warm-up, staleness guard and the pre-match attach loop are
# shared with the xG and shots layers; see model/form.py.
class _Form(RollingForm):
    """Goal form with this module's tuning baked in."""

    def __init__(self) -> None:
        super().__init__(WINDOW, MIN_MATCHES, MAX_AGE_DAYS)


def match_values() -> MatchValues:
    """(league, date, home, away) -> (home goals, away goals). Only played
    matches: a fixture with no score yet pushes nothing."""
    if not goals_available():
        return {}
    played = load_results().dropna(subset=["home_score", "away_score"])
    return {
        (r.league, r.date, r.home_team, r.away_team): (float(r.home_score),
                                                       float(r.away_score))
        for r in played.itertuples()
    }


def attach_goals(history: pd.DataFrame) -> pd.DataFrame:
    """Add `goals_net_diff` to a replay-history table (strictly pre-match)."""
    return attach(history, "goals_net_diff", match_values(), _Form)


def current_form() -> RollingForm:
    """Form state after every committed result."""
    return replay(match_values(), _Form)


def slate_diff(form: RollingForm, league: str, home: str, away: str,
               date: str) -> float:
    return form.diff(league, home, away, date)
