"""
Rolling shot-form features from data/shots_matches.csv (shots and shots on
target for both sides, football-data.co.uk, top-5 flights from each
league's first season in results.csv).

One feature reaches the outcome model: `sot_net_diff` — the home side's
rolling shots-on-target net (on target for − on target against, mean of
its last WINDOW league matches) minus the away side's.

Validated on the 2024-25 + 2025-26 holdout, added on top of the full
existing feature set (Elo gap + squad economics + xG form): logistic log
loss 1.0199 → 1.0180, +2.9 SE paired, and it improves every one of the
three season splits tried (2023-24 +0.0016, 2024-25 +0.0019, 2025-26
+0.0027). Total shots work too (+2.8 SE) but add essentially nothing on
top of shots on target, so only the on-target feature ships — volume
without the on-target filter is mostly noise about shot selection.

Two things this buys over the xG feature it sits next to:

- COVERAGE. Shots reach back to each league's first season rather than
  Understat's 2014-15, so 41.5% of training rows carry shot form against
  32.2% carrying xG form.
- LIVENESS. Understat has not updated since 2025-01-04, so `xg.py`'s
  staleness guard is currently zeroing the xG feature on every live
  slate. On the same holdout, Elo + economics + shot form (1.0181) beats
  Elo + economics + xG form (1.0199) — the shots layer is, for now, the
  chance-creation signal that is actually reaching production. Refreshing
  Understat would make them complements again rather than substitutes;
  the two features are independent and both stay in the model.

Same degrade-gracefully contract as every optional feature: no shots
(second divisions, MLS, or a club short of MIN_MATCHES) → 0.
"""

from pathlib import Path

import pandas as pd

from soccer.clubs.model.form import MatchValues, RollingForm, attach, replay

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SHOTS_CSV = DATA_DIR / "shots_matches.csv"

WINDOW = 10          # rolling window, league matches
MIN_MATCHES = 5      # form is 0 until a club has this many shot matches
MAX_AGE_DAYS = 130   # spans a summer break; a longer gap voids the form

SHOT_FEATURES = ["sot_net_diff"]


def shots_available() -> bool:
    return SHOTS_CSV.exists()


def load_shots() -> pd.DataFrame:
    return pd.read_csv(SHOTS_CSV)


# Window, warm-up, staleness guard and the pre-match attach loop are
# shared with the xG layer; see model/form.py.
class _Form(RollingForm):
    """Shot form with this module's tuning baked in."""

    def __init__(self) -> None:
        super().__init__(WINDOW, MIN_MATCHES, MAX_AGE_DAYS)


def match_values() -> MatchValues:
    """(league, date, home, away) -> (home SoT, away SoT)."""
    if not shots_available():
        return {}
    return {
        (r.league, r.date, r.home_team, r.away_team): (r.sot_home, r.sot_away)
        for r in load_shots().itertuples()
    }


def attach_shots(history: pd.DataFrame) -> pd.DataFrame:
    """Add `sot_net_diff` to a replay-history table (strictly pre-match)."""
    return attach(history, "sot_net_diff", match_values(), _Form)


def current_form() -> RollingForm:
    """Form state after every committed shot match."""
    return replay(match_values(), _Form)


def slate_diff(form: RollingForm, league: str, home: str, away: str,
               date: str) -> float:
    return form.diff(league, home, away, date)
