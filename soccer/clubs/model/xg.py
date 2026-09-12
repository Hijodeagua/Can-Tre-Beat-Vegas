"""
Rolling xG-form features from data/xg_matches.csv (per-match xG for both
sides, Understat, top-5 flights from 2014-15).

One feature reaches the outcome model: `xg_net_diff` — the home side's
rolling xG net (xG for − xG against, mean of its last WINDOW league
matches) minus the away side's. Validated on the 2023-24 holdout (the
last season the backfill covers end-to-end): logistic log loss
0.9662 → 0.9620, +2.1 SE paired — the first form-style feature to survive
testing here, because xG form carries chance-creation information that
neither Elo nor the table has. Same degrade-gracefully contract as the
economics features: no xG (second divisions, MLS, pre-2014, or a club
with fewer than MIN_MATCHES of history) → 0.

STALENESS GUARD: a club's form is only used while its latest xG match is
within MAX_AGE_DAYS of the match being featured; older form is worse than
none (the committed backfill ends 2025-01-04, so without the guard a
2026-27 slate would be scored on Jan-2025 form). The guard spans a summer
break but not a season-long gap, so predictions fall back to Elo-only
until `data/fetch_xg.py` (run by the daily Actions job) has refreshed the
file past the gap.

That fallback is not hypothetical right now: the committed file has not
moved past 2025-01-04, so every live slate is being scored with this
feature at 0. The shot-form layer (`shots.py`, football-data.co.uk) was
added as an independent chance-creation feed for exactly that reason —
it covers the same ground from a publisher that is currently updating.
Reviving this one is a fetcher problem, not a modeling problem.
"""

from pathlib import Path

import pandas as pd

from soccer.clubs.model.form import MatchValues, RollingForm, attach, replay

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
XG_CSV = DATA_DIR / "xg_matches.csv"

WINDOW = 10          # rolling window, league matches
MIN_MATCHES = 5      # form is 0 until a club has this many xG matches
MAX_AGE_DAYS = 130   # spans a summer break; a longer gap voids the form

XG_FEATURES = ["xg_net_diff"]


def xg_available() -> bool:
    return XG_CSV.exists()


def load_xg() -> pd.DataFrame:
    return pd.read_csv(XG_CSV)


# The rolling window, warm-up minimum, staleness guard and pre-match
# attach loop are shared with the shot-form layer; see model/form.py.
class _Form(RollingForm):
    """xG form with this module's tuning baked in."""

    def __init__(self) -> None:
        super().__init__(WINDOW, MIN_MATCHES, MAX_AGE_DAYS)


def match_values() -> MatchValues:
    """(league, date, home, away) -> (home xG, away xG)."""
    if not xg_available():
        return {}
    return {
        (r.league, r.date, r.home_team, r.away_team): (r.xg_home, r.xg_away)
        for r in load_xg().itertuples()
    }


def attach_xg(history: pd.DataFrame) -> pd.DataFrame:
    """Add `xg_net_diff` to a replay-history table (strictly pre-match:
    each row's feature uses only xG matches dated before it). Rows from
    leagues or eras the xG file doesn't cover get 0."""
    return attach(history, "xg_net_diff", match_values(), _Form)


def current_form() -> RollingForm:
    """Form state after every committed xG match — what the daily slate
    features against (with the same staleness guard applied at query
    time via `RollingForm.net`)."""
    return replay(match_values(), _Form)


def slate_diff(form: RollingForm, league: str, home: str, away: str,
               date: str) -> float:
    return form.diff(league, home, away, date)
