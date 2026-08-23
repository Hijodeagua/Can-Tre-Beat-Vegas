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
"""

from collections import deque
from pathlib import Path

import pandas as pd

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


class _Form:
    """Per-(league, team) rolling xG state, fed matches in date order."""

    def __init__(self) -> None:
        self.hist: dict[tuple, deque] = {}
        self.last_date: dict[tuple, str] = {}

    def net(self, league: str, team: str, asof: str) -> float | None:
        key = (league, team)
        q = self.hist.get(key)
        if q is None or len(q) < MIN_MATCHES:
            return None
        age = (pd.Timestamp(asof) - pd.Timestamp(self.last_date[key])).days
        if age > MAX_AGE_DAYS:
            return None
        return sum(f - a for f, a in q) / len(q)

    def push(self, league: str, home: str, away: str, date: str,
             xg_home: float, xg_away: float) -> None:
        for team, f, a in ((home, xg_home, xg_away), (away, xg_away, xg_home)):
            self.hist.setdefault((league, team), deque(maxlen=WINDOW)).append((f, a))
            self.last_date[(league, team)] = date


def _diff(form: _Form, league: str, home: str, away: str, date: str) -> float:
    h = form.net(league, home, date)
    a = form.net(league, away, date)
    if h is None or a is None:
        return 0.0   # one-sided form would bias the differential
    return h - a


def attach_xg(history: pd.DataFrame) -> pd.DataFrame:
    """Add `xg_net_diff` to a replay-history table (strictly pre-match:
    each row's feature uses only xG matches dated before it). Rows from
    leagues or eras the xG file doesn't cover get 0."""
    history = history.copy()
    if not xg_available():
        history["xg_net_diff"] = 0.0
        return history

    xg = load_xg()
    xg_by_key = {
        (r.league, r.date, r.home_team, r.away_team): (r.xg_home, r.xg_away)
        for r in xg.itertuples()
    }
    order = history["date"].astype(str).argsort(kind="stable")
    form = _Form()
    vals = pd.Series(0.0, index=history.index)
    for i in order:
        row = history.iloc[i]
        vals.iloc[i] = _diff(form, row["league"], row["home_team"],
                             row["away_team"], row["date"])
        hit = xg_by_key.get((row["league"], row["date"],
                             row["home_team"], row["away_team"]))
        if hit is not None:
            form.push(row["league"], row["home_team"], row["away_team"],
                      row["date"], *hit)
    history["xg_net_diff"] = vals
    return history


def current_form() -> _Form:
    """Form state after every committed xG match — what the daily slate
    features against (with the same staleness guard applied at query
    time via `_Form.net`)."""
    form = _Form()
    if not xg_available():
        return form
    for r in load_xg().sort_values("date").itertuples():
        form.push(r.league, r.home_team, r.away_team, r.date, r.xg_home, r.xg_away)
    return form


def slate_diff(form: _Form, league: str, home: str, away: str, date: str) -> float:
    return _diff(form, league, home, away, date)
