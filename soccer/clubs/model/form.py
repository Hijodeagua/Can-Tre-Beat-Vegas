"""
Shared rolling-form machinery for the per-match performance feeds (xG from
Understat, shots from football-data.co.uk).

Both feeds answer the same question in the same shape — "how has each side
been creating and conceding chances lately, and what is the home-minus-away
differential?" — so the window, the warm-up minimum, the staleness guard
and the strictly-pre-match attach loop live here once and each feed
supplies only its own file, columns and tuning.

Three rules are the reason this is a module rather than a groupby:

- STRICTLY PRE-MATCH. A row's feature is computed from matches dated
  before it and then the row's own performance is pushed into the state.
  Nothing else keeps a rolling feature from leaking the result it is
  supposed to predict.
- WARM-UP. A club with fewer than `min_matches` in the window has no
  form, and a differential where only one side has form is 0, not a
  one-sided lean — half a differential is a bias, not a signal.
- STALENESS. Form is used only while the club's last covered match is
  within `max_age_days`. A feed that stops updating (Understat's has been
  stale since 2025-01-04) would otherwise keep scoring today's slate on
  year-old form, which is worse than no feature at all. The guard spans a
  summer break but not a season-long gap, so a dead feed degrades to
  Elo-only on its own instead of quietly poisoning the model.
"""

from collections import deque
from typing import Callable

import pandas as pd


class RollingForm:
    """Per-(league, team) rolling (for, against) state, fed in date order."""

    def __init__(self, window: int, min_matches: int, max_age_days: int) -> None:
        self.window = window
        self.min_matches = min_matches
        self.max_age_days = max_age_days
        self.hist: dict[tuple, deque] = {}
        self.last_date: dict[tuple, str] = {}

    def net(self, league: str, team: str, asof: str) -> float | None:
        """Mean (for − against) over the window, or None when the club is
        short of the warm-up minimum or its form has gone stale."""
        key = (league, team)
        q = self.hist.get(key)
        if q is None or len(q) < self.min_matches:
            return None
        age = (pd.Timestamp(asof) - pd.Timestamp(self.last_date[key])).days
        if age > self.max_age_days:
            return None
        return sum(f - a for f, a in q) / len(q)

    def push(self, league: str, home: str, away: str, date: str,
             home_val: float, away_val: float) -> None:
        for team, f, a in ((home, home_val, away_val), (away, away_val, home_val)):
            self.hist.setdefault((league, team), deque(maxlen=self.window)).append((f, a))
            self.last_date[(league, team)] = date

    def diff(self, league: str, home: str, away: str, date: str) -> float:
        """Home form minus away form; 0 unless both sides have form."""
        h = self.net(league, home, date)
        a = self.net(league, away, date)
        if h is None or a is None:
            return 0.0
        return h - a


# (league, date, home_team, away_team) -> (home_value, away_value)
MatchValues = dict[tuple, tuple[float, float]]


def attach(history: pd.DataFrame, column: str, values: MatchValues,
           make_form: Callable[[], RollingForm]) -> pd.DataFrame:
    """Add `column` to a replay-history table, strictly pre-match.

    Rows from leagues or eras `values` doesn't cover simply never push
    anything, so their clubs never reach the warm-up minimum and the
    feature stays 0 — the graceful degradation every optional feature in
    this model contracts for.
    """
    history = history.copy()
    if not values:
        history[column] = 0.0
        return history

    order = history["date"].astype(str).argsort(kind="stable")
    form = make_form()
    vals = pd.Series(0.0, index=history.index)
    for i in order:
        row = history.iloc[i]
        vals.iloc[i] = form.diff(row["league"], row["home_team"],
                                 row["away_team"], row["date"])
        hit = values.get((row["league"], row["date"],
                          row["home_team"], row["away_team"]))
        if hit is not None:
            form.push(row["league"], row["home_team"], row["away_team"],
                      row["date"], *hit)
    history[column] = vals
    return history


def replay(values: MatchValues, make_form: Callable[[], RollingForm]) -> RollingForm:
    """Form state after every covered match — what a live slate features
    against, with the staleness guard still applied at query time."""
    form = make_form()
    for key in sorted(values, key=lambda k: k[1]):
        league, date, home, away = key
        form.push(league, home, away, date, *values[key])
    return form
