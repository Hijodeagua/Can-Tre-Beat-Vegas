"""
Rolling form measured against the rating rather than in absolute terms:
`surprise_net_diff` — the home side's mean (actual − Elo-expected) result
over its last WINDOW league matches, minus the away side's.

This is the form feature worth testing next to `goals.py`, and the reason
is the same reason raw goal form struggles. The Elo engine is built from
results, so a club's recent goal difference is largely *already* inside
`elo_gap`; a rolling average of it re-reads what the rating has read. A
surprise is different: it is the part of each result the rating did not
expect (`actual_home − exp_home`, the Elo update's own error term).

That makes this an explicit momentum test, and an honest one, because the
null is well defined. Elo already prices each surprise exactly once and
permanently — the rating moved by K × (actual − expected) when it
happened. The question here is whether a *run* of surprises predicts more
than that single adjustment already captured: whether form persists
beyond what the rating took from it. The answer is allowed to be no, and
`eval_goals_form.py` is where it gets asked.

No new data: the replay history already carries `exp_home` and
`actual_home` for every match, so this reaches every league and season
the ratings do. Same degrade-gracefully contract as the other form
layers — a club short of MIN_MATCHES, or whose last match is older than
MAX_AGE_DAYS, contributes 0.
"""

from __future__ import annotations

import pandas as pd

from soccer.clubs.model.form import MatchValues, RollingForm, attach

WINDOW = 10          # rolling window, league matches — same as the other feeds
MIN_MATCHES = 5      # form is 0 until a club has this many results
MAX_AGE_DAYS = 130   # spans a summer break; a longer gap voids the form

MOMENTUM_FEATURES = ["surprise_net_diff"]


class _Form(RollingForm):
    """Surprise form with this module's tuning baked in."""

    def __init__(self) -> None:
        super().__init__(WINDOW, MIN_MATCHES, MAX_AGE_DAYS)


def match_values(history: pd.DataFrame) -> MatchValues:
    """(league, date, home, away) -> (home surprise, away surprise).

    Split symmetrically so `RollingForm.net`'s (for − against) mean comes
    out as the club's own mean surprise: the home side is credited with
    half the surprise and the away side debited the same half, because one
    result is one piece of evidence about both.
    """
    if "exp_home" not in history or "actual_home" not in history:
        return {}
    out: MatchValues = {}
    for r in history.itertuples():
        half = (float(r.actual_home) - float(r.exp_home)) / 2.0
        out[(r.league, r.date, r.home_team, r.away_team)] = (half, -half)
    return out


def attach_momentum(history: pd.DataFrame) -> pd.DataFrame:
    """Add `surprise_net_diff` to a replay-history table (strictly
    pre-match: `attach` features a row before pushing its own result)."""
    return attach(history, "surprise_net_diff", match_values(history), _Form)


def current_form(history: pd.DataFrame) -> RollingForm:
    """Form state after every match in `history` — what a live slate
    features against, with the staleness guard still applied at query
    time."""
    form = _Form()
    values = match_values(history)
    for key in sorted(values, key=lambda k: k[1]):
        league, date, home, away = key
        form.push(league, home, away, date, *values[key])
    return form


def slate_diff(form: RollingForm, league: str, home: str, away: str,
               date: str) -> float:
    return form.diff(league, home, away, date)
