"""
Squad-value Elo anchor: places an unglued league (MLS today; any future
non-UEFA league tomorrow — see the `glued` field on `League`) onto the
glued leagues' shared Elo scale by regression, since it has no competitive
matches against them to calibrate against directly.

Method: fit ln(squad value €m) -> Elo across every club in the glued
pools that has both a current rating and a market-value upload, then
apply that same line to the unglued league's own squad values. Money
turns out to predict club strength tightly within the glued pools
(R^2 ~ 0.7 on ~125 clubs as of 2026) — good enough for "which tier is
this league roughly playing at", not for individual-match confidence.

This is deliberately a coarse anchor, not a measurement: every anchored
number the exporters publish carries the fit's R^2, residual spread and
club count so the site and emails can label it honestly rather than
presenting it as equivalent to a glued rating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# Below this many clubs, a log-value/Elo line is more noise than signal —
# report no anchor rather than a false-precision one. ~125 clubs is what
# the ten glued leagues give with today's market_values coverage.
MIN_FIT_CLUBS = 30


@dataclass(frozen=True)
class ValueEloFit:
    intercept: float
    slope: float
    n_clubs: int
    r2: float | None
    residual_std_elo: float


def fit_glued_value_elo(
    ratings: dict, values: pd.DataFrame, glued_leagues: Iterable[str]
) -> ValueEloFit | None:
    """Regress glued-pool club Elo on ln(squad value), one row per club
    using its most recent value-upload season. `ratings` is
    export_site.ratings_payload()'s output; `values` is
    features.load_market_values_raw()'s raw table."""
    if values.empty:
        return None
    glued_leagues = set(glued_leagues)
    sub = values[values["league"].isin(glued_leagues)]
    if sub.empty:
        return None
    latest = (
        sub.sort_values("season")
        .groupby(["league", "club"], as_index=False)
        .tail(1)
    )

    elo_by_club = {
        (league, c["team"]): c["elo"]
        for league, table in ratings.items()
        if league in glued_leagues
        for c in table.get("clubs", [])
    }

    xs, ys = [], []
    for r in latest.itertuples():
        elo = elo_by_club.get((r.league, r.club))
        value = getattr(r, "squad_value_eur_m", None)
        if elo is None or value is None or not (value > 0):
            continue
        xs.append(math.log(value))
        ys.append(elo)

    if len(xs) < MIN_FIT_CLUBS:
        return None

    x = np.array(xs)
    y = np.array(ys)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = round(1 - ss_res / ss_tot, 3) if ss_tot > 0 else None

    return ValueEloFit(
        intercept=float(intercept),
        slope=float(slope),
        n_clubs=len(xs),
        r2=r2,
        residual_std_elo=round(float(resid.std()), 1),
    )


def anchor_elo(fit: ValueEloFit, squad_value_eur_m: float | None) -> float | None:
    """One club's value-implied Elo on the glued scale, or None for a
    missing/non-positive value."""
    if squad_value_eur_m is None or not (squad_value_eur_m > 0):
        return None
    return fit.intercept + fit.slope * math.log(squad_value_eur_m)


def anchor_clubs(fit: ValueEloFit, squad_values: Iterable[float]) -> list[float]:
    """Value-implied Elo for a list of clubs, dropping any with no usable
    value — the caller decides what to do with a short/empty result."""
    out = []
    for v in squad_values:
        e = anchor_elo(fit, v)
        if e is not None:
            out.append(round(e, 1))
    return out
