"""
Score model for the daily pipeline: turn a pre-match Elo expectation into a
pair of Poisson goal rates.

Calibration is refit from the engine's own league history each run (same
posture as the MLB score model): expected goal margin is a linear map from
the Elo home expectancy, fit by least squares; the expected total comes
from the league's last `GOAL_RATE_SEASONS` completed seasons. The two
combine into per-side rates λ_home = (total + margin) / 2 and
λ_away = (total − margin) / 2.

Independent Poisson is deliberate: on this dataset the observed 1-1 rate
(11.7%) matches the independent-Poisson prediction (11.8%) and 0-0 is only
0.6pp underweight, so a Dixon–Coles correction isn't buying anything yet.
"""

from dataclasses import dataclass
from math import factorial

import numpy as np
import pandas as pd

from soccer.clubs.daily.config import GOAL_RATE_SEASONS, MAX_GOALS, MIN_LAMBDA


@dataclass
class ScoreParams:
    margin_a: float                    # margin = a + b * exp_home
    margin_b: float
    league_total: dict[str, float]     # league -> expected goals per match

    def lambdas(self, league: str, exp_home: float) -> tuple[float, float]:
        total = self.league_total[league]
        margin = self.margin_a + self.margin_b * exp_home
        lam_h = max(MIN_LAMBDA, (total + margin) / 2.0)
        lam_a = max(MIN_LAMBDA, (total - margin) / 2.0)
        return lam_h, lam_a


def fit(history: pd.DataFrame) -> ScoreParams:
    """Calibrate on league rows of a (glued) replay history."""
    league = history[~history["league"].str.startswith("uefa:")]
    margin = league["home_score"] - league["away_score"]
    b, a = np.polyfit(league["exp_home"], margin, 1)

    totals = {}
    for lg, sub in league.groupby("league"):
        recent = sorted(sub["season"].unique())[-GOAL_RATE_SEASONS:]
        r = sub[sub["season"].isin(recent)]
        totals[lg] = float((r["home_score"] + r["away_score"]).mean())
    return ScoreParams(float(a), float(b), totals)


def score_grid(lam_h: float, lam_a: float) -> np.ndarray:
    """Joint P(home=i, away=j) on [0, MAX_GOALS]^2, renormalized."""
    g = np.arange(MAX_GOALS + 1)
    fact = np.array([factorial(k) for k in g], dtype=float)
    ph = np.exp(-lam_h) * lam_h ** g / fact
    pa = np.exp(-lam_a) * lam_a ** g / fact
    grid = np.outer(ph, pa)
    return grid / grid.sum()


def most_likely_score(lam_h: float, lam_a: float) -> tuple[int, int]:
    grid = score_grid(lam_h, lam_a)
    i, j = np.unravel_index(int(grid.argmax()), grid.shape)
    return int(i), int(j)
