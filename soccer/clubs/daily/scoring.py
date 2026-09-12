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

Turning the grid back into ONE scoreline to publish is a separate problem
from modeling it, and the obvious answer is the wrong one. The single most
likely cell of a soccer score grid is a low draw almost regardless of who
is playing: at league-average rates 1-1 is the modal cell until the
favorite's edge is enormous, so an unconditional argmax printed 1-1 for
65% of the slate and contradicted the model's own pick on most of those
rows — "Pick: Manchester City / Score: 1-1". Draws are ~25% of real
results, so that display was wrong about the league as well as about
itself. `representative_score` conditions on the picked outcome instead:
the most likely exact scoreline **given** that side wins (or given a
draw). See its docstring for why that is the right conditional.
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


def outcome_mask(outcome: str) -> np.ndarray:
    """Boolean [0, MAX_GOALS]^2 mask of the cells consistent with H/D/A."""
    g = np.arange(MAX_GOALS + 1)
    home, away = np.meshgrid(g, g, indexing="ij")
    if outcome == "H":
        return home > away
    if outcome == "A":
        return home < away
    return home == away


def most_likely_score(lam_h: float, lam_a: float) -> tuple[int, int]:
    """Modal cell of the whole grid — the unconditional most likely exact
    score. Kept because it is the honest answer to "what one scoreline is
    likeliest", but it is not what the slate publishes: see
    `representative_score`."""
    grid = score_grid(lam_h, lam_a)
    i, j = np.unravel_index(int(grid.argmax()), grid.shape)
    return int(i), int(j)


def representative_score(lam_h: float, lam_a: float,
                         outcome: str) -> tuple[int, int, float]:
    """The scoreline to publish next to a pick of `outcome`: the most
    likely exact score among the scores that produce that outcome, with
    its unconditional probability.

    This is the exact form of "simulate the match a lot of times, throw
    away the sims that disagree with the pick, and report the scoreline
    that came up most often" — the conditional modal scoreline. Computing
    it off the grid rather than by sampling just removes the sampling
    noise: a 10-run Monte Carlo picks its answer from 10 draws of a
    distribution whose modal cell only holds ~11% of the mass, so it would
    disagree with itself run to run, and averaging away that noise by
    raising the sim count only converges back on the same unconditional
    1-1 that made this worth fixing. Conditioning is what fixes it, not
    the number of sims.

    The returned probability is unconditional (P of exactly this score,
    not P given the outcome), so it can be read next to p_H / p_D / p_A
    without rescaling.
    """
    grid = score_grid(lam_h, lam_a)
    masked = np.where(outcome_mask(outcome), grid, 0.0)
    i, j = np.unravel_index(int(masked.argmax()), grid.shape)
    return int(i), int(j), float(grid[i, j])
