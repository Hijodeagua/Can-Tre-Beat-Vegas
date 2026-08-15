"""Self-tuning calibration for the score model's expected totals.

Closes the predict -> grade -> refit loop: every run reconstructs the
current season's *raw* matchup-total predictions walk-forward (the same
TeamRates code the slate uses, each game predicted from games strictly
before it - no leakage, no dependence on what happened to be emailed), fits

    actual_total = a + b * raw_predicted_total

and stores the fit in data/mlb/score_calibration.json. The daily slate then
applies T' = a + b*T_raw to its totals, so systematic bias (a scoring
environment shift, an over/under-confident spread) corrects itself with a
one-day lag. The fourth daily email reports the fit and the realized errors.

Guards: the correction stays identity until the season sample reaches
MIN_GAMES, and the slope is clamped to [0.5, 1.5] so a weird stretch can
never flip or explode the totals. Fitting raw predictions (never calibrated
ones) keeps the loop free of feedback circularity.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import pandas as pd

from mlb.daily.config import CURRENT_SEASON, REPO
from mlb.daily.scoring import TOTAL_MAX, TOTAL_MIN, TeamRates

CALIBRATION_JSON = REPO / "data" / "mlb" / "score_calibration.json"

MIN_GAMES = 300
SLOPE_MIN, SLOPE_MAX = 0.5, 1.5


@dataclass
class Calibration:
    a: float = 0.0
    b: float = 1.0
    n: int = 0
    mae_raw: float | None = None
    mae_calibrated: float | None = None
    mae_constant: float | None = None
    bias_raw: float | None = None   # mean(pred - actual); + = over-predicting
    applied: bool = False

    def apply(self, total: float) -> float:
        if not self.applied:
            return total
        return float(min(max(self.a + self.b * total, TOTAL_MIN), TOTAL_MAX))


def season_pairs(games: pd.DataFrame,
                 season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Walk-forward raw predicted vs actual totals for one season."""
    g = games.sort_values(["date", "game_num"])
    rates = TeamRates()
    rows = []
    for row in g.itertuples(index=False):
        if row.season == season:
            rows.append({
                "date": row.date,
                "pred": rates.matchup_total(row.home_fr, row.away_fr),
                "constant": 2.0 * rates.league_mean,
                "actual": float(row.home_score + row.away_score),
            })
        rates.observe(row.home_fr, row.away_fr,
                      float(row.home_score), float(row.away_score))
    return pd.DataFrame(rows)


def fit(games: pd.DataFrame, season: int = CURRENT_SEASON) -> Calibration:
    pairs = season_pairs(games, season)
    n = len(pairs)
    if n < MIN_GAMES:
        return Calibration(n=n)

    x, y = pairs.pred, pairs.actual
    b = float(((x - x.mean()) * (y - y.mean())).sum()
              / ((x - x.mean()) ** 2).sum())
    b = min(max(b, SLOPE_MIN), SLOPE_MAX)
    a = float(y.mean() - b * x.mean())

    calibrated = (a + b * x).clip(TOTAL_MIN, TOTAL_MAX)
    return Calibration(
        a=round(a, 4), b=round(b, 4), n=n,
        mae_raw=round(float((x - y).abs().mean()), 4),
        mae_calibrated=round(float((calibrated - y).abs().mean()), 4),
        mae_constant=round(float((pairs.constant - y).abs().mean()), 4),
        bias_raw=round(float((x - y).mean()), 4),
        applied=True,
    )


def save(cal: Calibration) -> None:
    CALIBRATION_JSON.write_text(
        json.dumps(asdict(cal), indent=1), encoding="utf-8"
    )


def load() -> Calibration:
    if not CALIBRATION_JSON.exists():
        return Calibration()
    return Calibration(**json.loads(CALIBRATION_JSON.read_text()))
