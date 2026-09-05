"""Score model for the daily pipeline: turn a pre-game Elo edge into an
expected final score.

Two pieces, both refit from the engine's own replay history every run
(same posture as the MLB and club-soccer score models — no pickle drift):

- **Expected margin** is linear in the Elo difference (home advantage
  included): margin = a + b * elo_diff, fit by least squares on the last
  MARGIN_FIT_SEASONS seasons. 1/b is the engine's own "Elo per point"
  (plan §8 asked for it to be fit, not inherited from the NFL's 25).
- **Expected total** is matchup-specific: each program carries an
  exponentially weighted points-scored / points-allowed rate (half-life
  ~10 games, shrunk toward the FBS mean with a prior worth PRIOR_GAMES
  games), and the matchup total is T = L * (att_h * def_a + att_a * def_h)
  where L is the walk-forward mean points per team-game — the MLB
  pipeline's Dixon-Coles-style rates, re-tuned for football's scale.

Elo stays the sole authority on the win probability; the score is the
expected margin carved out of the expected total, reported at one decimal
(an average, not a literal final).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from CFB.daily.config import MARGIN_FIT_SEASONS, TOTAL_SEASONS
from CFB.data.teams import FBS, FCS_POOL

# Half-life of 10 team-games: GAMMA ** 10 = 0.5. A college season is 12
# games, so this is "this season, weighted toward recent weeks".
GAMMA = 0.5 ** (1.0 / 10.0)
PRIOR_GAMES = 12.0
TOTAL_MIN, TOTAL_MAX = 28.0, 95.0


@dataclass
class ScoreParams:
    margin_a: float          # margin = a + b * elo_diff
    margin_b: float
    league_total: float      # fallback total when no team rates are available

    @property
    def elo_per_point(self) -> float:
        return 1.0 / self.margin_b if self.margin_b else float("nan")

    def expected_margin(self, elo_diff: float) -> float:
        return self.margin_a + self.margin_b * elo_diff

    def expected_score(self, elo_diff: float,
                       total: float | None = None) -> tuple[float, float]:
        t = self.league_total if total is None else total
        margin = self.expected_margin(elo_diff)
        home = max((t + margin) / 2.0, 0.0)
        away = max((t - margin) / 2.0, 0.0)
        return round(home, 1), round(away, 1)


def fit(history: pd.DataFrame) -> ScoreParams:
    """Calibrate on the replay history (pre-game elo_diff vs. actual margin)."""
    seasons = sorted(history["season"].unique())
    recent = history[history["season"].isin(seasons[-MARGIN_FIT_SEASONS:])]
    b, a = np.polyfit(recent["elo_diff"].to_numpy(dtype=float),
                      recent["margin"].to_numpy(dtype=float), 1)
    tot = history[history["season"].isin(seasons[-TOTAL_SEASONS:])]
    league_total = float((tot["home_points"] + tot["away_points"]).mean())
    return ScoreParams(float(a), float(b), league_total)


@dataclass
class TeamRates:
    """Walk-forward EWMA points-for / points-against per program plus the
    FBS mean. Non-FBS opponents pool into one FCS bucket, same as the Elo."""
    w: dict[str, float] = field(default_factory=dict)
    pf: dict[str, float] = field(default_factory=dict)
    pa: dict[str, float] = field(default_factory=dict)
    league_w: float = 0.0
    league_pts: float = 0.0

    @property
    def league_mean(self) -> float:
        """Points per team-game (walk-forward); early fallback 27."""
        return self.league_pts / self.league_w if self.league_w > 20 else 27.0

    def _observe_side(self, team: str, scored: float, allowed: float) -> None:
        self.w[team] = self.w.get(team, 0.0) * GAMMA + 1.0
        self.pf[team] = self.pf.get(team, 0.0) * GAMMA + scored
        self.pa[team] = self.pa.get(team, 0.0) * GAMMA + allowed

    def observe(self, home: str, away: str, home_points: float,
                away_points: float) -> None:
        self._observe_side(home, home_points, away_points)
        self._observe_side(away, away_points, home_points)
        # Slow decay so the league mean tracks the era, not the week.
        self.league_w = self.league_w * 0.999 + 2.0
        self.league_pts = self.league_pts * 0.999 + home_points + away_points

    def _factor(self, weighted: dict[str, float], team: str) -> float:
        L = self.league_mean
        w = self.w.get(team, 0.0)
        rate = (weighted.get(team, 0.0) + PRIOR_GAMES * L) / (w + PRIOR_GAMES)
        return rate / L

    def attack(self, team: str) -> float:
        return self._factor(self.pf, team)

    def defense(self, team: str) -> float:
        """> 1 means the program ALLOWS more than average."""
        return self._factor(self.pa, team)

    def matchup_total(self, home: str, away: str) -> float:
        L = self.league_mean
        total = L * (self.attack(home) * self.defense(away)
                     + self.attack(away) * self.defense(home))
        return float(min(max(total, TOTAL_MIN), TOTAL_MAX))


def _key(team: str, division) -> str:
    return team if division == FBS else FCS_POOL


def rates_from_games(games: pd.DataFrame, before_date: str | None = None) -> TeamRates:
    """Replay completed games (optionally strictly before `before_date`) in
    kickoff order into a TeamRates state."""
    g = games[games["completed"].astype(bool)].sort_values(["start_utc", "game_id"])
    if before_date is not None:
        g = g[g["date"] < before_date]
    rates = TeamRates()
    for row in g.itertuples(index=False):
        rates.observe(_key(row.home_team, row.home_division),
                      _key(row.away_team, row.away_division),
                      float(row.home_points), float(row.away_points))
    return rates
