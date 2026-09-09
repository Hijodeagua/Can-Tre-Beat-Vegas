"""Score model for the NFL daily pipeline: turn a pre-game Elo edge into
an expected final score — `CFB/daily/scoring.py` re-scaled for the NFL.

Both pieces are refit from the engine's own replay history every run (no
pickle drift):

- **Expected margin** is linear in the Elo difference (every situational
  edge included): margin = a + b * elo_diff, least squares on the last
  MARGIN_FIT_SEASONS seasons. 1/b is the engine's own "Elo per point",
  fit rather than inherited from the 25 the v2 feature engine assumes.
- **Expected total** is matchup-specific: each team carries an
  exponentially weighted points-scored / points-allowed rate (half-life
  8 games — half a season), shrunk toward the league mean with a prior
  worth PRIOR_GAMES games, and the matchup total is
  T = L * (att_h * def_a + att_a * def_h) with L the walk-forward mean
  points per team-game.
- **Mean MOV multiplier** over the fit window, so the season simulation's
  margin-free K updates move ratings by a realistic amount.

Elo stays the sole authority on the win probability; the score is the
expected margin carved out of the expected total, at one decimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from NFL.daily.config import MARGIN_FIT_SEASONS, TOTAL_SEASONS
from NFL.elo.engine import mov_multiplier

# Half-life of 8 team-games: GAMMA ** 8 = 0.5 — half a 17-game season.
GAMMA = 0.5 ** (1.0 / 8.0)
PRIOR_GAMES = 8.0
TOTAL_MIN, TOTAL_MAX = 24.0, 72.0
LEAGUE_FALLBACK = 23.0


@dataclass
class ScoreParams:
    margin_a: float          # margin = a + b * elo_diff
    margin_b: float
    league_total: float      # fallback total when no team rates are available
    mov_mean: float = 2.1    # mean margin multiplier, for the sim's K

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


def fit(history: pd.DataFrame, margin_cap: float = 45.0) -> ScoreParams:
    """Calibrate on the replay history (pre-game elo_diff vs. actual margin)."""
    seasons = sorted(history["season"].unique())
    recent = history[history["season"].isin(seasons[-MARGIN_FIT_SEASONS:])]
    b, a = np.polyfit(recent["elo_diff"].to_numpy(dtype=float),
                      recent["margin"].to_numpy(dtype=float), 1)
    tot = history[history["season"].isin(seasons[-TOTAL_SEASONS:])]
    league_total = float((tot["home_score"] + tot["away_score"]).mean())
    decided = recent[recent["margin"] != 0]
    mults = [
        mov_multiplier(max(-margin_cap, min(margin_cap, m)), d if m > 0 else -d)
        for m, d in zip(decided["margin"], decided["elo_diff"])
    ]
    return ScoreParams(float(a), float(b), league_total,
                       float(np.mean(mults)) if mults else 2.1)


@dataclass
class TeamRates:
    """Walk-forward EWMA points-for / points-against per team plus the
    league mean."""
    w: dict[str, float] = field(default_factory=dict)
    pf: dict[str, float] = field(default_factory=dict)
    pa: dict[str, float] = field(default_factory=dict)
    league_w: float = 0.0
    league_pts: float = 0.0

    @property
    def league_mean(self) -> float:
        """Points per team-game (walk-forward); early fallback 23."""
        return self.league_pts / self.league_w if self.league_w > 20 else LEAGUE_FALLBACK

    def _observe_side(self, team: str, scored: float, allowed: float) -> None:
        self.w[team] = self.w.get(team, 0.0) * GAMMA + 1.0
        self.pf[team] = self.pf.get(team, 0.0) * GAMMA + scored
        self.pa[team] = self.pa.get(team, 0.0) * GAMMA + allowed

    def observe(self, home: str, away: str, home_score: float,
                away_score: float) -> None:
        self._observe_side(home, home_score, away_score)
        self._observe_side(away, away_score, home_score)
        # Slow decay so the league mean tracks the era, not the week.
        self.league_w = self.league_w * 0.998 + 2.0
        self.league_pts = self.league_pts * 0.998 + home_score + away_score

    def _factor(self, weighted: dict[str, float], team: str) -> float:
        L = self.league_mean
        w = self.w.get(team, 0.0)
        rate = (weighted.get(team, 0.0) + PRIOR_GAMES * L) / (w + PRIOR_GAMES)
        return rate / L

    def attack(self, team: str) -> float:
        return self._factor(self.pf, team)

    def defense(self, team: str) -> float:
        """> 1 means the team ALLOWS more than average."""
        return self._factor(self.pa, team)

    def matchup_total(self, home: str, away: str) -> float:
        L = self.league_mean
        total = L * (self.attack(home) * self.defense(away)
                     + self.attack(away) * self.defense(home))
        return float(min(max(total, TOTAL_MIN), TOTAL_MAX))


def rates_from_games(games: pd.DataFrame, before_date: str | None = None) -> TeamRates:
    """Replay completed games (optionally strictly before `before_date`) in
    kickoff order into a TeamRates state."""
    g = games[games["completed"].astype(bool)].sort_values(["date", "gametime", "game_id"])
    if before_date is not None:
        g = g[g["date"] < before_date]
    rates = TeamRates()
    for row in g.itertuples(index=False):
        rates.observe(row.home_team, row.away_team,
                      float(row.home_score), float(row.away_score))
    return rates
