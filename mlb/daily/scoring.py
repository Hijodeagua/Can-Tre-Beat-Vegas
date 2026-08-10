"""Team-level run-scoring rates for matchup-specific score predictions.

Dixon-Coles-style attack/defense factors adapted to MLB, computed strictly
walk-forward (a game's rates use only games completed before it):

- Each team carries an exponentially weighted mean of runs scored and runs
  allowed per game, half-life ~20 team-games. Recent form IS the weighting,
  so hot and cold scoring stretches move the numbers without any explicit
  streak flag.
- Rates are shrunk toward the league mean with a prior worth PRIOR_GAMES
  games, so early-season numbers stay sane; the EWMA carries across the
  offseason (same philosophy as Elo's season carryover - rosters mostly
  persist - but decayed like any other gap once games resume).
- A matchup's expected total is  T = L * (att_h * def_a + att_a * def_h)
  where L is the walk-forward league mean runs per team-game. Elo remains
  the sole authority on win probability and expected margin; these rates
  only set the run environment the margin is carved out of.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Half-life of 20 team-games: gamma^20 = 0.5
GAMMA = 0.5 ** (1.0 / 20.0)
# Shrinkage prior worth this many games of league-average scoring. Tuned by
# walk-forward grid (2021-2025): 60 minimizes total-runs MAE AND lands the
# calibration slope at ~0.95; lighter priors (15-30) widen the spread beyond
# what the signal supports (slope 0.5-0.7).
PRIOR_GAMES = 60.0
# Clip matchup totals to a sane MLB range.
TOTAL_MIN, TOTAL_MAX = 5.5, 13.5


@dataclass
class TeamRates:
    """Walk-forward EWMA state for every team plus the league mean."""
    w: dict[str, float] = field(default_factory=dict)
    rs: dict[str, float] = field(default_factory=dict)   # weighted runs scored
    ra: dict[str, float] = field(default_factory=dict)   # weighted runs allowed
    league_w: float = 0.0
    league_runs: float = 0.0

    @property
    def league_mean(self) -> float:
        """League runs per team-game (walk-forward); early fallback 4.5."""
        return self.league_runs / self.league_w if self.league_w > 20 else 4.5

    def _observe_side(self, team: str, scored: float, allowed: float) -> None:
        self.w[team] = self.w.get(team, 0.0) * GAMMA + 1.0
        self.rs[team] = self.rs.get(team, 0.0) * GAMMA + scored
        self.ra[team] = self.ra.get(team, 0.0) * GAMMA + allowed

    def observe(self, home: str, away: str,
                home_score: float, away_score: float) -> None:
        self._observe_side(home, home_score, away_score)
        self._observe_side(away, away_score, home_score)
        # League mean uses a slower decay so it tracks the era, not the week.
        self.league_w = self.league_w * 0.9995 + 2.0
        self.league_runs = self.league_runs * 0.9995 + home_score + away_score

    def _factor(self, weighted: dict[str, float], team: str) -> float:
        """Shrunk ratio of a team's EWMA rate to the league mean."""
        L = self.league_mean
        w = self.w.get(team, 0.0)
        rate = (weighted.get(team, 0.0) + PRIOR_GAMES * L) / (w + PRIOR_GAMES)
        return rate / L

    def attack(self, team: str) -> float:
        return self._factor(self.rs, team)

    def defense(self, team: str) -> float:
        """>1 means the team ALLOWS more runs than average."""
        return self._factor(self.ra, team)

    def matchup_total(self, home: str, away: str) -> float:
        L = self.league_mean
        total = L * (self.attack(home) * self.defense(away)
                     + self.attack(away) * self.defense(home))
        return float(min(max(total, TOTAL_MIN), TOTAL_MAX))


def rates_from_games(games: pd.DataFrame,
                     before_date: str | None = None) -> TeamRates:
    """Replay completed games (optionally only those strictly before
    `before_date`) in chronological order into a TeamRates state."""
    g = games.sort_values(["date", "game_num"])
    if before_date is not None:
        g = g[g.date < before_date]
    rates = TeamRates()
    for row in g.itertuples(index=False):
        rates.observe(row.home_fr, row.away_fr,
                      float(row.home_score), float(row.away_score))
    return rates
