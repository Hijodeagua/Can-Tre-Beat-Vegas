"""MOV-adjusted Elo ratings for NFL teams, computed walk-forward.

Ratings are updated game by game in chronological order, so the rating
attached to a game is always the *pregame* rating — there is no leakage by
construction. Between seasons every team regresses partway to the mean.

Conventions follow the FiveThirtyEight NFL Elo write-up closely enough to be
recognisable:

- ``K`` = 20, scaled by a margin-of-victory multiplier
- home-field advantage of ``HFA`` Elo points (~2.2 real points)
- 25% regression to 1500 at the start of each season
- playoff games carry a slightly larger K

Elo points convert to a point spread at roughly 25 Elo per point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BASE_RATING = 1500.0
K_FACTOR = 20.0
PLAYOFF_K_MULT = 1.2
HFA_ELO = 55.0
SEASON_REGRESSION = 0.25  # fraction pulled back to BASE_RATING each new season
ELO_PER_POINT = 25.0

PLAYOFF_TYPES = {"WC", "DIV", "CON", "SB"}


def expected_score(rating_a: float, rating_b: float) -> float:
    """Win probability for A given both ratings (A's rating already includes HFA)."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def mov_multiplier(margin: float, elo_diff_winner: float) -> float:
    """Dampen blowouts, and dampen them harder when the favourite wins big."""
    return np.log(abs(margin) + 1.0) * (2.2 / (elo_diff_winner * 0.001 + 2.2))


@dataclass
class EloEngine:
    k: float = K_FACTOR
    hfa: float = HFA_ELO
    regression: float = SEASON_REGRESSION
    ratings: dict[str, float] = field(default_factory=dict)
    _last_season: int | None = None

    def rating(self, team: str) -> float:
        return self.ratings.setdefault(team, BASE_RATING)

    def _roll_season(self, season: int) -> None:
        if self._last_season is not None and season != self._last_season:
            for team, r in self.ratings.items():
                self.ratings[team] = BASE_RATING + (1 - self.regression) * (r - BASE_RATING)
        self._last_season = season

    def pregame(self, season: int, home: str, away: str, neutral: bool = False) -> tuple[float, float, float]:
        """Return (home_elo, away_elo, home_win_prob) *before* the game is played."""
        self._roll_season(season)
        h, a = self.rating(home), self.rating(away)
        hfa = 0.0 if neutral else self.hfa
        return h, a, expected_score(h + hfa, a)

    def update(
        self,
        home: str,
        away: str,
        home_score: float,
        away_score: float,
        neutral: bool = False,
        playoff: bool = False,
    ) -> None:
        h, a = self.rating(home), self.rating(away)
        hfa = 0.0 if neutral else self.hfa
        exp_home = expected_score(h + hfa, a)
        margin = home_score - away_score
        actual_home = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)

        if margin == 0:
            mult = 1.0
        else:
            # Elo edge from the winner's point of view, including HFA.
            elo_diff_winner = (h + hfa - a) if margin > 0 else (a - h - hfa)
            mult = mov_multiplier(margin, elo_diff_winner)

        k = self.k * (PLAYOFF_K_MULT if playoff else 1.0) * mult
        delta = k * (actual_home - exp_home)
        self.ratings[home] = h + delta
        self.ratings[away] = a - delta


def compute_elo(games: pd.DataFrame) -> pd.DataFrame:
    """Attach pregame Elo columns to a chronologically sortable games frame.

    Expects columns: ``season``, ``gameday``, ``game_type``, ``home_team``,
    ``away_team``, ``home_score``, ``away_score``, ``location``. Unplayed games
    (null scores) get pregame ratings but do not update the engine.

    Returns the input frame plus ``home_elo``, ``away_elo``, ``elo_diff``
    (home minus away, HFA included) and ``elo_home_prob``.
    """
    df = games.sort_values(["gameday", "game_type", "home_team"]).reset_index(drop=True)
    engine = EloEngine()

    home_elo, away_elo, probs = [], [], []
    for row in df.itertuples(index=False):
        neutral = str(getattr(row, "location", "Home")) != "Home"
        h, a, p = engine.pregame(int(row.season), row.home_team, row.away_team, neutral)
        home_elo.append(h)
        away_elo.append(a)
        probs.append(p)
        if pd.notna(row.home_score) and pd.notna(row.away_score):
            engine.update(
                row.home_team,
                row.away_team,
                float(row.home_score),
                float(row.away_score),
                neutral=neutral,
                playoff=str(row.game_type) in PLAYOFF_TYPES,
            )

    df["home_elo"] = home_elo
    df["away_elo"] = away_elo
    df["elo_diff"] = (
        df["home_elo"] - df["away_elo"] + np.where(df["location"].eq("Home"), HFA_ELO, 0.0)
    )
    df["elo_home_prob"] = probs
    # Elo's own view of the spread, in points, sign-matched to nflverse
    # ``spread_line`` (positive = home favoured).
    df["elo_spread"] = df["elo_diff"] / ELO_PER_POINT
    return df


def final_ratings(games: pd.DataFrame) -> pd.Series:
    """Ratings after every played game in ``games`` — useful for reporting."""
    engine = EloEngine()
    played = games.dropna(subset=["home_score", "away_score"])
    for row in played.sort_values(["gameday", "game_type"]).itertuples(index=False):
        neutral = str(getattr(row, "location", "Home")) != "Home"
        engine.pregame(int(row.season), row.home_team, row.away_team, neutral)
        engine.update(
            row.home_team,
            row.away_team,
            float(row.home_score),
            float(row.away_score),
            neutral=neutral,
            playoff=str(row.game_type) in PLAYOFF_TYPES,
        )
    return pd.Series(engine.ratings).sort_values(ascending=False)
