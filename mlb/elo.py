"""
Betting-blind MLB team Elo engine.

Inputs are game outcomes only (data/mlb/games_2009_2026.csv) - no odds,
no market information. Conventions mirror soccer/model/elo.py:

- Fresh start: every franchise enters at 1500 before the 2009 season.
- +HOME_ADVANTAGE Elo points to the home side.
- Optional margin-of-victory multiplier (FiveThirtyEight MLB style).
- Between seasons each rating is regressed toward 1500:
      R_new = 1500 + CARRYOVER * (R_old - 1500)

Franchise codes are canonical current-day BRef codes (FLA->MIA, OAK->ATH).
"""

from dataclasses import dataclass, field
from math import log
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
GAMES_CSV = REPO / "data" / "mlb" / "games_2009_2026.csv"

BASE_RATING = 1500.0
# Defaults are overwritten by the tuned values in data/mlb/elo_params.json
# (see mlb/tune_elo.py); these match the tuned optimum.
HOME_ADVANTAGE = 24.0
K = 3.0
CARRYOVER = 0.6
USE_MOV = True


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / 400.0))


def mov_multiplier(run_diff: int, elo_gap_winner: float) -> float:
    """FiveThirtyEight-style margin multiplier: ln-damped run margin, shrunk
    when the winner was already the favorite (autocorrelation guard)."""
    d = abs(run_diff)
    return log(d + 1.0) * (2.2 / (max(elo_gap_winner, 0.0) * 0.001 + 2.2))


@dataclass
class EloEngine:
    k: float = K
    home_advantage: float = HOME_ADVANTAGE
    carryover: float = CARRYOVER
    use_mov: bool = USE_MOV
    base: float = BASE_RATING
    ratings: Dict[str, float] = field(default_factory=dict)

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.base)

    def new_season(self) -> None:
        for t in self.ratings:
            self.ratings[t] = self.base + self.carryover * (
                self.ratings[t] - self.base
            )

    def update(self, row) -> dict:
        home, away = row.home_fr, row.away_fr
        r_home, r_away = self.get(home), self.get(away)
        exp_home = expected_score(r_home + self.home_advantage, r_away)
        run_diff = row.home_score - row.away_score
        actual = 1.0 if run_diff > 0 else 0.0  # no ties in MLB

        k = self.k
        if self.use_mov:
            winner_gap = (
                (r_home + self.home_advantage) - r_away
                if run_diff > 0
                else r_away - (r_home + self.home_advantage)
            )
            k *= mov_multiplier(run_diff, winner_gap)

        delta = k * (actual - exp_home)
        self.ratings[home] = r_home + delta
        self.ratings[away] = r_away - delta

        return {
            "date": row.date,
            "season": row.season,
            "home": home,
            "away": away,
            "elo_home_pre": r_home,
            "elo_away_pre": r_away,
            "p_home": exp_home,
            "home_win": actual,
            "run_diff": run_diff,
        }


def load_games() -> pd.DataFrame:
    df = pd.read_csv(GAMES_CSV)
    return df.sort_values(["date", "game_num"]).reset_index(drop=True)


def run_history(
    k: float = K,
    home_advantage: float = HOME_ADVANTAGE,
    carryover: float = CARRYOVER,
    use_mov: bool = USE_MOV,
) -> tuple[EloEngine, pd.DataFrame, pd.DataFrame]:
    """Replay all games chronologically.

    Returns (engine, per-game history, per-team-season summary)."""
    games = load_games()
    engine = EloEngine(k, home_advantage, carryover, use_mov)

    records: List[dict] = []
    season_start: Dict[str, float] = {}
    summaries: List[dict] = []
    current_season = None

    for row in games.itertuples(index=False):
        if row.season != current_season:
            if current_season is not None:
                summaries += _summarize(current_season, season_start, engine)
                engine.new_season()
            current_season = row.season
            season_start = {}
        for t in (row.home_fr, row.away_fr):
            season_start.setdefault(t, engine.get(t))
        records.append(engine.update(row))

    summaries += _summarize(current_season, season_start, engine)
    return engine, pd.DataFrame(records), pd.DataFrame(summaries)


def _summarize(season, season_start, engine) -> List[dict]:
    return [
        {
            "season": season,
            "franchise": t,
            "elo_start": r0,
            "elo_end": engine.get(t),
            "elo_delta": engine.get(t) - r0,
        }
        for t, r0 in season_start.items()
    ]


if __name__ == "__main__":
    engine, hist, seasons = run_history()
    print(f"Processed {len(hist)} games")
    top = seasons[seasons.season == seasons.season.max()]
    print(top.sort_values("elo_end", ascending=False).head(10).to_string(index=False))
