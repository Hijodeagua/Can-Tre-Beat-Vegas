"""
Custom international soccer Elo engine.

Rules (see soccer/SPEC.md):
- Fresh start: every team enters at 1500 on START_DATE (2006-01-01).
- Tiered K-factors with friendlies weighted the absolute lowest.
- Margin-of-victory multiplier (eloratings.net convention).
- +80 home advantage at non-neutral venues.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

START_DATE = "2006-01-01"
BASE_RATING = 1500.0
HOME_ADVANTAGE = 80.0

K_WORLD_CUP = 60.0
K_CONTINENTAL = 50.0
K_QUALIFIER = 40.0
K_NATIONS_LEAGUE = 30.0
K_OTHER = 15.0
K_FRIENDLY = 5.0

CONTINENTAL_FINALS = {
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "Confederations Cup",
}


def k_factor(tournament: str) -> float:
    t = str(tournament)
    if t == "Friendly":
        return K_FRIENDLY
    if t == "FIFA World Cup":
        return K_WORLD_CUP
    if t in CONTINENTAL_FINALS:
        return K_CONTINENTAL
    if "qualification" in t:
        return K_QUALIFIER
    if "Nations League" in t:
        return K_NATIONS_LEAGUE
    return K_OTHER


def mov_multiplier(goal_diff: int) -> float:
    d = abs(goal_diff)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11 + d) / 8


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / 400.0))


@dataclass
class EloEngine:
    base: float = BASE_RATING
    home_advantage: float = HOME_ADVANTAGE
    ratings: Dict[str, float] = field(default_factory=dict)
    matches_played: Dict[str, int] = field(default_factory=dict)

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.base)

    def update(self, row: pd.Series) -> dict:
        """Process one played match; returns the pre-match feature record."""
        home, away = row["home_team"], row["away_team"]
        r_home, r_away = self.get(home), self.get(away)
        neutral = bool(row["neutral"])
        adv = 0.0 if neutral else self.home_advantage

        exp_home = expected_score(r_home + adv, r_away)
        goal_diff = int(row["home_score"]) - int(row["away_score"])
        actual = 1.0 if goal_diff > 0 else (0.0 if goal_diff < 0 else 0.5)

        k = k_factor(row["tournament"]) * mov_multiplier(goal_diff)
        delta = k * (actual - exp_home)
        self.ratings[home] = r_home + delta
        self.ratings[away] = r_away - delta
        self.matches_played[home] = self.matches_played.get(home, 0) + 1
        self.matches_played[away] = self.matches_played.get(away, 0) + 1

        return {
            "date": row["date"],
            "home_team": home,
            "away_team": away,
            "tournament": row["tournament"],
            "neutral": neutral,
            "host_home": row["country"] == home,
            "host_away": row["country"] == away,
            "elo_home_pre": r_home,
            "elo_away_pre": r_away,
            "elo_gap": (r_home + adv) - r_away,
            "outcome": "H" if goal_diff > 0 else ("A" if goal_diff < 0 else "D"),
            "home_score": int(row["home_score"]),
            "away_score": int(row["away_score"]),
        }

    def table(self, min_matches: int = 10) -> pd.DataFrame:
        rows = [
            {"team": t, "elo": r, "matches": self.matches_played.get(t, 0)}
            for t, r in self.ratings.items()
            if self.matches_played.get(t, 0) >= min_matches
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )


def load_results(start: str = START_DATE) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "results.csv")
    df = df[df["date"] >= start].reset_index(drop=True)
    return df


def run_history(
    start: str = START_DATE, end: Optional[str] = None
) -> tuple[EloEngine, pd.DataFrame]:
    """Replay all played matches chronologically; return engine + per-match
    pre-rating feature records (the training table)."""
    df = load_results(start)
    played = df.dropna(subset=["home_score", "away_score"])
    if end:
        played = played[played["date"] < end]
    engine = EloEngine()
    records: List[dict] = [engine.update(row) for _, row in played.iterrows()]
    return engine, pd.DataFrame(records)


if __name__ == "__main__":
    engine, history = run_history()
    print(f"Processed {len(history)} matches since {START_DATE}\n")
    print("Top 20 teams by custom Elo:")
    print(engine.table().head(20).to_string(index=False))
