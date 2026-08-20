"""
Club Elo engine for the top-5 European leagues.

Same skeleton as the international engine (`soccer/model/elo.py`) — logistic
expectation, margin-of-victory multiplier, draws as 0.5 — with the two things
club league play adds:

- **Separate pools per league.** Domestic results never compare clubs across
  leagues, so each league is its own closed Elo economy; an EPL 1600 and a
  Ligue 1 1600 are not claims about each other.
- **Season structure.** At every season boundary all known ratings regress
  toward the base (squads churn over a summer), and clubs promoted into the
  league start below base at an entry rating — the newly promoted side is
  almost always worse than the average incumbent. A relegated club's rating
  keeps regressing while it's away and is picked back up on return.

Per-league parameters (K, home advantage, regression, entry rating) live in
`artifacts/tuned_params.json`, written by `tune.py`; DEFAULTS applies before
tuning has run.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
PARAMS_FILE = ARTIFACTS / "tuned_params.json"

BASE_RATING = 1500.0

DEFAULTS = {
    "k": 20.0,
    "home_advantage": 60.0,
    "season_regression": 0.20,  # rating -> base pull at each season boundary
    "entry_rating": 1420.0,     # first-ever appearance (promoted club)
}


def mov_multiplier(goal_diff: int) -> float:
    d = abs(goal_diff)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11 + d) / 8


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / 400.0))


def league_params(league: str) -> dict:
    """Tuned parameters for a league, or DEFAULTS before tune.py has run."""
    if PARAMS_FILE.exists():
        tuned = json.loads(PARAMS_FILE.read_text()).get("params", {})
        if league in tuned:
            # tuned entries also carry diagnostics (brier, matches_scored)
            return {k: tuned[league].get(k, v) for k, v in DEFAULTS.items()}
    return dict(DEFAULTS)


@dataclass
class ClubEloEngine:
    """One league's Elo pool. Feed it that league's matches in date order."""

    k: float = DEFAULTS["k"]
    home_advantage: float = DEFAULTS["home_advantage"]
    season_regression: float = DEFAULTS["season_regression"]
    entry_rating: float = DEFAULTS["entry_rating"]
    base: float = BASE_RATING
    ratings: Dict[str, float] = field(default_factory=dict)
    matches_played: Dict[str, int] = field(default_factory=dict)
    last_season: Dict[str, str] = field(default_factory=dict)
    current_season: Optional[str] = None

    @classmethod
    def for_league(cls, league: str) -> "ClubEloEngine":
        return cls(**league_params(league))

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.entry_rating)

    def _roll_season(self, season: str) -> None:
        """Regress every known club toward base once per season boundary —
        including clubs currently out of the league, so a returning club's
        old form has faded in proportion to its time away."""
        if self.current_season is not None:
            for t in self.ratings:
                self.ratings[t] += self.season_regression * (self.base - self.ratings[t])
        self.current_season = season

    def update(self, row: pd.Series) -> dict:
        """Process one played match; returns the pre-match feature record."""
        if row["season"] != self.current_season:
            self._roll_season(row["season"])

        home, away = row["home_team"], row["away_team"]
        r_home, r_away = self.get(home), self.get(away)

        exp_home = expected_score(r_home + self.home_advantage, r_away)
        goal_diff = int(row["home_score"]) - int(row["away_score"])
        actual = 1.0 if goal_diff > 0 else (0.0 if goal_diff < 0 else 0.5)

        delta = self.k * mov_multiplier(goal_diff) * (actual - exp_home)
        self.ratings[home] = r_home + delta
        self.ratings[away] = r_away - delta
        for t in (home, away):
            self.matches_played[t] = self.matches_played.get(t, 0) + 1
            self.last_season[t] = row["season"]

        return {
            "date": row["date"],
            "season": row["season"],
            "league": row["league"],
            "home_team": home,
            "away_team": away,
            "elo_home_pre": r_home,
            "elo_away_pre": r_away,
            "elo_gap": (r_home + self.home_advantage) - r_away,
            "exp_home": exp_home,
            "actual_home": actual,
            "outcome": "H" if goal_diff > 0 else ("A" if goal_diff < 0 else "D"),
            "home_score": int(row["home_score"]),
            "away_score": int(row["away_score"]),
        }

    def table(self, season: Optional[str] = None) -> pd.DataFrame:
        """Current ratings; pass `season` to keep only clubs who played in it."""
        rows = [
            {
                "team": t,
                "elo": r,
                "matches": self.matches_played.get(t, 0),
                "last_season": self.last_season.get(t, ""),
            }
            for t, r in self.ratings.items()
            if season is None or self.last_season.get(t) == season
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )


def load_results() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "results.csv")
    return df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)


def run_league(
    league: str,
    df: Optional[pd.DataFrame] = None,
    engine: Optional[ClubEloEngine] = None,
    end: Optional[str] = None,
) -> tuple[ClubEloEngine, pd.DataFrame]:
    """Replay one league's played matches chronologically; return the engine +
    per-match pre-rating feature records (the training table)."""
    if df is None:
        df = load_results()
    matches = df[df["league"] == league]
    if end:
        matches = matches[matches["date"] < end]
    engine = engine or ClubEloEngine.for_league(league)
    records: List[dict] = [engine.update(row) for _, row in matches.iterrows()]
    return engine, pd.DataFrame(records)


def run_all(
    df: Optional[pd.DataFrame] = None, end: Optional[str] = None
) -> tuple[Dict[str, ClubEloEngine], pd.DataFrame]:
    """All five leagues; returns {league: engine} + the stacked history."""
    from soccer.clubs.data.leagues import LEAGUES

    if df is None:
        df = load_results()
    engines: Dict[str, ClubEloEngine] = {}
    histories = []
    for league in LEAGUES:
        engines[league], history = run_league(league, df=df, end=end)
        histories.append(history)
    return engines, pd.concat(histories, ignore_index=True)


if __name__ == "__main__":
    engines, history = run_all()
    print(f"Processed {len(history)} club matches\n")
    for league, engine in engines.items():
        latest = history[history["league"] == league]["season"].max()
        print(f"=== {league} — top 5 (through {latest}) ===")
        print(engine.table(season=latest).head(5).to_string(index=False))
        print()
