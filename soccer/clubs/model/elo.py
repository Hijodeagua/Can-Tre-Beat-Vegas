"""
Club Elo engine for the top-5 European leagues and their second divisions.

Same skeleton as the international engine (`soccer/model/elo.py`) — logistic
expectation, margin-of-victory multiplier, draws as 0.5 — with the things
club league play adds:

- **One pool per country.** The top flight and its second division share an
  Elo pool (EPL + Championship = the "epl" pool), so promotion and
  relegation are just clubs changing which fixtures they play: a relegated
  club keeps playing rated matches, and a promoted club arrives carrying
  its actual second-division form. Pools are still closed across countries
  — an EPL 1600 and a Ligue 1 1600 are not claims about each other, except
  through the UEFA glue in `europe.py`.
- **Season structure.** At every season boundary all known ratings regress
  toward the base (squads churn over a summer). A club never seen before
  enters at a tier-dependent entry rating: `entry_rating` for a first
  top-flight appearance, the lower `entry_rating_t2` for a club coming up
  into the second division from the third tier.

Per-pool parameters (K, home advantage, regression, entry ratings) live in
`artifacts/tuned_params.json`, written by `tune.py`; DEFAULTS applies before
tuning has run.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from soccer.clubs.data.leagues import LEAGUES, POOLS, pool_of

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
PARAMS_FILE = ARTIFACTS / "tuned_params.json"

BASE_RATING = 1500.0

DEFAULTS = {
    "k": 20.0,
    "home_advantage": 60.0,
    "season_regression": 0.20,  # rating -> base pull at each season boundary
    "entry_rating": 1420.0,     # first-ever appearance in the top flight
    "entry_rating_t2": 1250.0,  # first-ever appearance, second division
    # How much of a club's rating survives a division switch: on its first
    # match in the other tier, r <- entry(tier) + carry * (r - entry(tier)).
    # Pinned at 1.0 — carry the rating unchanged, the way ClubElo does.
    # The old tuned values (0.25-0.5) came from an objective that scored
    # top-flight matches only, which is blind to the damage on the way
    # DOWN: a relegated West Ham went 1460 -> 1228 on its first
    # Championship match, landing below mid-table Serie B sides while
    # public ClubElo had it as the strongest club in the division. Entry
    # ratings still apply to clubs the pool has never seen.
    "division_carry": 1.0,
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
    """Tuned parameters for a league's pool, or DEFAULTS before tuning.
    Both divisions of a country resolve to the same pool entry."""
    pool = pool_of(league) if league in LEAGUES else league
    if PARAMS_FILE.exists():
        tuned = json.loads(PARAMS_FILE.read_text()).get("params", {})
        if pool in tuned:
            # tuned entries also carry diagnostics (brier, matches_scored)
            return {k: tuned[pool].get(k, v) for k, v in DEFAULTS.items()}
    return dict(DEFAULTS)


@dataclass
class ClubEloEngine:
    """One country's Elo pool (top flight + second division). Feed it that
    pool's matches in date order."""

    k: float = DEFAULTS["k"]
    home_advantage: float = DEFAULTS["home_advantage"]
    season_regression: float = DEFAULTS["season_regression"]
    entry_rating: float = DEFAULTS["entry_rating"]
    entry_rating_t2: float = DEFAULTS["entry_rating_t2"]
    division_carry: float = DEFAULTS["division_carry"]
    base: float = BASE_RATING
    ratings: Dict[str, float] = field(default_factory=dict)
    matches_played: Dict[str, int] = field(default_factory=dict)
    last_season: Dict[str, str] = field(default_factory=dict)
    last_league: Dict[str, str] = field(default_factory=dict)
    current_season: Optional[str] = None

    @classmethod
    def for_league(cls, league: str) -> "ClubEloEngine":
        return cls(**league_params(league))

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.entry_rating)

    def _entry(self, league: str) -> float:
        tier = LEAGUES[league].tier if league in LEAGUES else 1
        return self.entry_rating if tier == 1 else self.entry_rating_t2

    def rating_for(self, team: str, league: str) -> float:
        """The rating a club would carry into a match in `league` — applies
        the division-switch blend virtually for a club whose last match was
        in the other tier (a just-promoted club before its first top-flight
        result). Non-mutating; `update()` applies the real blend."""
        r = self.ratings.get(team)
        if r is None:
            return self._entry(league)
        last = self.last_league.get(team)
        if (
            last is not None
            and last in LEAGUES
            and league in LEAGUES
            and LEAGUES[last].tier != LEAGUES[league].tier
        ):
            entry = self._entry(league)
            return entry + self.division_carry * (r - entry)
        return r

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
        entry = self._entry(row["league"])
        for t in (home, away):
            last = self.last_league.get(t)
            if (
                t in self.ratings
                and last is not None
                and last in LEAGUES
                and row["league"] in LEAGUES
                and LEAGUES[last].tier != LEAGUES[row["league"]].tier
            ):
                # First match after a promotion/relegation: blend the carried
                # rating toward the new tier's entry level.
                self.ratings[t] = entry + self.division_carry * (self.ratings[t] - entry)
                self.last_league[t] = row["league"]
        r_home = self.ratings.get(home, entry)
        r_away = self.ratings.get(away, entry)

        exp_home = expected_score(r_home + self.home_advantage, r_away)
        goal_diff = int(row["home_score"]) - int(row["away_score"])
        actual = 1.0 if goal_diff > 0 else (0.0 if goal_diff < 0 else 0.5)

        delta = self.k * mov_multiplier(goal_diff) * (actual - exp_home)
        self.ratings[home] = r_home + delta
        self.ratings[away] = r_away - delta
        for t in (home, away):
            self.matches_played[t] = self.matches_played.get(t, 0) + 1
            self.last_season[t] = row["season"]
            self.last_league[t] = row["league"]

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

    def table(self, season: Optional[str] = None,
              league: Optional[str] = None) -> pd.DataFrame:
        """Current ratings; `season` keeps clubs whose last match was in it,
        `league` keeps clubs whose last match was in that division."""
        rows = [
            {
                "team": t,
                "elo": r,
                "matches": self.matches_played.get(t, 0),
                "last_season": self.last_season.get(t, ""),
                "last_league": self.last_league.get(t, ""),
            }
            for t, r in self.ratings.items()
            if (season is None or self.last_season.get(t) == season)
            and (league is None or self.last_league.get(t) == league)
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )


def load_results() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "results.csv")
    return df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)


def run_pool(
    pool: str,
    df: Optional[pd.DataFrame] = None,
    engine: Optional[ClubEloEngine] = None,
    end: Optional[str] = None,
) -> tuple[ClubEloEngine, pd.DataFrame]:
    """Replay one country pool's played matches (both divisions,
    chronological); return the engine + per-match pre-rating feature
    records. Rows keep their own division in the `league` column."""
    if df is None:
        df = load_results()
    matches = df[df["league"].isin(POOLS[pool])].sort_values("date", kind="stable")
    if end:
        matches = matches[matches["date"] < end]
    engine = engine or ClubEloEngine.for_league(pool)
    records: List[dict] = [engine.update(row) for _, row in matches.iterrows()]
    return engine, pd.DataFrame(records)


def run_all(
    df: Optional[pd.DataFrame] = None, end: Optional[str] = None
) -> tuple[Dict[str, ClubEloEngine], pd.DataFrame]:
    """All five country pools; returns {pool: engine} + the stacked history
    (pool keys are the tier-1 league keys: "epl", "bundesliga", …)."""
    if df is None:
        df = load_results()
    engines: Dict[str, ClubEloEngine] = {}
    histories = []
    for pool in POOLS:
        engines[pool], history = run_pool(pool, df=df, end=end)
        histories.append(history)
    return engines, pd.concat(histories, ignore_index=True)


if __name__ == "__main__":
    engines, history = run_all()
    print(f"Processed {len(history)} club matches\n")
    for pool, engine in engines.items():
        for league in POOLS[pool]:
            latest = history[history["league"] == league]["season"].max()
            print(f"=== {league} — top 5 (through {latest}) ===")
            print(
                engine.table(season=latest, league=league)
                .head(5)[["team", "elo", "matches"]]
                .to_string(index=False)
            )
            print()
