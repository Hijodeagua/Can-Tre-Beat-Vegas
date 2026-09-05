"""NFL Elo engine — the betting-blind sibling of `CFB/model/elo.py`,
`mlb/daily/ratings.py` and `soccer/clubs/model/elo.py`, grown from the
feature engine in `NFL/model/v2/elo.py`.

That engine exists to feed a LightGBM model that also sees the closing
line; this one has to stand on its own, so it gets the pieces the daily
pipelines gave the other sports and a few the NFL specifically wants:

- **Tuned parameters** (`artifacts/tuned_params.json`, written by
  `tune.py`) instead of the FiveThirtyEight constants copied into v2.
- **Franchise continuity.** STL/SD/OAK map onto LA/LAC/LV
  (`teams.FRANCHISE`), so a relocation carries its rating.
- **Rest edge.** A side coming off a bye (rest of `REST_BONUS_DAYS`+
  days) gets `rest_bonus` Elo at prediction time — 538's +25, but tuned.
- **Playoff K.** Postseason updates are scaled by `playoff_k_mult`.
- **Capped margin.** |margin| is clipped at `margin_cap` before the
  ln-damped multiplier, so a 50-point blowout is a 35-point one.
- **Ties.** The NFL still plays them: actual = 0.5, plain K, and the log
  loss scores both halves.

Otherwise the skeleton is the shared one: logistic expectation, home
advantage in Elo points (none at neutral sites), K scaled by ln(margin+1)
shrunk when the favourite wins, a fractional regression toward 1500 at
every season boundary. DEFAULTS are the v2 constants and apply before
tuning has run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import log
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from NFL.elo.teams import PLAYOFF_TYPES, canonical

REPO_ROOT = Path(__file__).resolve().parents[2]
GAMES_CSV = REPO_ROOT / "data" / "schedules" / "nflverse_games.csv"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
PARAMS_FILE = ARTIFACTS / "tuned_params.json"

BASE_RATING = 1500.0
# A bye week shows up as 13-14 days of rest; 10 separates it from the
# ordinary 6-8 (a Thursday-to-Sunday turnaround is 3-4, a Monday-to-Sunday 6).
REST_BONUS_DAYS = 10

DEFAULTS = {
    "k": 20.0,
    "home_advantage": 55.0,      # Elo points; ~2.2 real points at 25 Elo/pt
    "season_regression": 0.25,   # fraction of (base - rating) applied each new season
    "playoff_k_mult": 1.2,
    "margin_cap": 35.0,
    "rest_bonus": 0.0,           # Elo for the side coming off a bye
}


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / 400.0))


def mov_multiplier(margin: float, elo_diff_winner: float) -> float:
    """FiveThirtyEight-style: ln-damped margin, shrunk when the winner was
    already the favourite (autocorrelation guard)."""
    return log(abs(margin) + 1.0) * (2.2 / (max(elo_diff_winner, 0.0) * 0.001 + 2.2))


def load_params() -> dict:
    """Tuned parameters, or DEFAULTS before `tune.py` has run."""
    if PARAMS_FILE.exists():
        tuned = json.loads(PARAMS_FILE.read_text()).get("params", {})
        return {k: float(tuned.get(k, v)) for k, v in DEFAULTS.items()}
    return dict(DEFAULTS)


@dataclass
class NflEloEngine:
    k: float = DEFAULTS["k"]
    home_advantage: float = DEFAULTS["home_advantage"]
    season_regression: float = DEFAULTS["season_regression"]
    playoff_k_mult: float = DEFAULTS["playoff_k_mult"]
    margin_cap: float = DEFAULTS["margin_cap"]
    rest_bonus: float = DEFAULTS["rest_bonus"]
    base: float = BASE_RATING
    ratings: Dict[str, float] = field(default_factory=dict)
    games_played: Dict[str, int] = field(default_factory=dict)
    current_season: Optional[int] = None

    @classmethod
    def tuned(cls) -> "NflEloEngine":
        return cls(**load_params())

    # --- ratings ------------------------------------------------------------

    def rating_for(self, team: str) -> float:
        return self.ratings.get(team, self.base)

    def roll_season(self, season: int) -> None:
        """Season boundary: every team regresses `season_regression` of the
        way back to base. Idempotent within a season."""
        if self.current_season is not None and season != self.current_season:
            for t, r in self.ratings.items():
                self.ratings[t] = r + self.season_regression * (self.base - r)
        self.current_season = season

    def edges(self, neutral: bool, home_rest: float | None,
              away_rest: float | None) -> tuple[float, float]:
        """(home_adj, away_adj): the situational Elo each side carries into
        the game — home advantage and the bye-week rest bonus."""
        home_adj = 0.0 if neutral else self.home_advantage
        if home_rest is not None and home_rest == home_rest and home_rest >= REST_BONUS_DAYS:
            home_adj += self.rest_bonus
        away_adj = 0.0
        if away_rest is not None and away_rest == away_rest and away_rest >= REST_BONUS_DAYS:
            away_adj += self.rest_bonus
        return home_adj, away_adj

    def pregame(self, home: str, away: str, neutral: bool = False,
                home_rest: float | None = None,
                away_rest: float | None = None) -> tuple[float, float, float]:
        """(home_elo, away_elo, p_home) before the game — the raw ratings
        plus the probability with every situational edge applied."""
        r_home, r_away = self.rating_for(home), self.rating_for(away)
        h_adj, a_adj = self.edges(neutral, home_rest, away_rest)
        return r_home, r_away, expected_score(r_home + h_adj, r_away + a_adj)

    def update(self, home: str, away: str, home_score: float, away_score: float,
               neutral: bool = False, playoff: bool = False,
               home_rest: float | None = None,
               away_rest: float | None = None) -> dict:
        """Process one played game; returns the pre-game feature record."""
        r_home, r_away, p_home = self.pregame(home, away, neutral, home_rest, away_rest)
        h_adj, a_adj = self.edges(neutral, home_rest, away_rest)
        eff_home, eff_away = r_home + h_adj, r_away + a_adj
        margin = float(home_score) - float(away_score)
        actual = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)

        if margin == 0:
            mult = 1.0
        else:
            capped = max(-self.margin_cap, min(self.margin_cap, margin))
            diff_winner = (eff_home - eff_away) if margin > 0 else (eff_away - eff_home)
            mult = mov_multiplier(capped, diff_winner)
        k = self.k * (self.playoff_k_mult if playoff else 1.0)
        delta = k * mult * (actual - p_home)

        self.ratings[home] = r_home + delta
        self.ratings[away] = r_away - delta
        self.games_played[home] = self.games_played.get(home, 0) + 1
        self.games_played[away] = self.games_played.get(away, 0) + 1

        return {
            "home_team": home, "away_team": away, "neutral": neutral, "playoff": playoff,
            "elo_home_pre": r_home, "elo_away_pre": r_away,
            "elo_diff": eff_home - eff_away,
            "p_home": p_home, "home_win": actual, "margin": margin,
            "home_score": float(home_score), "away_score": float(away_score),
        }

    def table(self) -> pd.DataFrame:
        rows = [{"team": t, "elo": r, "games": self.games_played.get(t, 0)}
                for t, r in self.ratings.items()]
        return (pd.DataFrame(rows).sort_values("elo", ascending=False)
                .reset_index(drop=True))


# --- replay ---------------------------------------------------------------------

def load_games(path: Path | str = GAMES_CSV) -> pd.DataFrame:
    """The nflverse spine with franchise abbreviations canonicalised and a
    `date` column (the ET game date nflverse carries as `gameday`), sorted
    into kickoff order. `neutral` and `playoff` are derived flags."""
    g = pd.read_csv(path, low_memory=False)
    g["home_team"] = g["home_team"].map(canonical)
    g["away_team"] = g["away_team"].map(canonical)
    g["date"] = g["gameday"].astype(str).str.slice(0, 10)
    g["gametime"] = g["gametime"].fillna("").astype(str)
    g["neutral"] = g["location"].astype(str).ne("Home")
    g["playoff"] = g["game_type"].isin(PLAYOFF_TYPES)
    g["completed"] = g["home_score"].notna() & g["away_score"].notna()
    return g.sort_values(["date", "gametime", "game_id"]).reset_index(drop=True)


def _rest(v) -> float | None:
    return None if v is None or v != v else float(v)


def replay(games: pd.DataFrame | None = None,
           engine: NflEloEngine | None = None,
           end: str | None = None) -> tuple[NflEloEngine, pd.DataFrame]:
    """Replay every played game chronologically. `end` (ISO date, exclusive)
    stops early for walk-forward checks. Returns the engine plus the
    per-game pre-rating history the daily pipeline calibrates and grades
    from — the same shape the CFB, MLB and club engines return."""
    if games is None:
        games = load_games()
    played = games[games["completed"].astype(bool)]
    if end:
        played = played[played["date"] < end]
    played = played.sort_values(["date", "gametime", "game_id"])
    engine = engine or NflEloEngine.tuned()

    records = []
    for row in played.itertuples(index=False):
        season = int(row.season)
        if season != engine.current_season:
            engine.roll_season(season)
        rec = engine.update(
            row.home_team, row.away_team, row.home_score, row.away_score,
            neutral=bool(row.neutral), playoff=bool(row.playoff),
            home_rest=_rest(row.home_rest), away_rest=_rest(row.away_rest),
        )
        rec.update({
            "game_id": row.game_id, "date": row.date, "season": season,
            "week": int(row.week), "game_type": row.game_type,
        })
        records.append(rec)
    return engine, pd.DataFrame(records)


def fast_replay(rows: Iterable[tuple], params: dict,
                score_from: int, score_to: int) -> tuple[float, int]:
    """Stripped-down twin of the engine for the tuner's grid search:
    returns (mean log loss, n) over games with score_from <= season <
    score_to. `rows` are (season, home, away, neutral, playoff,
    home_bye, away_bye, home_score, away_score) in kickoff order. The
    engine stays the source of truth for anything exported."""
    k = params["k"]; adv_home = params["home_advantage"]
    reg = params["season_regression"]; pk = params["playoff_k_mult"]
    cap = params["margin_cap"]; bye = params["rest_bonus"]
    base = BASE_RATING
    ratings: Dict[str, float] = {}
    current = None
    ll_sum, n = 0.0, 0
    for season, home, away, neutral, playoff, h_bye, a_bye, hs, as_ in rows:
        if season != current:
            if current is not None:
                for t, r in ratings.items():
                    ratings[t] = r + reg * (base - r)
            current = season
        r_home = ratings.get(home, base)
        r_away = ratings.get(away, base)
        eff_h = r_home + (0.0 if neutral else adv_home) + (bye if h_bye else 0.0)
        eff_a = r_away + (bye if a_bye else 0.0)
        p = 1.0 / (1.0 + 10 ** (-(eff_h - eff_a) / 400.0))
        margin = hs - as_
        actual = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        if score_from <= season < score_to:
            pc = min(max(p, 1e-6), 1 - 1e-6)
            ll_sum -= actual * log(pc) + (1 - actual) * log(1 - pc)
            n += 1
        if margin == 0:
            mult = 1.0
        else:
            capped = max(-cap, min(cap, margin))
            dw = (eff_h - eff_a) if margin > 0 else (eff_a - eff_h)
            mult = log(abs(capped) + 1.0) * (2.2 / (max(dw, 0.0) * 0.001 + 2.2))
        delta = k * (pk if playoff else 1.0) * mult * (actual - p)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta
    return (ll_sum / n if n else float("nan")), n


if __name__ == "__main__":
    engine, history = replay()
    print(f"Processed {len(history)} games through {history['date'].max()}\n")
    print(engine.table().to_string(index=False))
