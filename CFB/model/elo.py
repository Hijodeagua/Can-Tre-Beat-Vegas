"""College-football Elo engine — the FBS sibling of `mlb/elo.py` and
`soccer/clubs/model/elo.py`, with the four things college needs that the
NFL engine (`NFL/model/v2/elo.py`) doesn't have (`CFB/DATA_PULL_PLAN.md`
§1 lists why each matters):

- **Conference-aware season regression.** At every season boundary each
  FBS program regresses toward a blend of its *new* conference's mean
  rating and the 1500 base (`conf_weight` sets the blend). Conference
  membership is read per season from the game spine, so realignment is
  handled by construction. Independents regress toward the base.
- **One pooled FCS opponent.** Every non-FBS program is rated as one fixed
  synthetic team (`fcs_rating`); a game against it updates only the FBS
  side. Loses the NDSU-vs-bottom-feeder distinction, keeps ~13% of the
  schedule in the data (plan §3.4).
- **FBS entry rating.** A program's first FBS game starts it at
  `entry_rating`, well below base — new arrivals from FCS are not average
  FBS teams.
- **Capped margin of victory.** The margin feeding the 538-style
  multiplier is clipped at `margin_cap` points: a 63-point blowout carries
  no information a 35-point one doesn't.

Otherwise the skeleton is the shared one: logistic expectation, +home
advantage in Elo points for the home side (none at neutral sites), K
scaled by ln(margin + 1) shrunk when the favourite wins. Tuned parameters
live in `artifacts/tuned_params.json` (written by `tune.py`); DEFAULTS
applies before tuning has run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import log
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from CFB.data.fetch_schedule import GAMES_CSV
from CFB.data.teams import FBS, FCS_POOL

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
PARAMS_FILE = ARTIFACTS / "tuned_params.json"

BASE_RATING = 1500.0
INDEPENDENT = "FBS Independents"

DEFAULTS = {
    "k": 30.0,
    "home_advantage": 65.0,       # Elo points; ~2.5 real points
    "season_regression": 0.40,    # fraction of (target - rating) applied each new season
    "conf_weight": 0.5,           # target = conf_weight * conference mean + (1 - w) * base
    "fcs_rating": 1200.0,         # the pooled FCS opponent's fixed rating
    "entry_rating": 1300.0,       # a program's rating on its first FBS game
    "margin_cap": 35.0,           # |margin| clipped here before the MOV multiplier
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
class CfbEloEngine:
    k: float = DEFAULTS["k"]
    home_advantage: float = DEFAULTS["home_advantage"]
    season_regression: float = DEFAULTS["season_regression"]
    conf_weight: float = DEFAULTS["conf_weight"]
    fcs_rating: float = DEFAULTS["fcs_rating"]
    entry_rating: float = DEFAULTS["entry_rating"]
    margin_cap: float = DEFAULTS["margin_cap"]
    base: float = BASE_RATING
    ratings: Dict[str, float] = field(default_factory=dict)
    games_played: Dict[str, int] = field(default_factory=dict)
    last_season: Dict[str, int] = field(default_factory=dict)
    conference: Dict[str, str] = field(default_factory=dict)  # current-season membership
    current_season: Optional[int] = None

    @classmethod
    def tuned(cls) -> "CfbEloEngine":
        return cls(**load_params())

    # --- ratings ------------------------------------------------------------

    def rating_for(self, team: str, division: str | None = FBS) -> float:
        """The rating a side carries into a game: the pooled FCS rating for
        any non-FBS program, the entry rating for an FBS program never seen."""
        if division != FBS:
            return self.fcs_rating
        return self.ratings.get(team, self.entry_rating)

    def roll_season(self, season: int, conferences: Dict[str, str]) -> None:
        """Season boundary: regress every rated program toward its new
        conference's mean (blended with base). `conferences` is the NEW
        season's membership; a program absent from it (dropped to FCS,
        or simply not yet scheduled) regresses toward base."""
        if self.current_season is not None and season != self.current_season:
            means: Dict[str, float] = {}
            buckets: Dict[str, list] = {}
            for t, r in self.ratings.items():
                conf = conferences.get(t)
                if conf and conf != INDEPENDENT:
                    buckets.setdefault(conf, []).append(r)
            for conf, rs in buckets.items():
                means[conf] = sum(rs) / len(rs)
            for t, r in self.ratings.items():
                conf = conferences.get(t)
                if conf in means:
                    target = self.conf_weight * means[conf] + (1 - self.conf_weight) * self.base
                else:
                    target = self.base
                self.ratings[t] = r + self.season_regression * (target - r)
        self.current_season = season
        self.conference = dict(conferences)

    def pregame(self, home: str, away: str, home_division: str,
                away_division: str, neutral: bool) -> tuple[float, float, float]:
        """(home_elo, away_elo, p_home) before the game."""
        r_home = self.rating_for(home, home_division)
        r_away = self.rating_for(away, away_division)
        adv = 0.0 if neutral else self.home_advantage
        return r_home, r_away, expected_score(r_home + adv, r_away)

    def update(self, home: str, away: str, home_points: float, away_points: float,
               home_division: str = FBS, away_division: str = FBS,
               neutral: bool = False) -> dict:
        """Process one played game; returns the pre-game feature record.
        The pooled FCS side never updates."""
        r_home, r_away, p_home = self.pregame(home, away, home_division,
                                             away_division, neutral)
        adv = 0.0 if neutral else self.home_advantage
        margin = float(home_points) - float(away_points)
        actual = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)

        if margin == 0:
            mult = 1.0
        else:
            capped = max(-self.margin_cap, min(self.margin_cap, margin))
            diff_winner = (r_home + adv - r_away) if margin > 0 else (r_away - r_home - adv)
            mult = mov_multiplier(capped, diff_winner)
        delta = self.k * mult * (actual - p_home)

        if home_division == FBS:
            self.ratings[home] = r_home + delta
            self.games_played[home] = self.games_played.get(home, 0) + 1
            self.last_season[home] = self.current_season
        if away_division == FBS:
            self.ratings[away] = r_away - delta
            self.games_played[away] = self.games_played.get(away, 0) + 1
            self.last_season[away] = self.current_season

        return {
            "home_team": home, "away_team": away,
            "home_fcs": home_division != FBS, "away_fcs": away_division != FBS,
            "neutral": neutral,
            "elo_home_pre": r_home, "elo_away_pre": r_away,
            "elo_diff": (r_home + adv) - r_away,
            "p_home": p_home, "home_win": actual,
            "margin": margin,
            "home_points": float(home_points), "away_points": float(away_points),
        }

    def table(self, season: Optional[int] = None) -> pd.DataFrame:
        """Current ratings; `season` keeps programs whose last game was in it
        or who are in that season's conference map (a program yet to play
        this season still belongs on the board)."""
        rows = []
        for t, r in self.ratings.items():
            if season is not None and self.last_season.get(t) != season \
                    and t not in self.conference:
                continue
            rows.append({
                "team": t, "elo": r,
                "conference": self.conference.get(t),
                "games": self.games_played.get(t, 0),
                "last_season": self.last_season.get(t),
            })
        return (pd.DataFrame(rows).sort_values("elo", ascending=False)
                .reset_index(drop=True))


# --- replay ---------------------------------------------------------------------

def load_games() -> pd.DataFrame:
    return pd.read_csv(GAMES_CSV, low_memory=False)


def season_conferences(games: pd.DataFrame) -> Dict[int, Dict[str, str]]:
    """{season: {fbs_team: conference}} from the spine (both sides)."""
    out: Dict[int, Dict[str, str]] = {}
    for side in ("home", "away"):
        sub = games[games[f"{side}_division"] == FBS]
        for season, team, conf in zip(sub["season"], sub[f"{side}_team"],
                                      sub[f"{side}_conference"]):
            if isinstance(conf, str):
                out.setdefault(int(season), {})[team] = conf
    return out


def replay(games: pd.DataFrame | None = None,
           engine: CfbEloEngine | None = None,
           end: str | None = None) -> tuple[CfbEloEngine, pd.DataFrame]:
    """Replay every played game chronologically. `end` (ISO date, exclusive,
    on the ET `date` column) stops early for walk-forward checks. Returns
    the engine plus the per-game pre-rating history — the same shape the
    MLB and club engines return, so the daily pipeline's calibration and
    grading code reads all three alike."""
    if games is None:
        games = load_games()
    conferences = season_conferences(games)
    played = games[games["completed"].astype(bool)]
    if end:
        played = played[played["date"] < end]
    played = played.sort_values(["start_utc", "game_id"])
    engine = engine or CfbEloEngine.tuned()

    records = []
    for row in played.itertuples(index=False):
        season = int(row.season)
        if season != engine.current_season:
            engine.roll_season(season, conferences.get(season, {}))
        rec = engine.update(
            row.home_team, row.away_team, row.home_points, row.away_points,
            home_division=row.home_division if isinstance(row.home_division, str) else "",
            away_division=row.away_division if isinstance(row.away_division, str) else "",
            neutral=bool(row.neutral_site),
        )
        rec.update({
            "game_id": int(row.game_id), "date": row.date, "season": season,
            "week": int(row.week), "season_type": row.season_type,
            "home_conference": row.home_conference, "away_conference": row.away_conference,
        })
        records.append(rec)
    return engine, pd.DataFrame(records)


def fast_replay(rows: Iterable[tuple], conferences: Dict[int, Dict[str, str]],
                params: dict, score_from: int, score_to: int) -> tuple[float, int]:
    """Stripped-down twin of the engine for the tuner's grid search: returns
    (mean log loss, n) over games with score_from <= season < score_to.
    `rows` are (season, home, away, home_is_fbs, away_is_fbs, neutral,
    home_points, away_points) in date order. The engine stays the source of
    truth for anything exported."""
    k = params["k"]; adv_home = params["home_advantage"]
    reg = params["season_regression"]; w = params["conf_weight"]
    fcs = params["fcs_rating"]; entry = params["entry_rating"]; cap = params["margin_cap"]
    base = BASE_RATING
    ratings: Dict[str, float] = {}
    current = None
    ll_sum, n = 0.0, 0
    for season, home, away, h_fbs, a_fbs, neutral, hp, ap in rows:
        if season != current:
            if current is not None:
                confs = conferences.get(season, {})
                buckets: Dict[str, list] = {}
                for t, r in ratings.items():
                    c = confs.get(t)
                    if c and c != INDEPENDENT:
                        buckets.setdefault(c, []).append(r)
                means = {c: sum(v) / len(v) for c, v in buckets.items()}
                for t, r in ratings.items():
                    c = confs.get(t)
                    target = w * means[c] + (1 - w) * base if c in means else base
                    ratings[t] = r + reg * (target - r)
            current = season
        r_home = ratings.get(home, entry) if h_fbs else fcs
        r_away = ratings.get(away, entry) if a_fbs else fcs
        adv = 0.0 if neutral else adv_home
        p = 1.0 / (1.0 + 10 ** (-((r_home + adv) - r_away) / 400.0))
        margin = hp - ap
        actual = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        if score_from <= season < score_to:
            pc = min(max(p, 1e-6), 1 - 1e-6)
            ll_sum -= actual * log(pc) + (1 - actual) * log(1 - pc)
            n += 1
        if margin == 0:
            mult = 1.0
        else:
            capped = max(-cap, min(cap, margin))
            dw = (r_home + adv - r_away) if margin > 0 else (r_away - r_home - adv)
            mult = log(abs(capped) + 1.0) * (2.2 / (max(dw, 0.0) * 0.001 + 2.2))
        delta = k * mult * (actual - p)
        if h_fbs:
            ratings[home] = r_home + delta
        if a_fbs:
            ratings[away] = r_away - delta
    return (ll_sum / n if n else float("nan")), n


if __name__ == "__main__":
    engine, history = replay()
    latest = int(history["season"].max())
    print(f"Processed {len(history)} games through {history['date'].max()}\n")
    print(engine.table(season=latest).head(25)[["team", "conference", "elo", "games"]]
          .to_string(index=False))
