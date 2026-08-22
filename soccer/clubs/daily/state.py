"""
Shared per-run state for the daily pipeline: one glued Elo replay + the
outcome model + score calibration, built once and passed around.

The outcome model is refit in-run from the replay history rather than
unpickled — 25k rows fit in ~a second, and it keeps the daily job immune to
sklearn pickle drift in CI. Feature set and training rows match
`soccer/clubs/model/train.py` exactly.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression

from soccer.clubs.daily import scoring
from soccer.clubs.data.leagues import LEAGUES
from soccer.clubs.model.elo import ClubEloEngine
from soccer.clubs.model.europe import run_all_european
from soccer.clubs.model.features import ALL_FEATURES, attach_features

FEATURES = ["elo_gap"] + ALL_FEATURES
CLASSES = ["A", "D", "H"]


@dataclass
class DailyState:
    engines: dict[str, ClubEloEngine]
    history: pd.DataFrame          # league rows only, features attached
    results: pd.DataFrame          # raw results.csv incl. unplayed fixtures
    outcome_model: LogisticRegression
    score_params: scoring.ScoreParams

    def feature_row(self, league: str, home: str, away: str,
                    season: str, neutral: bool = False) -> dict:
        """Pre-match features for one fixture, from current ratings."""
        from soccer.clubs.model.elo import expected_score

        e = self.engines[league]
        r_home, r_away = e.get(home), e.get(away)
        adv = 0.0 if neutral else e.home_advantage
        row = {
            "league": league,
            "season": season,
            "home_team": home,
            "away_team": away,
            "elo_home_pre": r_home,
            "elo_away_pre": r_away,
            "elo_gap": (r_home + adv) - r_away,
            "exp_home": expected_score(r_home + adv, r_away),
        }
        return row

    def outcome_probs(self, feature_rows: pd.DataFrame) -> pd.DataFrame:
        """P(A), P(D), P(H) columns for a frame of feature rows."""
        f = attach_features(feature_rows)
        probs = self.outcome_model.predict_proba(f[FEATURES])
        out = feature_rows.copy()
        for i, c in enumerate(self.outcome_model.classes_):
            out[f"p_{c}"] = probs[:, i]
        return out


def build_state() -> DailyState:
    from soccer.clubs.model.elo import DATA_DIR

    engines, history = run_all_european()
    league_hist = history[~history["league"].str.startswith("uefa:")].copy()
    featured = attach_features(league_hist)

    model = LogisticRegression(max_iter=2000)
    model.fit(featured[FEATURES], featured["outcome"])

    results = pd.read_csv(DATA_DIR / "results.csv")
    return DailyState(
        engines=engines,
        history=featured,
        results=results,
        outcome_model=model,
        score_params=scoring.fit(league_hist),
    )
