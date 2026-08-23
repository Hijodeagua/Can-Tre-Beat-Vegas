"""
Predict the slate: every league fixture dated inside the run's window,
with W/D/L probabilities, the model pick, and the most likely scoreline.

The slate is persisted to data/soccer_clubs/predictions/slate_{D}.csv the
day it is predicted — that file, written before the matches were played, is
what `grade.py` later grades. Re-running a date overwrites its slate file,
but grading is idempotent through the ledger: a match graded once is never
graded again, so a regenerated slate can't rewrite the record.
"""

from datetime import date, timedelta

import pandas as pd

from soccer.clubs.daily import scoring
from soccer.clubs.daily.config import PREDICTIONS_DIR, SLATE_WINDOW_DAYS
from soccer.clubs.daily.state import DailyState
from soccer.clubs.data.leagues import season_for_date

SLATE_COLUMNS = [
    "date", "league", "season", "home_team", "away_team",
    "elo_home_pre", "elo_away_pre", "p_H", "p_D", "p_A",
    "pick", "lambda_home", "lambda_away", "score_home", "score_away",
]


def upcoming_fixtures(state: DailyState, run_date: str,
                      window_days: int = SLATE_WINDOW_DAYS) -> pd.DataFrame:
    end = (date.fromisoformat(run_date) + timedelta(days=window_days)).isoformat()
    r = state.results
    return r[
        r["home_score"].isna()
        & (r["date"] >= run_date)
        & (r["date"] < end)
    ].copy()


def build_slate(state: DailyState, run_date: str,
                window_days: int = SLATE_WINDOW_DAYS) -> pd.DataFrame:
    fixtures = upcoming_fixtures(state, run_date, window_days)
    if fixtures.empty:
        return pd.DataFrame(columns=SLATE_COLUMNS)

    rows = []
    for _, f in fixtures.iterrows():
        season = f["season"] if isinstance(f["season"], str) else season_for_date(f["date"])
        row = state.feature_row(f["league"], f["home_team"], f["away_team"], season,
                                date=f["date"])
        row["date"] = f["date"]
        rows.append(row)
    feats = pd.DataFrame(rows)
    slate = state.outcome_probs(feats)

    picks, lams, scores = [], [], []
    for _, m in slate.iterrows():
        pick = max("HDA", key=lambda c: m[f"p_{c}"])
        lam = state.score_params.lambdas(m["league"], m["exp_home"])
        picks.append(pick)
        lams.append(lam)
        scores.append(scoring.most_likely_score(*lam))
    slate["pick"] = picks
    slate["lambda_home"] = [round(l[0], 2) for l in lams]
    slate["lambda_away"] = [round(l[1], 2) for l in lams]
    slate["score_home"] = [s[0] for s in scores]
    slate["score_away"] = [s[1] for s in scores]

    for c in ("p_H", "p_D", "p_A"):
        slate[c] = slate[c].round(4)
    for c in ("elo_home_pre", "elo_away_pre"):
        slate[c] = slate[c].round(1)
    return slate[SLATE_COLUMNS].sort_values(["date", "league", "home_team"])


def persist_slate(slate: pd.DataFrame, run_date: str) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"slate_{run_date}.csv"
    slate.to_csv(path, index=False)
