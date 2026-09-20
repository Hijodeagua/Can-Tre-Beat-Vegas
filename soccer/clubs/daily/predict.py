"""
Predict the slate: every league fixture dated inside the run's window,
with W/D/L probabilities, the model pick, and a scoreline that agrees
with that pick.

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

# The persisted slate schema. Deliberately unchanged by the per-side
# stats below: this file is what `grade.py` reads back, so widening it
# would rewrite the shape of every historical prediction for no gain.
# The per-side numbers ride on the in-memory frame to the site JSON
# instead.
SLATE_COLUMNS = [
    "date", "league", "season", "home_team", "away_team",
    "elo_home_pre", "elo_away_pre", "p_H", "p_D", "p_A",
    "pick", "lambda_home", "lambda_away", "score_home", "score_away",
    "score_prob",
]

# Every per-side number the model is built from, as
# (key, home column, away column, label, group).
#
# The model only ever sees differentials — it is trained on
# `home minus away` and nothing else — so these columns are not model
# inputs and reading them is not reading the model's mind. They are the
# two halves each differential was made from, which is what a match card
# needs: "+0.4 xG for" is a different match when it is 1.9 against 1.5
# than when it is 0.7 against 0.3.
#
# `higher_is_better` is None where the direction is not a judgement
# (rest days, squad value) and False where a low number is the good one
# (goals conceded, PPDA — fewer opposition passes per defensive action is
# more pressing, not less).
SIDE_METRICS: list[tuple[str, str, str, str, str, bool | None]] = [
    ("elo", "elo_home_pre", "elo_away_pre", "Elo", "Rating", True),

    ("xg_for_ewm", "home_xg_for_ewm", "away_xg_for_ewm", "xG for (EWM)", "Attack", True),
    ("xg_for_r10", "home_xg_for_r10", "away_xg_for_r10", "xG for (last 10)", "Attack", True),
    ("npxg_for_ewm", "home_npxg_for_ewm", "away_npxg_for_ewm", "npxG for (EWM)", "Attack", True),
    ("npxg_for_r10", "home_npxg_for_r10", "away_npxg_for_r10", "npxG for (last 10)", "Attack", True),
    ("xg_for_split", "home_xg_for_split", "away_xg_for_split", "xG for, this venue", "Attack", True),
    ("npxg_for_split", "home_npxg_for_split", "away_npxg_for_split", "npxG for, this venue", "Attack", True),

    ("xg_against_ewm", "home_xg_against_ewm", "away_xg_against_ewm", "xG against (EWM)", "Defence", False),
    ("xg_against_r10", "home_xg_against_r10", "away_xg_against_r10", "xG against (last 10)", "Defence", False),
    ("npxg_against_ewm", "home_npxg_against_ewm", "away_npxg_against_ewm", "npxG against (EWM)", "Defence", False),
    ("npxg_against_r10", "home_npxg_against_r10", "away_npxg_against_r10", "npxG against (last 10)", "Defence", False),
    ("xg_against_split", "home_xg_against_split", "away_xg_against_split", "xG against, this venue", "Defence", False),
    ("npxg_against_split", "home_npxg_against_split", "away_npxg_against_split", "npxG against, this venue", "Defence", False),

    ("xg_per_shot_ewm", "home_xg_per_shot_ewm", "away_xg_per_shot_ewm", "xG per shot (EWM)", "Chance quality", True),
    ("xg_per_shot_r10", "home_xg_per_shot_r10", "away_xg_per_shot_r10", "xG per shot (last 10)", "Chance quality", True),

    ("ppda_for_ewm", "home_ppda_for_ewm", "away_ppda_for_ewm", "PPDA (EWM)", "Pressing & territory", False),
    ("ppda_for_r10", "home_ppda_for_r10", "away_ppda_for_r10", "PPDA (last 10)", "Pressing & territory", False),
    ("deep_for_ewm", "home_deep_for_ewm", "away_deep_for_ewm", "Deep completions for (EWM)", "Pressing & territory", True),
    ("deep_for_r10", "home_deep_for_r10", "away_deep_for_r10", "Deep completions for (last 10)", "Pressing & territory", True),
    ("deep_against_ewm", "home_deep_against_ewm", "away_deep_against_ewm", "Deep completions against (EWM)", "Pressing & territory", False),
    ("deep_against_r10", "home_deep_against_r10", "away_deep_against_r10", "Deep completions against (last 10)", "Pressing & territory", False),
    ("deep_share_ewm", "home_deep_share_ewm", "away_deep_share_ewm", "Deep share (EWM)", "Pressing & territory", True),
    ("deep_share_r10", "home_deep_share_r10", "away_deep_share_r10", "Deep share (last 10)", "Pressing & territory", True),

    ("xpts_for_ewm", "home_xpts_for_ewm", "away_xpts_for_ewm", "xPts (EWM)", "Expected points", True),
    ("xpts_for_r10", "home_xpts_for_r10", "away_xpts_for_r10", "xPts (last 10)", "Expected points", True),

    ("xg_net", "home_xg_net", "away_xg_net", "xG net form", "Rolling form", True),
    ("sot_net", "home_sot_net", "away_sot_net", "Shots-on-target net form", "Rolling form", True),

    ("rest", "rest_home", "rest_away", "Rest days", "Fatigue", None),
    ("congestion14", "congestion14_home", "congestion14_away", "Matches in 14 days", "Fatigue", None),
    ("uefa7", "uefa7_home", "uefa7_away", "European ties in 7 days", "Fatigue", None),

    ("value_z", "home_value_z", "away_value_z", "Squad value (z)", "Squad economics", None),
    ("wage_z", "home_wage_z", "away_wage_z", "Wage bill (z)", "Squad economics", None),
    ("spend_z", "home_spend_z", "away_spend_z", "Transfer spend (z)", "Squad economics", None),
    ("net_z", "home_net_z", "away_net_z", "Net transfer spend (z)", "Squad economics", None),
]

SIDE_COLUMNS = [c for _, h, a, *_ in SIDE_METRICS for c in (h, a)]


def published_metrics() -> list[tuple]:
    """SIDE_METRICS minus anything whose feed is empty today.

    Only wage survives this filter for now: `wage_z` is 0.0 on every row
    because no wage source is wired up, and a published 0 would claim a
    league-average wage bill rather than admitting to no data. Everything
    else either has data or goes to NaN on its own, which the card draws
    as a dash.
    """
    from soccer.clubs.model.features import wages_available

    if wages_available():
        return list(SIDE_METRICS)
    return [m for m in SIDE_METRICS if m[0] != "wage_z"]


def side_stats(row) -> dict:
    """One match's per-side numbers as {home: {...}, away: {...}}.

    A metric the frame never carried, or carried as NaN (a club short of
    the warm-up minimum, or whose feed has gone stale), is omitted rather
    than zeroed — the site draws a dash, which is the honest rendering of
    "no reading".
    """
    import math

    out = {"home": {}, "away": {}}
    for key, home_col, away_col, *_ in published_metrics():
        for side, col in (("home", home_col), ("away", away_col)):
            if col not in row:
                continue
            v = row[col]
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(v):
                continue
            out[side][key] = round(v, 3)
    return out


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

    # The published scoreline is conditioned on the pick, so a row can
    # never read "Pick: Manchester City / Score: 1-1". See
    # scoring.representative_score.
    picks, lams, scores = [], [], []
    for _, m in slate.iterrows():
        pick = max("HDA", key=lambda c: m[f"p_{c}"])
        lam = state.score_params.lambdas(m["league"], m["exp_home"])
        picks.append(pick)
        lams.append(lam)
        scores.append(scoring.representative_score(*lam, pick))
    slate["pick"] = picks
    slate["lambda_home"] = [round(l[0], 2) for l in lams]
    slate["lambda_away"] = [round(l[1], 2) for l in lams]
    slate["score_home"] = [s[0] for s in scores]
    slate["score_away"] = [s[1] for s in scores]
    slate["score_prob"] = [round(s[2], 4) for s in scores]

    for c in ("p_H", "p_D", "p_A"):
        slate[c] = slate[c].round(4)
    for c in ("elo_home_pre", "elo_away_pre"):
        slate[c] = slate[c].round(1)
    # The published frame is SLATE_COLUMNS plus whichever per-side columns
    # this run actually produced; `persist_slate` narrows it back down, so
    # the graded CSV keeps its schema.
    extra = [c for c in SIDE_COLUMNS if c in slate.columns and c not in SLATE_COLUMNS]
    return slate[SLATE_COLUMNS + extra].sort_values(["date", "league", "home_team"])


def persist_slate(slate: pd.DataFrame, run_date: str) -> None:
    """Write the graded-slate CSV — SLATE_COLUMNS only.

    The in-memory frame also carries the per-side stats for the site, but
    this file is the one `grade.py` reads back and the one every past
    prediction already lives in, so it keeps its schema.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"slate_{run_date}.csv"
    cols = [c for c in SLATE_COLUMNS if c in slate.columns]
    slate[cols].to_csv(path, index=False)
