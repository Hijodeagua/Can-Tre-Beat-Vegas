"""
Pregame efficiency features for the NFL second-stage model, from the
team-game aggregates in `data/nfl/team_games.csv` (built by
`NFL.data.pbp` from nflverse play-by-play).

Two kinds of pregame number, both computed only from games completed
before the game they describe:

1. **Opponent-adjusted ratings** (`*_adj`) — for each efficiency metric,
   a weighted ridge regression over every offence-vs-defence observation
   completed before the week being predicted:

       y_obs = off[offence] + def[defence] + hfa · home + ε

   `off` is EPA (or success rate, …) above league average the offence
   produces against an average defence; `def` is what the defence
   *allows* above average against an average offence, so a good defence
   is negative and the expected value of an offence against a defence is
   `off + def`. Observations are weighted by recency (half-life
   HALF_LIFE_WEEKS) with the previous season carried at PRIOR_DISCOUNT so
   Week 1 starts from a regressed prior rather than nothing; the ridge
   penalty shrinks every team toward the league mean, hardest when it has
   few observations. Ratings are recomputed once per (season, week) from
   games with `week < W`, so a Sunday game never sees Thursday's result
   from the same week either. This is an adjusted efficiency rating. It
   is not DVOA and is not labelled as one.

2. **Unadjusted form** (`*_ewm`) — the team's own exponentially weighted
   mean of each metric over its previous games (half-life HALF_LIFE_GAMES),
   shrunk toward the league mean by sample size, shifted so the current
   game is excluded. Cheaper, no opponent correction, kept for the
   ablation to compare against.

`build_game_table` joins both to the Elo replay history and derives the
cross-unit matchup features. Feature groups are the ones the ablation
compares; production reads `PRODUCTION_FEATURES` from the evaluation's
verdict, not from this file's imagination.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import freshness, learners, pregame
from NFL.data.pbp import load_team_games
from NFL.elo.teams import canonical

HALF_LIFE_WEEKS = 5.0
HALF_LIFE_GAMES = 5.0
PRIOR_DISCOUNT = 0.5      # previous season's games count half, on top of recency
RIDGE = 2.0               # shrinkage toward league average (weights sum to ~1 per team-game)
PRIOR_GAMES = 4.0         # shrinkage pseudo-games for the unadjusted EWMA

# Metrics that get an opponent-adjusted rating. All are offence-perspective
# columns of team_games; the defence rating falls out of the same fit.
ADJ_METRICS = [
    "epa", "success", "db_epa", "db_success", "rush_epa", "rush_success",
    "early_epa", "early_success", "explosive_rate", "points_per_drive",
    "epa_per_drive", "rz_td_per_trip", "third_conv",
]
# Metrics whose EWMA form is used directly (no opponent adjustment makes sense).
FORM_METRICS = [
    "off_epa", "def_epa",     # unadjusted twins of the core ratings, for the ablation
    "off_proe", "off_proe_neutral", "off_pass_rate", "off_neutral_pace",
    "off_drives", "off_plays_per_drive", "off_yards_per_drive", "off_series_success",
    "off_rz_trips", "off_rz_pts_per_trip", "off_rz_epa",
    "off_third_epa", "off_third_dist", "off_third_short_conv", "off_third_long_conv",
    "off_sack_rate", "def_sack_rate", "st_epa_net",
]

# --- feature groups for the ablation ---------------------------------------
ELO = ["elo_logit"]
CORE_EPA = ["home_off_epa_adj", "away_off_epa_adj", "home_def_epa_adj", "away_def_epa_adj",
            "epa_matchup_net"]
SUCCESS = ["home_off_success_adj", "away_off_success_adj",
           "home_def_success_adj", "away_def_success_adj", "success_matchup_net"]
SPLITS = ["db_epa_matchup_net", "rush_epa_matchup_net", "early_epa_matchup_net",
          "home_off_proe_ewm", "away_off_proe_ewm", "explosive_matchup_net"]
DRIVE = ["points_per_drive_matchup_net", "epa_per_drive_matchup_net",
         "home_off_drives_ewm", "away_off_drives_ewm",
         "home_off_plays_per_drive_ewm", "away_off_plays_per_drive_ewm",
         "home_off_yards_per_drive_ewm", "away_off_yards_per_drive_ewm",
         "home_off_series_success_ewm", "away_off_series_success_ewm"]
RED_ZONE_THIRD = ["rz_td_matchup_net", "third_conv_matchup_net",
                  "home_off_rz_epa_ewm", "away_off_rz_epa_ewm",
                  "home_off_third_epa_ewm", "away_off_third_epa_ewm",
                  "home_off_third_dist_ewm", "away_off_third_dist_ewm"]
PACE_ST = ["home_off_neutral_pace_ewm", "away_off_neutral_pace_ewm",
           "home_st_epa_net_ewm", "away_st_epa_net_ewm",
           "home_off_sack_rate_ewm", "away_off_sack_rate_ewm",
           "home_def_sack_rate_ewm", "away_def_sack_rate_ewm"]

# Unadjusted EWMA twin of CORE_EPA: shows what the opponent adjustment buys.
CORE_EWM = ["home_off_epa_ewm", "away_off_epa_ewm", "home_def_epa_ewm", "away_def_epa_ewm"]

# What the daily pipeline ships: every pregame feature this module builds,
# with the home and away Elo as their own inputs beside the Elo logit so
# the learner can find a level effect or a threshold the gap alone hides.
# The learner is the one `eval_advanced` measured best on the clean
# 2024-2025 window with this full set (docs/ADVANCED_METRICS.md).
RAW_ELO = ["elo_home_pre", "elo_away_pre"]
PRODUCTION_FEATURES = (ELO + RAW_ELO + CORE_EPA + SUCCESS + SPLITS + DRIVE
                       + RED_ZONE_THIRD + PACE_ST + CORE_EWM)
PRODUCTION_METRICS = ADJ_METRICS
PRODUCTION_LEARNER = "gbm"
PRODUCTION_C = 0.03            # the logistic's C, when that is the learner
# The aggregates must reach within this many days of the newest completed
# game in the spine, or the second stage is off for the run (Elo only).
# nflverse rebuilds nightly; a Monday game lands by Tuesday morning.
TOLERANCE_DAYS = 10

GROUPS = {
    "elo": ELO,
    "core_epa": CORE_EPA,
    "core_ewm": CORE_EWM,
    "success": SUCCESS,
    "splits": SPLITS,
    "drive": DRIVE,
    "red_zone_third": RED_ZONE_THIRD,
    "pace_st": PACE_ST,
}


# --------------------------------------------------------------------------
# 1. opponent-adjusted ratings, once per (season, week)
# --------------------------------------------------------------------------
def _weighted_ridge(obs: pd.DataFrame, metric: str, teams: list[str],
                    ridge: float = RIDGE) -> tuple[dict, dict, float]:
    """One fit. obs rows: offence, defence, home (0/1), weight, y."""
    y = obs[metric].to_numpy(dtype=float)
    w = obs["weight"].to_numpy(dtype=float)
    ok = np.isfinite(y)
    y, w, ob = y[ok], w[ok], obs[ok]
    if len(y) == 0:
        return {t: 0.0 for t in teams}, {t: 0.0 for t in teams}, 0.0
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    X = np.zeros((len(y), 2 * n + 1))
    X[np.arange(len(y)), [idx[t] for t in ob["offence"]]] = 1.0
    X[np.arange(len(y)), [n + idx[t] for t in ob["defence"]]] = 1.0
    X[:, 2 * n] = ob["home"].to_numpy(dtype=float)
    mean = np.average(y, weights=w)
    yc = y - mean
    Xw = X * w[:, None]
    A = X.T @ Xw + ridge * np.eye(2 * n + 1)
    A[2 * n, 2 * n] -= ridge * 0.9      # the home effect is barely shrunk
    beta = np.linalg.solve(A, Xw.T @ yc)
    off = {t: float(beta[idx[t]]) for t in teams}
    dfn = {t: float(beta[n + idx[t]]) for t in teams}
    return off, dfn, float(beta[2 * n])


def _observations(tg: pd.DataFrame) -> pd.DataFrame:
    """Offence-perspective observations: one per team-game, regular season."""
    reg = tg[tg["season_type"] == "REG"]
    return pd.DataFrame({
        "season": reg["season"].to_numpy(), "week": reg["week"].to_numpy(),
        "offence": reg["team"].to_numpy(), "defence": reg["opponent"].to_numpy(),
        "home": reg["is_home"].to_numpy(),
        **{m: reg[f"off_{m}"].to_numpy() for m in ADJ_METRICS},
    })


def rating_snapshot(tg: pd.DataFrame, season: int, week: int,
                    metrics: list[str] = ADJ_METRICS,
                    half_life: float = HALF_LIFE_WEEKS,
                    prior_discount: float = PRIOR_DISCOUNT,
                    obs: pd.DataFrame | None = None) -> pd.DataFrame:
    """The ratings in force before `week` of `season`: one row per team
    with `off_{m}_adj`, `def_{m}_adj`, `hfa_{m}` and `n_eff`, fit on the
    season's games with `week < week` plus the previous season's games at
    `prior_discount`. Works for a week no game has been played in yet
    (that is what the daily slate asks for). Empty if there is nothing
    to fit on (first season of the aggregates, week 1)."""
    obs = _observations(tg) if obs is None else obs
    this = obs[obs["season"] == season]
    prev = obs[obs["season"] == season - 1]
    teams = sorted(set(this["offence"]) | set(this["defence"]) | set(prev["offence"]))
    last_prev_week = int(prev["week"].max()) if len(prev) else 0
    cur = this[this["week"] < week].copy()
    cur["weeks_ago"] = week - cur["week"]
    old = prev.copy()
    old["weeks_ago"] = week + (last_prev_week - old["week"]) + 1
    window = pd.concat([cur, old], ignore_index=True)
    if window.empty:
        return pd.DataFrame(columns=["season", "week", "team", "n_eff"])
    window["weight"] = 0.5 ** (window["weeks_ago"] / half_life)
    window.loc[window["season"] < season, "weight"] *= prior_discount
    n_eff = window.groupby("offence")["weight"].sum()
    fits = {m: _weighted_ridge(window, m, teams) for m in metrics}
    rows = []
    for t in teams:
        row = {"season": int(season), "week": int(week), "team": t, "n_eff": float(n_eff.get(t, 0.0))}
        for m, (off, dfn, hfa) in fits.items():
            row[f"off_{m}_adj"] = off[t]
            row[f"def_{m}_adj"] = dfn[t]
            row[f"hfa_{m}"] = hfa
        rows.append(row)
    return pd.DataFrame(rows)


def adjusted_ratings(tg: pd.DataFrame, metrics: list[str] = ADJ_METRICS,
                     half_life: float = HALF_LIFE_WEEKS,
                     prior_discount: float = PRIOR_DISCOUNT) -> pd.DataFrame:
    """One row per (season, week, team) for every week in the aggregates:
    the ratings in force *before* that week's games."""
    obs = _observations(tg)
    frames = []
    for s in sorted(obs["season"].unique()):
        for W in sorted(tg.loc[tg["season"] == s, "week"].unique()):
            snap = rating_snapshot(tg, int(s), int(W), metrics, half_life, prior_discount, obs=obs)
            if len(snap):
                frames.append(snap)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# 2. unadjusted EWMA form, shifted, shrunk
# --------------------------------------------------------------------------
def ewm_form(tg: pd.DataFrame, metrics: list[str] = FORM_METRICS,
             half_life: float = HALF_LIFE_GAMES) -> pd.DataFrame:
    """Per (game_id, team): each metric's EWMA over the team's *previous*
    games (all season types, seasons chained), shrunk toward the league
    mean of the training-era rows by games played this season."""
    df = tg.sort_values(["team", "season", "week", "date"]).reset_index(drop=True)
    df = pregame.shifted_ewm(df, "team", metrics, half_life, min_periods=1, suffix="ewm")
    df["_season_games"] = df.groupby(["team", "season"]).cumcount()
    league_mean = df[metrics].mean()
    for m in metrics:
        df[f"{m}_ewm"] = pregame.shrink(df[f"{m}_ewm"], df["_season_games"],
                                        float(league_mean[m]), PRIOR_GAMES)
    return df[["game_id", "team"] + [f"{m}_ewm" for m in metrics]]


def current_form(tg: pd.DataFrame, season: int,
                 metrics: list[str] = FORM_METRICS) -> pd.DataFrame:
    """Per team: the EWMA form the team's *next* game would see — every
    game played so far, shrunk by games played in `season` (zero for a
    team that has not played in it yet). One virtual row per team is
    appended at the end of its history and read back."""
    teams = sorted(tg["team"].unique())
    virtual = pd.DataFrame({
        "game_id": [f"virtual_{t}" for t in teams], "team": teams,
        "season": season, "week": 99, "date": "9999-12-31",
    })
    for m in metrics:
        virtual[m] = np.nan
    df = pd.concat([tg[["game_id", "team", "season", "week", "date"] + metrics], virtual],
                   ignore_index=True)
    form = ewm_form(df, metrics)
    out = form[form["game_id"].str.startswith("virtual_")].drop(columns=["game_id"])
    return out.set_index("team")


def make_model(kind: str = PRODUCTION_LEARNER):
    """The unfitted production learner (`common/learners.py`)."""
    return learners.make(kind, C=PRODUCTION_C)


# --------------------------------------------------------------------------
# 3. game table
# --------------------------------------------------------------------------
def build_game_table(history: pd.DataFrame, tg: pd.DataFrame | None = None,
                     ratings: pd.DataFrame | None = None,
                     form: pd.DataFrame | None = None) -> pd.DataFrame:
    """Elo replay history rows (home_team, away_team, season, week,
    p_home, game_id, …) + every pregame feature. Rows with no efficiency
    data (pre-2002, a week the aggregates haven't reached) keep NaN
    features and are the fallback case downstream."""
    tg = load_team_games() if tg is None else tg
    ratings = adjusted_ratings(tg) if ratings is None else ratings
    form = ewm_form(tg) if form is None else form

    h = history.copy()
    h["home_key"] = h["home_team"].map(canonical)
    h["away_key"] = h["away_team"].map(canonical)
    p = h["p_home"].clip(1e-4, 1 - 1e-4)
    h["elo_logit"] = np.log(p / (1 - p))

    rate_cols = [c for c in ratings.columns if c.endswith("_adj")]
    for side, key in (("home", "home_key"), ("away", "away_key")):
        r = ratings.rename(columns={c: f"{side}_{c}" for c in rate_cols + ["n_eff"]})
        h = h.merge(r[["season", "week", "team"] + [f"{side}_{c}" for c in rate_cols + ["n_eff"]]],
                    left_on=["season", "week", key], right_on=["season", "week", "team"],
                    how="left").drop(columns=["team"])
        f = form.rename(columns={c: f"{side}_{c}" for c in form.columns if c.endswith("_ewm")})
        h = h.merge(f, left_on=["game_id", key], right_on=["game_id", "team"],
                    how="left").drop(columns=["team"])

    # Cross-unit matchups: expected value of an offence against a defence
    # is off + def (def = allowed above average), net = home − away.
    for m in ADJ_METRICS:
        if f"home_off_{m}_adj" not in h.columns:     # ratings fit on a subset
            continue
        h[f"{m}_home_vs_away"] = h[f"home_off_{m}_adj"] + h[f"away_def_{m}_adj"]
        h[f"{m}_away_vs_home"] = h[f"away_off_{m}_adj"] + h[f"home_def_{m}_adj"]
        h[f"{m}_matchup_net"] = h[f"{m}_home_vs_away"] - h[f"{m}_away_vs_home"]
    for alias, src in (("rz_td_matchup_net", "rz_td_per_trip_matchup_net"),
                       ("explosive_matchup_net", "explosive_rate_matchup_net")):
        if src in h.columns:
            h[alias] = h[src]
    return h


def all_features() -> list[str]:
    return [f for g in GROUPS.values() for f in g]


# --------------------------------------------------------------------------
# 4. production second stage
# --------------------------------------------------------------------------
def aggregates_freshness(tg: pd.DataFrame, games: pd.DataFrame,
                         tolerance_days: int = TOLERANCE_DAYS) -> freshness.FreshnessReport:
    """Are the aggregates caught up with the spine? The run date for the
    check is the newest *completed* game in the spine, so an off-season
    run reads fresh (nothing to catch up on) and an in-season run reads
    stale the moment nflverse's rebuild falls more than `tolerance_days`
    behind the results."""
    played = games[games["completed"].astype(bool)]
    anchor = str(played["date"].max()) if len(played) else "1970-01-01"
    return freshness.check("nflverse pbp aggregates (data/nfl/team_games.csv)",
                           tg, "date", anchor, tolerance_days)


class SecondStage:
    """Elo + every pregame efficiency feature -> home win probability, fit
    in-run on the replay history joined to the aggregates (2002 onward,
    ties excluded) with the production learner. `p_home` returns None
    when either side has no rating for the requested week, and the caller
    falls back to Elo."""

    def __init__(self, tg: pd.DataFrame, history: pd.DataFrame,
                 features: list[str] | None = None, learner: str = PRODUCTION_LEARNER):
        self.tg = tg
        self.features = list(features or PRODUCTION_FEATURES)
        self.learner = learner
        self._obs = _observations(tg)
        self._snapshots: dict[tuple[int, int], pd.DataFrame] = {}
        self._form: dict[int, pd.DataFrame] = {}
        ratings = adjusted_ratings(tg, metrics=PRODUCTION_METRICS)
        form = ewm_form(tg)
        table = build_game_table(history, tg, ratings, form)
        train = table[(table["season"] >= 2002) & (table["home_win"] != 0.5)]
        # Only the Elo inputs are required; a NaN efficiency column is a
        # missing feed the learner handles (median for the logistic and
        # forest, a learned direction for boosting).
        train = train[train[[f for f in self.features if f in ELO + RAW_ELO]].notna().all(axis=1)]
        self.n_train = int(len(train))
        self.seasons = (int(train["season"].min()), int(train["season"].max())) if self.n_train else None
        self.model = make_model(learner).fit(train[self.features],
                                             (train["home_win"] == 1.0).astype(int))

    def snapshot(self, season: int, week: int) -> pd.DataFrame:
        key = (int(season), int(week))
        if key not in self._snapshots:
            snap = rating_snapshot(self.tg, key[0], key[1], PRODUCTION_METRICS, obs=self._obs)
            self._snapshots[key] = snap.set_index("team") if len(snap) else snap
        return self._snapshots[key]

    def form(self, season: int) -> pd.DataFrame:
        if int(season) not in self._form:
            self._form[int(season)] = current_form(self.tg, int(season))
        return self._form[int(season)]

    def feature_row(self, home: str, away: str, p_elo: float,
                    season: int, week: int,
                    elo_home: float | None = None, elo_away: float | None = None) -> dict | None:
        snap = self.snapshot(season, week)
        form = self.form(season)
        hk, ak = canonical(home), canonical(away)
        if snap.empty or hk not in snap.index or ak not in snap.index:
            return None
        p = min(max(p_elo, 1e-4), 1 - 1e-4)
        row = {"elo_logit": float(np.log(p / (1 - p))),
               "elo_home_pre": elo_home, "elo_away_pre": elo_away}
        for side, key in (("home", hk), ("away", ak)):
            for m in PRODUCTION_METRICS:
                row[f"{side}_off_{m}_adj"] = float(snap.at[key, f"off_{m}_adj"])
                row[f"{side}_def_{m}_adj"] = float(snap.at[key, f"def_{m}_adj"])
            for m in FORM_METRICS:
                row[f"{side}_{m}_ewm"] = float(form.at[key, f"{m}_ewm"]) if key in form.index else np.nan
        for m in PRODUCTION_METRICS:
            row[f"{m}_home_vs_away"] = row[f"home_off_{m}_adj"] + row[f"away_def_{m}_adj"]
            row[f"{m}_away_vs_home"] = row[f"away_off_{m}_adj"] + row[f"home_def_{m}_adj"]
            row[f"{m}_matchup_net"] = row[f"{m}_home_vs_away"] - row[f"{m}_away_vs_home"]
        if "rz_td_per_trip_matchup_net" in row:
            row["rz_td_matchup_net"] = row["rz_td_per_trip_matchup_net"]
        if "explosive_rate_matchup_net" in row:
            row["explosive_matchup_net"] = row["explosive_rate_matchup_net"]
        return row

    def p_home(self, home: str, away: str, p_elo: float,
               season: int, week: int,
               elo_home: float | None = None, elo_away: float | None = None) -> float | None:
        row = self.feature_row(home, away, p_elo, season, week, elo_home, elo_away)
        if row is None:
            return None
        X = pd.DataFrame([row]).reindex(columns=self.features)
        if X[[f for f in self.features if f in ELO + RAW_ELO]].isna().any(axis=None):
            return None
        return float(self.model.predict_proba(X)[0, 1])
