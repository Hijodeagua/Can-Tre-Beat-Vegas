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
from common import freshness
from soccer.clubs.daily import scoring
from soccer.clubs.daily.config import (
    FEED_TOLERANCE_IN_SEASON_DAYS,
    RESULTS_GRACE_DAYS,
    RESULTS_OVERDUE_LIST,
    RESULTS_OVERDUE_MAX,
    SEASON_ACTIVE_LOOKBACK_DAYS,
)
from soccer.clubs.data.leagues import LEAGUES, TIER1, pool_of
from soccer.clubs.model import advanced as adv
from soccer.clubs.model import shots, xg
from soccer.clubs.model.elo import ClubEloEngine
from soccer.clubs.model.europe import run_all_european
from soccer.clubs.model.features import attach_features
from soccer.clubs.model.train import FEATURES, LEARNER, attach_context, make_model

CLASSES = ["A", "D", "H"]


@dataclass
class DailyState:
    engines: dict[str, ClubEloEngine]  # keyed by pool (tier-1 league key)
    history: pd.DataFrame          # league rows only, features attached
    results: pd.DataFrame          # raw results.csv incl. unplayed fixtures
    outcome_model: object          # the fitted production learner (train.LEARNER)
    score_params: scoring.ScoreParams
    xg_form: "xg._Form"            # rolling xG state after all committed matches
    shot_form: "shots._Form"       # rolling shots-on-target state, same posture
    adv_matches: pd.DataFrame = None   # match-metrics table the advanced form is built from
    calendar: pd.DataFrame = None      # every dated club appearance, for rest/congestion

    def feature_row(self, league: str, home: str, away: str,
                    season: str, neutral: bool = False,
                    date: str | None = None) -> dict:
        """Pre-match features for one fixture, from current ratings.
        `date` anchors both form feeds' staleness guards; without one
        the form reads as of today, which is what a live slate wants."""
        from datetime import date as _date

        from soccer.clubs.model.elo import expected_score

        e = self.engines[pool_of(league)]
        r_home = e.rating_for(home, league)
        r_away = e.rating_for(away, league)
        adv = 0.0 if neutral else e.home_advantage
        asof = date or _date.today().isoformat()
        row = {
            "league": league,
            "season": season,
            "home_team": home,
            "away_team": away,
            "elo_home_pre": r_home,
            "elo_away_pre": r_away,
            "elo_gap": (r_home + adv) - r_away,
            "exp_home": expected_score(r_home + adv, r_away),
            "xg_net_diff": xg.slate_diff(self.xg_form, league, home, away, asof),
            "sot_net_diff": shots.slate_diff(self.shot_form, league, home, away, asof),
        }
        # Each side's own rolling form, not just the gap between them. The
        # model only ever sees the differential, but a published match
        # card has to be able to say whether "+0.4 xG" is two good attacks
        # or two bad ones. None (short of the warm-up minimum, or stale)
        # becomes NaN rather than 0: "no reading" and "level" are
        # different claims, and only the differential is entitled to
        # collapse them.
        for side, team in (("home", home), ("away", away)):
            for name, form in (("xg_net", self.xg_form),
                               ("sot_net", self.shot_form)):
                v = form.net(league, team, asof)
                row[f"{side}_{name}"] = float("nan") if v is None else v
        return row

    def outcome_probs(self, feature_rows: pd.DataFrame) -> pd.DataFrame:
        """P(A), P(D), P(H) columns for a frame of feature rows, plus every
        per-side value the features were built from.

        The advanced form columns are attached here from each club's
        earlier matches (the fixture itself carries no metrics), with the
        same staleness rule the training table used.

        `keep_sides=True` matches the training frame — `FEATURES` names
        the per-side columns as well as the differentials, so a slate
        frame without them cannot be scored at all. The returned frame
        also keeps the columns the site's match card reads.
        """
        f = attach_context(adv.attach_advanced(
            attach_features(feature_rows, keep_sides=True),
            matches=self.adv_matches, calendar=self.calendar, keep_sides=True))
        probs = self.outcome_model.predict_proba(f[FEATURES])
        out = f.copy()
        for i, c in enumerate(self.outcome_model.classes_):
            out[f"p_{c}"] = probs[:, i]
        return out


def season_active(results: pd.DataFrame, run_date: str,
                  lookback_days: int = SEASON_ACTIVE_LOOKBACK_DAYS) -> bool:
    """Whether league football is being played around the run date: any
    top-flight fixture (played or not) dated inside the trailing window.
    Fixture lists are published for the whole season up front, so this
    reads true through a results outage - which is the point."""
    from datetime import date as _date, timedelta
    start = (_date.fromisoformat(run_date) - timedelta(days=lookback_days)).isoformat()
    top = results[results["league"].isin([k for k in TIER1 if LEAGUES[k].source == "openfootball"])]
    window = top[(top["date"] >= start) & (top["date"] <= run_date)]
    return bool(len(window))


def feed_status(run_date: str, results: pd.DataFrame | None = None) -> dict[str, freshness.FreshnessReport]:
    """Freshness of every form feed the slate features against, on the
    run date.

    The tolerance each feed is reported against is its own staleness
    guard off-season (the guard is what zeroes the feature, and it spans
    a summer break), but FEED_TOLERANCE_IN_SEASON_DAYS while league
    fixtures are being played: a guard sized for the summer let a
    six-week-old shots feed read "fresh" in October 2026. The guard
    itself - a model parameter - is not touched here; only what the log,
    the site JSON and the weekly review call fresh. The understat table
    is the advanced-metrics feed (`soccer/clubs/model/advanced.py`):
    logged so its revival is visible, not yet a production input."""
    from soccer.clubs.data import understat
    from soccer.clubs.model import advanced
    from soccer.clubs.model.elo import DATA_DIR

    def _load(exists, load):
        return load() if exists() else None

    if results is None:
        results = pd.read_csv(DATA_DIR / "results.csv", usecols=["date", "league"])
    in_season = season_active(results, run_date)

    def tol(guard: int) -> int:
        return min(guard, FEED_TOLERANCE_IN_SEASON_DAYS) if in_season else guard

    return {
        "xg": freshness.check("understat xG (xg_matches.csv)",
                              _load(xg.xg_available, xg.load_xg), "date", run_date,
                              tol(xg.MAX_AGE_DAYS)),
        "shots": freshness.check("football-data shots (shots_matches.csv)",
                                 _load(shots.shots_available, shots.load_shots), "date", run_date,
                                 tol(shots.MAX_AGE_DAYS)),
        "understat_advanced": freshness.check(
            "understat advanced (understat_matches.csv)",
            _load(understat.OUT_CSV.exists, understat.load_processed), "date", run_date,
            tol(advanced.MAX_AGE_DAYS)),
    }


def results_status(results: pd.DataFrame, run_date: str,
                   grace_days: int = RESULTS_GRACE_DAYS,
                   overdue_max: int = RESULTS_OVERDUE_MAX) -> dict:
    """Per-league results freshness on the run date, built from the
    fixtures that should already have a score.

    Returns {"as_of": newest played date across the openfootball leagues,
    "run_date", "stale_leagues": [...], "thresholds": {...}, "leagues":
    {key: {name, tier, season, newest_played, newest_played_age_days,
    played_season, fixtures_season, overdue, oldest_overdue, stale,
    reason}}}. A league with no fixtures upstream for the season (the
    second divisions at 2026-27 launch) has nothing to be overdue and
    reads not stale with reason "no_fixtures"; MLS, a completed-match
    log with no fixture list, is reported for its newest result only.
    See config.py for the thresholds and why they are what they are."""
    from datetime import date as _date, timedelta

    from soccer.clubs.data.leagues import current_season_for

    run = _date.fromisoformat(run_date)
    cutoff = (run - timedelta(days=grace_days)).isoformat()
    df = results[["date", "season", "league", "home_score", "away_score"]].copy()
    results = results[["date", "season", "league", "home_team", "away_team", "home_score"]]
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    out: dict = {"run_date": run_date, "as_of": None, "stale_leagues": [],
                 "thresholds": {"grace_days": grace_days, "overdue_max": overdue_max},
                 "leagues": {}}
    for key, lg in LEAGUES.items():
        season = current_season_for(key, run_date)
        sub = df[(df["league"] == key)]
        season_rows = sub[sub["season"].astype(str) == season]
        played = sub[sub["played"] & (sub["date"] <= run_date)]
        newest = played["date"].max() if len(played) else None
        age = (run - _date.fromisoformat(newest)).days if newest else None
        overdue_rows = results[(results["league"] == key)
                               & (results["season"].astype(str) == season)
                               & results["home_score"].isna() & (results["date"] <= cutoff)]
        overdue_rows = overdue_rows.sort_values("date")
        oldest = overdue_rows["date"].min() if len(overdue_rows) else None
        entry = {
            "name": lg.name, "tier": lg.tier, "season": season,
            "newest_played": newest, "newest_played_age_days": age,
            "played_season": int(season_rows["played"].sum()),
            "fixtures_season": int(len(season_rows)),
            "overdue": int(len(overdue_rows)), "oldest_overdue": oldest,
            "overdue_fixtures": [f"{r.date} {r.home_team} v {r.away_team}"
                                 for r in overdue_rows.head(RESULTS_OVERDUE_LIST).itertuples()],
            "stale": False, "reason": None,
        }
        if lg.source != "openfootball":
            entry["reason"] = "no_fixture_list"
        elif len(season_rows) == 0:
            entry["reason"] = "no_fixtures"
        elif len(overdue_rows) > overdue_max:
            entry["stale"] = True
            entry["reason"] = (f"{len(overdue_rows)} fixtures dated on or before {cutoff} "
                               f"have no result (limit {overdue_max}); newest result {newest}")
        if entry["stale"]:
            out["stale_leagues"].append(key)
        if lg.source == "openfootball" and newest and (out["as_of"] is None or newest > out["as_of"]):
            out["as_of"] = newest
        out["leagues"][key] = entry
    return out


def build_state() -> DailyState:
    from soccer.clubs.model.elo import DATA_DIR

    engines, history = run_all_european()
    league_hist = history[~history["league"].str.startswith("uefa:")].copy()
    adv_matches = adv.match_metrics()
    calendar = adv._calendar()
    # keep_sides=True on the training frame too: FEATURES now names the
    # per-side columns as well as the differentials, and this in-run refit
    # has to produce the same frame `model/train.py` fits offline.
    featured = attach_context(adv.attach_advanced(
        shots.attach_shots(xg.attach_xg(
            attach_features(league_hist, keep_sides=True))),
        matches=adv_matches, calendar=calendar, keep_sides=True))

    model = make_model(LEARNER)
    model.fit(featured[FEATURES], featured["outcome"])

    results = pd.read_csv(DATA_DIR / "results.csv")
    return DailyState(
        engines=engines,
        history=featured,
        results=results,
        outcome_model=model,
        score_params=scoring.fit(league_hist),
        xg_form=xg.current_form(),
        shot_form=shots.current_form(),
        adv_matches=adv_matches,
        calendar=calendar,
    )
