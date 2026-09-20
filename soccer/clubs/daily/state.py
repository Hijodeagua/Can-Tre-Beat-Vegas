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
from soccer.clubs.data.leagues import LEAGUES, pool_of
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

        `keep_sides=True` is the difference from the training path: the
        model is handed exactly `FEATURES` either way, so the prediction
        is bit-identical, but the returned frame also carries each side's
        own numbers for the site's match card. Training calls the same
        functions without it and its frame is unchanged.
        """
        f = attach_context(adv.attach_advanced(
            attach_features(feature_rows, keep_sides=True),
            matches=self.adv_matches, calendar=self.calendar, keep_sides=True))
        probs = self.outcome_model.predict_proba(f[FEATURES])
        out = f.copy()
        for i, c in enumerate(self.outcome_model.classes_):
            out[f"p_{c}"] = probs[:, i]
        return out


def feed_status(run_date: str) -> dict[str, freshness.FreshnessReport]:
    """Freshness of every form feed the slate features against, on the
    run date. Tolerance is each feed's own staleness guard, so "STALE"
    here means exactly "the guard is zeroing this feature on today's
    slate and the model is running without it". The understat table is
    the advanced-metrics feed (`soccer/clubs/model/advanced.py`): logged
    so its revival is visible, not yet a production input."""
    from soccer.clubs.data import understat
    from soccer.clubs.model import advanced

    def _load(exists, load):
        return load() if exists() else None

    return {
        "xg": freshness.check("understat xG (xg_matches.csv)",
                              _load(xg.xg_available, xg.load_xg), "date", run_date, xg.MAX_AGE_DAYS),
        "shots": freshness.check("football-data shots (shots_matches.csv)",
                                 _load(shots.shots_available, shots.load_shots), "date", run_date,
                                 shots.MAX_AGE_DAYS),
        "understat_advanced": freshness.check(
            "understat advanced (understat_matches.csv)",
            _load(understat.OUT_CSV.exists, understat.load_processed), "date", run_date,
            advanced.MAX_AGE_DAYS),
    }


def build_state() -> DailyState:
    from soccer.clubs.model.elo import DATA_DIR

    engines, history = run_all_european()
    league_hist = history[~history["league"].str.startswith("uefa:")].copy()
    adv_matches = adv.match_metrics()
    calendar = adv._calendar()
    featured = attach_context(adv.attach_advanced(
        shots.attach_shots(xg.attach_xg(attach_features(league_hist))),
        matches=adv_matches, calendar=calendar))

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
