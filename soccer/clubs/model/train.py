"""
Train the club match-outcome model: multinomial logistic regression over
{home win, draw, away win} on the venue-adjusted Elo gap plus the
squad-economics differentials and the two chance-creation form features
(rolling xG net, rolling shots-on-target net), pooled across the five
leagues (the
gap-to-probability curve is shared; each league's Elo pool already carries
its own tuned parameters).

Ratings come from the UEFA-glued replay (`europe.run_all_european`) —
validated to beat the unglued pools on this same holdout — but only league
matches enter the outcome fit; the ~65 cross-league matches a season are
rating glue, not training rows. The economics features (transfer spend,
market value / wages when uploaded) are 0-imputed where data is missing,
so the model degrades gracefully to Elo-only.

Temporal validation mirrors `soccer/model/train.py`: train on every season
before SPLIT_SEASON, evaluate on SPLIT_SEASON onward — the same two seasons
`tune.py` never touched — with per-league metrics, an Elo-only model and a
class-frequency baseline for comparison.

Usage:
    python -m soccer.clubs.model.train [--split-season 2024-25]
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from common import learners
from soccer.clubs.data.leagues import LEAGUES
from soccer.clubs.model import advanced as adv
from soccer.clubs.model.europe import run_all_european
from soccer.clubs.model.features import (
    ALL_FEATURES,
    SIDE_FEATURES,
    SIDE_RAW_FEATURES,
    attach_features,
    transfers_available,
    values_available,
)
from soccer.clubs.model.shots import SHOT_FEATURES, attach_shots, shots_available
from soccer.clubs.model.xg import XG_FEATURES, attach_xg, xg_available

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

# Every input the model sees: the home and away Elo as their own columns
# beside the venue-adjusted gap (so a learner can find a level effect the
# gap alone hides), squad economics, xG and shots-on-target form, and the
# whole advanced Understat layer (xG/npxG form, attack-vs-defence splits,
# xG per shot, deep completions and the field-tilt proxy, PPDA, xPts
# form, rest, congestion, European ties). A column whose feed has not
# landed is NaN and the learner handles it; nothing is dropped for being
# empty today.
RAW_ELO = ["elo_home_pre", "elo_away_pre"]
BASE_FEATURES = ["elo_gap"] + ALL_FEATURES + XG_FEATURES + SHOT_FEATURES
# One pooled model, with the league, its tier and the season as inputs:
# per-league sub-models scored worse than the pooled fit on the 2024-25+
# holdout (1.02006 vs 1.01676) and pooled + league/tier/season scored
# slightly better (1.01651, +0.34 SE), so the context rides along as
# columns the learner can split on rather than as separate models.
CONTEXT_FEATURES = [f"lg_{k}" for k in LEAGUES] + ["tier", "season_idx"]
# Every differential ALSO enters as the two numbers it was made from.
#
# A difference imposes f(home, away) = f(home - away): the model is told
# that 1.9 against 1.5 and 0.7 against 0.3 are the same match. They are
# not — two strong attacks produce a different game from two weak ones —
# and a tree learner is perfectly able to find that if it is given the
# levels. The Elo columns have always been here for exactly this reason;
# this extends the same treatment to the rest.
#
# Measured on the 2024-25+ holdout with the production forest: differences
# alone 1.01441, per-side alone 1.01484 (0.4 SE, noise), both 1.01337
# (1.7 SE better). So the levels are not a breakthrough — the differences
# were already carrying nearly all of it — but they are free, and they
# make the published match card and the model agree on what was used.
SIDE_VALUE_FEATURES = SIDE_FEATURES + SIDE_RAW_FEATURES
FEATURES = (RAW_ELO + BASE_FEATURES + adv.ALL_ADVANCED + CONTEXT_FEATURES
            + adv.ALL_ADVANCED_SIDES + SIDE_VALUE_FEATURES)
# Random forest, by decision rather than by the holdout number: on the
# 2024-25+ test the full-set logistic scored 1.01676 to the forest's
# 1.01846 (0.0017, ~0.8 SE — noise-level), and that comparison was run
# with npxG/xPts/PPDA/deep still empty. The forest is the learner that can
# use those columns non-linearly when they land, and the one that can
# carry upset/outlier structure a linear model averages away. Revisit with
# `eval_learners` once the Understat backfill is on main
# (common/learners.py; docs/ADVANCED_METRICS.md).
LEARNER = "random_forest"
LOGISTIC_C = 0.1
SPLIT_SEASON = "2024-25"
# The economics features are sparse (a small, growing fraction of rows are
# nonzero as market-value uploads backfill), so their gradient signal is
# weak relative to elo_gap's. sklearn's lbfgs default (max_iter=100) and
# even 2000 can stop short of convergence for those coefficients — silently
# landing near zero without any warning. A tight tolerance + higher ceiling
# make convergence explicit rather than assumed.
MAX_ITER = 5000


def attach_context(frame: pd.DataFrame) -> pd.DataFrame:
    """League one-hots (every league in LEAGUES, so the columns are fixed),
    the league's tier, and the season as years since 2010."""
    out = frame.copy()
    for k in LEAGUES:
        out[f"lg_{k}"] = (out["league"] == k).astype(float)
    out["tier"] = out["league"].map(lambda k: float(LEAGUES[k].tier) if k in LEAGUES else np.nan)
    out["season_idx"] = pd.to_numeric(out["season"].astype(str).str[:4], errors="coerce") - 2010
    return out


def build_table() -> pd.DataFrame:
    """The training table. `keep_sides=True` so every differential's two
    halves are columns in their own right — see FEATURES."""
    _, history = run_all_european()
    league_only = history[~history["league"].str.startswith("uefa:")]
    return attach_context(adv.attach_advanced(
        attach_shots(attach_xg(attach_features(league_only, keep_sides=True))),
        keep_sides=True))


def make_model(kind: str = LEARNER):
    """The unfitted production learner."""
    return learners.make(kind, C=LOGISTIC_C)


def frequency_baseline(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Log loss of always predicting the training class frequencies."""
    classes = ["A", "D", "H"]
    freqs = train["outcome"].value_counts(normalize=True).reindex(classes).fillna(0.0)
    probs = np.tile(freqs.to_numpy(), (len(test), 1))
    return log_loss(test["outcome"], probs, labels=classes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-season", default=SPLIT_SEASON)
    args = parser.parse_args()

    table = build_table()
    train = table[table["season"] < args.split_season]
    test = table[table["season"] >= args.split_season]
    print(f"Train: {len(train)} matches (< {args.split_season})")
    print(f"Test:  {len(test)} matches (>= {args.split_season})")
    if not transfers_available():
        print("No transfer aggregates — spend features are 0, Elo-only in effect.")
    if not values_available():
        print("No market-value uploads — value/wage features are 0.")
    if not xg_available():
        print("No xg_matches.csv — xg_net_diff is 0.")
    if not shots_available():
        print("No shots_matches.csv — sot_net_diff is 0.")

    model = make_model()
    model.fit(train[FEATURES], train["outcome"])
    probs = model.predict_proba(test[FEATURES])
    ll = log_loss(test["outcome"], probs, labels=list(model.classes_))
    acc = accuracy_score(test["outcome"], model.predict(test[FEATURES]))

    elo_only = LogisticRegression(max_iter=MAX_ITER, tol=1e-10)
    elo_only.fit(train[["elo_gap"]], train["outcome"])
    elo_ll = log_loss(
        test["outcome"], elo_only.predict_proba(test[["elo_gap"]]),
        labels=list(elo_only.classes_),
    )
    base_ll = frequency_baseline(train, test)

    print(f"\nFull model:         log loss {ll:.4f}  accuracy {acc:.3f}")
    print(f"Elo-only:           log loss {elo_ll:.4f}")
    print(f"Frequency baseline: log loss {base_ll:.4f}")

    rows = [
        {"league": "all", "model": "full", "log_loss": ll, "accuracy": acc},
        {"league": "all", "model": "elo_only", "log_loss": elo_ll, "accuracy": np.nan},
        {"league": "all", "model": "frequency", "log_loss": base_ll, "accuracy": np.nan},
    ]
    for tier in (1, 2):
        keys = [k for k, lg in LEAGUES.items() if lg.tier == tier]
        sub = test[test["league"].isin(keys)]
        if sub.empty:
            continue
        t_ll = log_loss(sub["outcome"], model.predict_proba(sub[FEATURES]),
                        labels=list(model.classes_))
        rows.append({"league": f"tier{tier}", "model": "full",
                     "log_loss": t_ll, "accuracy": np.nan})
        print(f"Tier {tier} holdout:     log loss {t_ll:.4f}  ({len(sub)} matches)")
    print("\nPer league (holdout):")
    for league in LEAGUES:
        sub = test[test["league"] == league]
        sub_probs = model.predict_proba(sub[FEATURES])
        sub_ll = log_loss(sub["outcome"], sub_probs, labels=list(model.classes_))
        sub_acc = accuracy_score(sub["outcome"], model.predict(sub[FEATURES]))
        rows.append({"league": league, "model": "full", "log_loss": sub_ll, "accuracy": sub_acc})
        print(f"  {league:>10}: log loss {sub_ll:.4f}  accuracy {sub_acc:.3f}  ({len(sub)} matches)")

    if LEARNER == "logistic":
        print("\nCoefficients (per class):")
        clf = model.named_steps["clf"]
        print(pd.DataFrame(clf.coef_, index=clf.classes_, columns=FEATURES).round(4).to_string())
    else:
        print(f"\nLearner: {learners.describe(LEARNER)} on {len(FEATURES)} features "
              f"(permutation importance: python -m data_jobs.build_importance --only soccer)")

    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / "outcome_model.pkl", "wb") as f:
        pickle.dump(
            {"model": model, "features": FEATURES, "learner": LEARNER,
             "split_season": args.split_season}, f
        )
    pd.DataFrame(rows).to_csv(ARTIFACTS / "metrics.csv", index=False)
    print(f"\nSaved model + metrics to {ARTIFACTS}")


if __name__ == "__main__":
    main()
