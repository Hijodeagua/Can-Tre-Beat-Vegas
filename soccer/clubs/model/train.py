"""
Train the club match-outcome model: multinomial logistic regression over
{home win, draw, away win} on the venue-adjusted Elo gap plus the
squad-economics differentials, pooled across the five leagues (the
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

from soccer.clubs.data.leagues import LEAGUES
from soccer.clubs.model.europe import run_all_european
from soccer.clubs.model.features import (
    ALL_FEATURES,
    attach_features,
    transfers_available,
    values_available,
)

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

FEATURES = ["elo_gap"] + ALL_FEATURES
SPLIT_SEASON = "2024-25"


def build_table() -> pd.DataFrame:
    _, history = run_all_european()
    league_only = history[~history["league"].str.startswith("uefa:")]
    return attach_features(league_only)


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

    model = LogisticRegression(max_iter=2000)
    model.fit(train[FEATURES], train["outcome"])
    probs = model.predict_proba(test[FEATURES])
    ll = log_loss(test["outcome"], probs, labels=list(model.classes_))
    acc = accuracy_score(test["outcome"], model.predict(test[FEATURES]))

    elo_only = LogisticRegression(max_iter=2000)
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
    print("\nPer league (holdout):")
    for league in LEAGUES:
        sub = test[test["league"] == league]
        sub_probs = model.predict_proba(sub[FEATURES])
        sub_ll = log_loss(sub["outcome"], sub_probs, labels=list(model.classes_))
        sub_acc = accuracy_score(sub["outcome"], model.predict(sub[FEATURES]))
        rows.append({"league": league, "model": "full", "log_loss": sub_ll, "accuracy": sub_acc})
        print(f"  {league:>10}: log loss {sub_ll:.4f}  accuracy {sub_acc:.3f}  ({len(sub)} matches)")

    print("\nCoefficients (per class):")
    print(pd.DataFrame(model.coef_, index=model.classes_, columns=FEATURES).round(4).to_string())

    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / "outcome_model.pkl", "wb") as f:
        pickle.dump(
            {"model": model, "features": FEATURES, "split_season": args.split_season}, f
        )
    pd.DataFrame(rows).to_csv(ARTIFACTS / "metrics.csv", index=False)
    print(f"\nSaved model + metrics to {ARTIFACTS}")


if __name__ == "__main__":
    main()
