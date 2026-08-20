"""
Train the club match-outcome model: multinomial logistic regression over
{home win, draw, away win} on the venue-adjusted Elo gap, pooled across the
five leagues (the gap-to-probability curve is shared; each league's Elo pool
already carries its own tuned parameters).

Temporal validation mirrors `soccer/model/train.py`: train on every season
before SPLIT_SEASON, evaluate on SPLIT_SEASON onward — the same two seasons
`tune.py` never touched — with per-league metrics and a class-frequency
baseline for comparison.

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
from soccer.clubs.model.elo import run_all

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

FEATURES = ["elo_gap"]
SPLIT_SEASON = "2024-25"


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

    _, table = run_all()
    train = table[table["season"] < args.split_season]
    test = table[table["season"] >= args.split_season]
    print(f"Train: {len(train)} matches (< {args.split_season})")
    print(f"Test:  {len(test)} matches (>= {args.split_season})")

    model = LogisticRegression(max_iter=2000)
    model.fit(train[FEATURES], train["outcome"])

    probs = model.predict_proba(test[FEATURES])
    ll = log_loss(test["outcome"], probs, labels=list(model.classes_))
    acc = accuracy_score(test["outcome"], model.predict(test[FEATURES]))
    base_ll = frequency_baseline(train, test)

    print(f"\nElo model:          log loss {ll:.4f}  accuracy {acc:.3f}")
    print(f"Frequency baseline: log loss {base_ll:.4f}")

    rows = [
        {"league": "all", "model": "elo", "log_loss": ll, "accuracy": acc},
        {"league": "all", "model": "frequency", "log_loss": base_ll, "accuracy": np.nan},
    ]
    print("\nPer league (holdout):")
    for league in LEAGUES:
        sub = test[test["league"] == league]
        sub_probs = model.predict_proba(sub[FEATURES])
        sub_ll = log_loss(sub["outcome"], sub_probs, labels=list(model.classes_))
        sub_acc = accuracy_score(sub["outcome"], model.predict(sub[FEATURES]))
        sub_base = frequency_baseline(train[train["league"] == league], sub)
        rows.append({"league": league, "model": "elo", "log_loss": sub_ll, "accuracy": sub_acc})
        rows.append({"league": league, "model": "frequency", "log_loss": sub_base, "accuracy": np.nan})
        print(
            f"  {league:>10}: log loss {sub_ll:.4f} (freq {sub_base:.4f})  "
            f"accuracy {sub_acc:.3f}  ({len(sub)} matches)"
        )

    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / "outcome_model.pkl", "wb") as f:
        pickle.dump(
            {"model": model, "features": FEATURES, "split_season": args.split_season}, f
        )
    pd.DataFrame(rows).to_csv(ARTIFACTS / "metrics.csv", index=False)
    print(f"\nSaved model + metrics to {ARTIFACTS}")


if __name__ == "__main__":
    main()
