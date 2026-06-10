"""
Train the match-outcome model: multinomial logistic regression over
{home win, draw, away win} on the venue-adjusted Elo gap, squad-strength
differentials (when FIFA ratings are uploaded), host flag, and a
knockout-stage indicator.

Temporal validation: train < SPLIT_DATE, evaluate on everything after,
with an Elo-only baseline for comparison.

Usage:
    python -m soccer.model.train [--split 2024-01-01] [--start 2006-01-01]
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from soccer.model.elo import run_history
from soccer.model.squad import attach_squad_features, ratings_available

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

FEATURES = ["elo_gap", "depth_diff_z", "star_diff_z", "host_home", "knockout"]
CLASSES = ["A", "D", "H"]  # away win, draw, home win (sklearn sorts labels)

KNOCKOUT_TOURNAMENTS = {"FIFA World Cup"} | {
    "UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup", "Gold Cup",
}


def build_table(start: str) -> pd.DataFrame:
    _, history = run_history(start=start)
    history = attach_squad_features(history)
    # Group games at finals are also flagged; a true stage column needs a
    # fixtures source with round labels — good enough as a v1 proxy.
    history["knockout"] = history["tournament"].isin(KNOCKOUT_TOURNAMENTS).astype(int)
    history["host_home"] = history["host_home"].astype(int)
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--split", default="2024-01-01")
    args = parser.parse_args()

    table = build_table(args.start)
    train = table[table["date"] < args.split]
    test = table[table["date"] >= args.split]
    print(f"Train: {len(train)} matches (< {args.split})")
    print(f"Test:  {len(test)} matches (>= {args.split})")

    if not ratings_available():
        print("No FIFA ratings uploaded — squad features are 0, model is Elo-only.")

    model = LogisticRegression(max_iter=2000)
    model.fit(train[FEATURES], train["outcome"])

    probs = model.predict_proba(test[FEATURES])
    preds = model.predict(test[FEATURES])
    ll = log_loss(test["outcome"], probs, labels=list(model.classes_))
    acc = accuracy_score(test["outcome"], preds)

    # Elo-only baseline: same model class, gap feature only
    base = LogisticRegression(max_iter=2000)
    base.fit(train[["elo_gap"]], train["outcome"])
    base_probs = base.predict_proba(test[["elo_gap"]])
    base_ll = log_loss(test["outcome"], base_probs, labels=list(base.classes_))
    base_acc = accuracy_score(test["outcome"], base.predict(test[["elo_gap"]]))

    print(f"\nFull model:    log loss {ll:.4f}  accuracy {acc:.3f}")
    print(f"Elo-only base: log loss {base_ll:.4f}  accuracy {base_acc:.3f}")
    print("\nCoefficients (per class):")
    coef = pd.DataFrame(model.coef_, index=model.classes_, columns=FEATURES)
    print(coef.round(4).to_string())

    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / "outcome_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": FEATURES, "split": args.split}, f)
    metrics = pd.DataFrame(
        [
            {"model": "full", "log_loss": ll, "accuracy": acc},
            {"model": "elo_only", "log_loss": base_ll, "accuracy": base_acc},
        ]
    )
    metrics.to_csv(ARTIFACTS / "metrics.csv", index=False)
    print(f"\nSaved model + metrics to {ARTIFACTS}")


if __name__ == "__main__":
    main()
