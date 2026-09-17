"""
Do the two results-derived form candidates add anything on top of the
shipping feature set — raw goal form (`goals.py`), form as surprise
against the rating (`momentum.py`), or both?

The question asked of every candidate feature here before it ships: fit
the shipping model and the shipping model plus the candidate on the same
training seasons, score both on the same holdout, and report the paired
difference in standard errors. A feature that cannot clear its own noise
does not go in.

Paired SE, not two separate log losses: both models score the *same*
matches, so the right test is on the per-match differences in log loss.
Two independent confidence intervals on 1.018 vs 1.017 would overlap
hugely and say nothing.

Reported across several splits, because one split is one draw: a feature
that helps on 2024-25 and hurts on 2023-24 has found a season, not a
signal (the shots layer earned its place by improving all three).

Both candidates come free of new data, which is the whole reason they are
worth an experiment: goals from the result log the ratings already read,
surprise from the replay's own expectation column.

Usage:
    python -m soccer.clubs.model.eval_goals_form [--splits 2023-24 2024-25 2025-26]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from soccer.clubs.model.features import ALL_FEATURES, attach_features
from soccer.clubs.model.goals import GOAL_FEATURES, attach_goals
from soccer.clubs.model.momentum import MOMENTUM_FEATURES, attach_momentum
from soccer.clubs.model.europe import run_all_european
from soccer.clubs.model.shots import SHOT_FEATURES, attach_shots
from soccer.clubs.model.train import FEATURES, MAX_ITER
from soccer.clubs.model.xg import XG_FEATURES, attach_xg

SPLITS = ["2023-24", "2024-25", "2025-26"]


CANDIDATES = {
    "goals": GOAL_FEATURES,
    "surprise": MOMENTUM_FEATURES,
    "both": GOAL_FEATURES + MOMENTUM_FEATURES,
}


def build_table() -> pd.DataFrame:
    """The shipping feature table plus both candidate columns."""
    _, history = run_all_european()
    league_only = history[~history["league"].str.startswith("uefa:")]
    return attach_momentum(
        attach_goals(attach_shots(attach_xg(attach_features(league_only)))))


def _per_match_loss(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Log loss of each row under `model` — the paired quantity."""
    probs = model.predict_proba(frame[features])
    classes = list(model.classes_)
    taken = np.array([probs[i, classes.index(o)]
                      for i, o in enumerate(frame["outcome"])])
    return -np.log(np.clip(taken, 1e-12, None))


def evaluate(table: pd.DataFrame, split: str) -> list[dict]:
    """One row per candidate: its holdout log loss against the shipping
    set, and the paired test on the per-match differences."""
    train = table[table["season"] < split]
    test = table[table["season"] >= split]

    def fit(feats):
        model = LogisticRegression(max_iter=MAX_ITER, tol=1e-10)
        model.fit(train[feats], train["outcome"])
        return model

    base_model = fit(FEATURES)
    base_losses = _per_match_loss(base_model, test, FEATURES)
    base_ll = log_loss(test["outcome"], base_model.predict_proba(test[FEATURES]),
                       labels=list(base_model.classes_))

    rows = []
    for label, extra in CANDIDATES.items():
        feats = FEATURES + extra
        model = fit(feats)
        ll = log_loss(test["outcome"], model.predict_proba(test[feats]),
                      labels=list(model.classes_))
        # Paired: the per-match improvement from adding the candidate.
        gain = base_losses - _per_match_loss(model, test, feats)
        se = gain.std(ddof=1) / np.sqrt(len(gain))
        home = list(model.classes_).index("H")
        coefs = {f: round(model.coef_[home][feats.index(f)], 5) for f in extra}
        rows.append({
            "split": split,
            "candidate": label,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "base": base_ll,
            "with": ll,
            "delta": base_ll - ll,
            "se_units": float(gain.mean() / se) if se > 0 else float("nan"),
            "coefs": coefs,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="*", default=SPLITS)
    args = parser.parse_args()

    table = build_table()
    cover = lambda c: (table[c] != 0).mean()
    print(f"{len(table)} league matches. Coverage: "
          f"goals_net_diff {cover('goals_net_diff'):.1%}, "
          f"surprise_net_diff {cover('surprise_net_diff'):.1%}, "
          f"sot_net_diff {cover('sot_net_diff'):.1%}, "
          f"xg_net_diff {cover('xg_net_diff'):.1%}\n")

    rows = []
    for split in args.splits:
        for r in evaluate(table, split):
            rows.append(r)
            verdict = "better" if r["delta"] > 0 else "worse"
            print(f"holdout {r['split']}  +{r['candidate']:<9} "
                  f"{r['base']:.4f} -> {r['with']:.4f}  "
                  f"({verdict} by {abs(r['delta']):.4f}, "
                  f"{r['se_units']:+.1f} SE paired, n={r['n_test']})  "
                  f"{r['coefs']}")
        print()

    print("The bar this repo holds a form feature to: every split improves "
          "and the paired test clears ~+2 SE (see shots.py).")
    for label in CANDIDATES:
        mine = [r for r in rows if r["candidate"] == label]
        wins = sum(1 for r in mine if r["delta"] > 0)
        worst = min(r["se_units"] for r in mine)
        print(f"  +{label:<9} {wins}/{len(mine)} splits improved, "
              f"weakest split {worst:+.1f} SE -> "
              f"{'SHIP' if wins == len(mine) and worst >= 2.0 else 'does not clear the bar'}")


if __name__ == "__main__":
    main()
