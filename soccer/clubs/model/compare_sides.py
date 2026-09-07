"""Sides vs. differences: does the club outcome model do better when it
sees each feature as its home value and its away value instead of one
home-minus-away number?

The production model (`train.py`, `daily/state.py`) is a multinomial
logistic over six differentials: the venue-adjusted Elo gap, four
squad-economics differentials and the xG-form differential. This script
builds the same rows with every one of those split back into its two
sides — home Elo, away Elo, home spend z, away spend z, and so on — and
fits both feature sets on identical training rows, walk-forward one
season at a time, so the comparison is out of sample everywhere.

Two learners, so the answer does not hinge on linearity:

- multinomial logistic (the production learner);
- LightGBM multiclass, which can let the two sides interact in ways a
  linear model cannot (a strong home side vs. a weak away side need not
  be worth the same as the reverse gap).

Feature sets compared (DIFFS is exactly the production FEATURES):

    DIFFS   elo_gap, spend_diff_z, net_diff_z, value_diff_z, wage_diff_z, xg_net_diff
    SIDES   elo_home_adj, elo_away_pre, home_spend_z, away_spend_z, home_net_z,
            away_net_z, home_value_z, away_value_z, home_wage_z, away_wage_z,
            home_xg_net, away_xg_net
    BOTH    DIFFS + SIDES

`elo_home_adj` is the home rating plus the league's home advantage — the
same venue adjustment the gap carries — so SIDES has every number DIFFS
has, split rather than subtracted. Elo-only versions of each (gap alone;
home + away alone) are reported too.

Outputs `artifacts/sides_vs_diffs.csv` (per test season and overall, per
learner and feature set) and, for the record, the fully trained sides
model on every row as `artifacts/outcome_model_sides.pkl`.

    python -m soccer.clubs.model.compare_sides [--first-test-season 2018-19]
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from soccer.clubs.model import xg
from soccer.clubs.model.europe import run_all_european
from soccer.clubs.model.features import (
    ALL_FEATURES, _load_transfer_z, _load_value_z, attach_features,
    transfers_available, values_available,
)
from soccer.clubs.model.train import MAX_ITER

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
CLASSES = ["A", "D", "H"]

DIFFS = ["elo_gap"] + ALL_FEATURES + xg.XG_FEATURES
SIDES = ["elo_home_adj", "elo_away_pre",
         "home_spend_z", "away_spend_z", "home_net_z", "away_net_z",
         "home_value_z", "away_value_z", "home_wage_z", "away_wage_z",
         "home_xg_net", "away_xg_net"]
FEATURE_SETS = {
    "diffs": DIFFS,
    "sides": SIDES,
    "both": DIFFS + SIDES,
    "elo gap only": ["elo_gap"],
    "elo home + away only": ["elo_home_adj", "elo_away_pre"],
}


# --- side features ---------------------------------------------------------

def _attach_sides(history: pd.DataFrame, table: pd.DataFrame,
                  z_cols: list[str]) -> pd.DataFrame:
    """Join a (league, season, club) -> z table on both sides and keep the
    sides (features.py's _attach_diff subtracts and drops them)."""
    for side in ("home", "away"):
        renames = {z: f"{side}_{z}" for z in z_cols}
        history = history.merge(
            table.rename(columns={"club": f"{side}_team", **renames}),
            on=["league", "season", f"{side}_team"], how="left")
        for z in z_cols:
            history[f"{side}_{z}"] = history[f"{side}_{z}"].fillna(0.0)
    return history


def _attach_xg_sides(history: pd.DataFrame) -> pd.DataFrame:
    """Each side's own rolling xG net, under the same availability rule as
    `xg_net_diff` (both sides must have form, else both are 0) so the two
    feature sets see exactly the same matches as informative."""
    history = history.copy()
    if not xg.xg_available():
        history["home_xg_net"] = history["away_xg_net"] = 0.0
        return history
    table = xg.load_xg()
    by_key = {(r.league, r.date, r.home_team, r.away_team): (r.xg_home, r.xg_away)
              for r in table.itertuples()}
    order = history["date"].astype(str).argsort(kind="stable")
    form = xg._Form()
    home_vals = pd.Series(0.0, index=history.index)
    away_vals = pd.Series(0.0, index=history.index)
    for i in order:
        row = history.iloc[i]
        h = form.net(row["league"], row["home_team"], row["date"])
        a = form.net(row["league"], row["away_team"], row["date"])
        if h is not None and a is not None:
            home_vals.iloc[i], away_vals.iloc[i] = h, a
        hit = by_key.get((row["league"], row["date"], row["home_team"], row["away_team"]))
        if hit is not None:
            form.push(row["league"], row["home_team"], row["away_team"], row["date"], *hit)
    history["home_xg_net"], history["away_xg_net"] = home_vals, away_vals
    return history


def build_table() -> pd.DataFrame:
    """League rows with both the production differentials and the sides."""
    _, history = run_all_european()
    league_only = history[~history["league"].str.startswith("uefa:")].copy()
    t = xg.attach_xg(attach_features(league_only))
    # Venue-adjusted home rating: the gap already carries the league's home
    # advantage, so recover it rather than re-deriving per league.
    t["elo_home_adj"] = t["elo_home_pre"] + (t["elo_gap"] - (t["elo_home_pre"] - t["elo_away_pre"]))
    if transfers_available():
        t = _attach_sides(t, _load_transfer_z(), ["spend_z", "net_z"])
    else:
        t[["home_spend_z", "away_spend_z", "home_net_z", "away_net_z"]] = 0.0
    if values_available():
        t = _attach_sides(t, _load_value_z(), ["value_z", "wage_z"])
    else:
        t[["home_value_z", "away_value_z", "home_wage_z", "away_wage_z"]] = 0.0
    t = _attach_xg_sides(t)
    # Sanity: the sides must reproduce every differential exactly.
    assert np.allclose(t["home_spend_z"] - t["away_spend_z"], t["spend_diff_z"])
    assert np.allclose(t["home_value_z"] - t["away_value_z"], t["value_diff_z"])
    assert np.allclose(t["home_xg_net"] - t["away_xg_net"], t["xg_net_diff"])
    assert np.allclose(t["elo_home_adj"] - t["elo_away_pre"], t["elo_gap"])
    return t


# --- learners ----------------------------------------------------------------

def fit_logistic(X: np.ndarray, y: np.ndarray):
    return LogisticRegression(max_iter=MAX_ITER, tol=1e-10).fit(X, y)


def fit_lgbm(X: np.ndarray, y: np.ndarray):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=300, learning_rate=0.03,
                           num_leaves=15, min_child_samples=100, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
                           verbose=-1, random_state=7)
    return m.fit(X, y)


LEARNERS = {"logistic": fit_logistic, "lightgbm": fit_lgbm}


def _per_row_ll(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = model.predict_proba(X)
    order = [list(model.classes_).index(c) for c in CLASSES]
    p = np.clip(p[:, order], 1e-9, 1.0)
    idx = np.array([CLASSES.index(c) for c in y])
    return -np.log(p[np.arange(len(y)), idx])


# --- the comparison ----------------------------------------------------------

def walk_forward(table: pd.DataFrame, first_test: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(table["season"].unique())
    tests = [s for s in seasons if s >= first_test]
    metrics, paired = [], []
    for learner, fitter in LEARNERS.items():
        per_row: dict[str, list[np.ndarray]] = {k: [] for k in FEATURE_SETS}
        ys, tags = [], []
        for season in tests:
            train = table[table["season"] < season]
            test = table[table["season"] == season]
            if test.empty or len(train) < 2000:
                continue
            ys.append(test["outcome"].to_numpy())
            tags.append(test[["season", "league"]])
            for name, feats in FEATURE_SETS.items():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = fitter(train[feats].to_numpy(), train["outcome"].to_numpy())
                ll = _per_row_ll(m, test[feats].to_numpy(), test["outcome"].to_numpy())
                per_row[name].append(ll)
                pred = m.predict(test[feats].to_numpy())
                metrics.append({"learner": learner, "features": name, "season": season,
                                "n": len(test), "log_loss": float(ll.mean()),
                                "accuracy": float((pred == test["outcome"]).mean())})
                print(f"  {learner:9s} {name:22s} {season:8s} n={len(test):5d} "
                      f"ll={ll.mean():.4f}", flush=True)
        y_all = np.concatenate(ys)
        tag = pd.concat(tags, ignore_index=True)
        rows = {k: np.concatenate(v) for k, v in per_row.items()}
        for name in FEATURE_SETS:
            metrics.append({"learner": learner, "features": name, "season": "ALL",
                            "n": len(y_all), "log_loss": float(rows[name].mean()),
                            "accuracy": np.nan})
        # Paired deltas vs. the production feature set, overall and per league.
        for name in FEATURE_SETS:
            if name == "diffs":
                continue
            d = rows[name] - rows["diffs"]
            paired.append({"learner": learner, "features": name, "league": "ALL",
                           "n": len(d), "d_ll_mean": float(d.mean()),
                           "d_ll_se": float(d.std(ddof=1) / np.sqrt(len(d)))})
            for league, sub in tag.groupby("league"):
                dd = d[sub.index.to_numpy()]
                paired.append({"learner": learner, "features": name, "league": league,
                               "n": len(dd), "d_ll_mean": float(dd.mean()),
                               "d_ll_se": float(dd.std(ddof=1) / np.sqrt(len(dd)))})
    return pd.DataFrame(metrics), pd.DataFrame(paired)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-test-season", default="2018-19")
    args = ap.parse_args()

    table = build_table()
    print(f"{len(table)} league matches, seasons {table['season'].min()}..{table['season'].max()}")
    metrics, paired = walk_forward(table, args.first_test_season)

    overall = metrics[metrics["season"] == "ALL"].pivot(index="features", columns="learner",
                                                        values="log_loss")
    print("\nOut-of-sample log loss, every test season pooled:")
    print(overall.round(4).to_string())
    print("\nPaired Δ log loss vs. the production differentials (negative = better), ± 1 SE:")
    p = paired[paired["league"] == "ALL"].copy()
    p["delta"] = p.apply(lambda r: f"{r['d_ll_mean']:+.4f} ± {r['d_ll_se']:.4f}", axis=1)
    print(p.pivot(index="features", columns="learner", values="delta").to_string())

    ARTIFACTS.mkdir(exist_ok=True)
    metrics.to_csv(ARTIFACTS / "sides_vs_diffs.csv", index=False)
    paired.to_csv(ARTIFACTS / "sides_vs_diffs_paired.csv", index=False)

    # The fully trained sides model, on every row, for the record.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sides_model = fit_logistic(table[SIDES].to_numpy(), table["outcome"].to_numpy())
    with open(ARTIFACTS / "outcome_model_sides.pkl", "wb") as f:
        pickle.dump({"model": sides_model, "features": SIDES}, f)
    print("\nSides model coefficients (per class), fit on every row:")
    print(pd.DataFrame(sides_model.coef_, index=sides_model.classes_, columns=SIDES)
          .round(4).to_string())
    print(f"\nSaved {ARTIFACTS / 'sides_vs_diffs.csv'} and outcome_model_sides.pkl")


if __name__ == "__main__":
    main()
