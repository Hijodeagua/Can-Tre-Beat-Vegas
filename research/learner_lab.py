"""
Two questions about the shipped second-stage learners that the sandbox
this repo is developed in cannot answer well, because both want more CPU
than it has. Run them on a real machine.

    python -m research.learner_lab tune      [--sport soccer|nfl|cfb|all]
    python -m research.learner_lab calibrate [--sport ...] [--seeds 8]

**tune** — every learner in `common/learners.py` currently runs on one
set of hyperparameters, shared across all three sports, chosen once and
never searched. That makes one published conclusion weaker than it
looks: "boosting overfits" is really "boosting at those defaults, on
this many rows, overfits". This grid-searches each family on a
validation window that ends before the test window, reports the best of
each, and scores only the winners on the test window, once. Nothing here
selects on test data.

**calibrate** — a forest's probabilities are leaf frequencies, which are
usually less calibrated than a logistic's, and these probabilities are
not only scored: they are the published pick confidence and they drive
the season simulations. This reports a reliability table (predicted vs
actual, by probability bucket), and whether wrapping the forest in
isotonic or sigmoid calibration — fit on the validation window, never on
test — improves the test log loss. It also refits the shipped model
across several seeds and reports the spread, which is the honest scale
against which to read a 0.002 gap between two learners.

Both subcommands cache each sport's feature table under
`research/.cache/` (gitignored), because building the soccer one replays
~59k matches. Delete the cache after changing a feature builder.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import time
from pathlib import Path

# Set before numpy or scikit-learn load: OpenMP reads its thread policy
# once, when libgomp initialises, and that happens on the first import of
# a compiled extension.
#
# The default policy is an active spin-wait, which is right when a
# process owns the machine and catastrophic when two do. Two tune runs
# side by side on four cores turned a 2-second boosting fit into minutes
# — not slower arithmetic, just threads burning cores waiting for each
# other. A passive wait costs a little when uncontended and removes the
# cliff entirely. `_single_run()` below stops the overlap happening in
# the first place; this is the belt to that pair of braces, because
# nothing stops a second run from another shell or another checkout.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from common import evaluate, learners

CACHE = Path(__file__).resolve().parent / ".cache"
OUT = Path(__file__).resolve().parent / "artifacts"


# --------------------------------------------------------------------------
# the three sports, each as (table, features, target, windows)
# --------------------------------------------------------------------------
class Sport:
    """One sport's table and its three chronological windows.

    `fit` is everything before validation; `valid` picks hyperparameters;
    `fit_for_test` is everything before the test window; `test` is scored
    once. Validation always ends before the test window starts, so a
    grid searched on `valid` has not seen a test row.
    """

    def __init__(self, key, table, features, target, classes,
                 fit, valid, fit_for_test, test):
        self.key, self.table, self.features = key, table, features
        self.target, self.classes = target, classes
        self.fit, self.valid = fit, valid
        self.fit_for_test, self.test = fit_for_test, test

    def score(self, model, test_mask) -> evaluate.Scores:
        X = self.table.loc[test_mask, self.features]
        y = self.table.loc[test_mask, self.target].to_numpy()
        probs = model.predict_proba(X)
        if self.classes is None:
            return evaluate.score_binary(y, probs[:, 1])
        order = [list(model.classes_).index(c) for c in self.classes]
        return evaluate.score_multiclass(y, probs[:, order], self.classes)

    def fit_model(self, kind, train_mask, **overrides):
        model = learners.make(kind, **overrides)
        return model.fit(self.table.loc[train_mask, self.features],
                         self.table.loc[train_mask, self.target])


def _cached(key: str, build) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    t0 = time.time()
    table = build()
    print(f"   built {key} table in {time.time() - t0:.0f}s ({len(table)} rows)")
    table.to_parquet(path)
    return table


def load_soccer() -> Sport:
    from soccer.clubs.model import train
    table = _cached("soccer", train.build_table)
    s = table["season"]
    return Sport("soccer", table, list(train.FEATURES), "outcome", ["A", "D", "H"],
                 fit=s < "2023-24", valid=s == "2023-24",
                 fit_for_test=s < "2024-25", test=s >= "2024-25")


def load_nfl() -> Sport:
    from NFL.model import advanced as adv
    from NFL.model.eval_advanced import CLEAN_FROM, LAST_TEST, build_table
    table = _cached("nfl", build_table)
    s = table["season"]
    # The shipped evaluation walks forward over 2015-2023 for selection;
    # a grid cannot afford that, so the validation window here is the
    # last four seasons before the clean one, fit on everything earlier.
    return Sport("nfl", table, list(adv.PRODUCTION_FEATURES), "y", None,
                 fit=s < 2020, valid=(s >= 2020) & (s < CLEAN_FROM),
                 fit_for_test=s < CLEAN_FROM,
                 test=(s >= CLEAN_FROM) & (s <= LAST_TEST))


def load_cfb() -> Sport:
    from CFB.model import advanced as adv
    from CFB.model.eval_advanced import TEST, TRAIN_FROM, VALIDATION, build_table
    table = _cached("cfb", build_table)
    s = table["season"]
    return Sport("cfb", table, list(adv.PRODUCTION_FEATURES), "y", None,
                 fit=(s >= TRAIN_FROM) & (s < VALIDATION), valid=s == VALIDATION,
                 fit_for_test=(s >= TRAIN_FROM) & (s < TEST[0]),
                 test=(s >= TEST[0]) & (s <= TEST[1]))


LOADERS = {"soccer": load_soccer, "nfl": load_nfl, "cfb": load_cfb}

# Grids are deliberately small: the point is whether the defaults are
# roughly right, not to squeeze the last 0.0001 out of a leaderboard.
GRIDS = {
    "logistic": [{"C": c} for c in (0.03, 0.1, 0.3, 1.0, 3.0)],
    "random_forest": [
        {"min_samples_leaf": leaf, "max_features": mf, "n_estimators": 500}
        for leaf, mf in itertools.product((5, 10, 25, 50, 100), ("sqrt", 0.3, 0.6))
    ],
    "gbm": [
        {"learning_rate": lr, "max_leaf_nodes": leaves,
         "min_samples_leaf": leaf, "l2_regularization": l2, "max_iter": 400}
        for lr, leaves, leaf, l2 in itertools.product(
            (0.01, 0.03, 0.1), (3, 7, 15, 31), (20, 40, 100), (0.0, 1.0, 10.0))
    ],
}


# --------------------------------------------------------------------------
# tune
# --------------------------------------------------------------------------
def tune(sport: Sport) -> dict:
    print(f"\n=== {sport.key}: {len(sport.features)} features, "
          f"fit {int(sport.fit.sum())} / valid {int(sport.valid.sum())} / "
          f"test {int(sport.test.sum())}")
    report = {"sport": sport.key, "n_features": len(sport.features),
              "n_fit": int(sport.fit.sum()), "n_valid": int(sport.valid.sum()),
              "n_test": int(sport.test.sum()), "grids": {}, "test": []}

    winners = {}
    for kind, grid in GRIDS.items():
        rows, t0 = [], time.time()
        for params in grid:
            model = sport.fit_model(kind, sport.fit, **params)
            sc = sport.score(model, sport.valid)
            rows.append({**params, "valid_log_loss": round(sc.log_loss, 5),
                         "valid_brier": round(sc.brier, 5)})
        rows.sort(key=lambda r: r["valid_log_loss"])
        winners[kind] = {k: v for k, v in rows[0].items()
                         if k not in ("valid_log_loss", "valid_brier")}
        report["grids"][kind] = rows
        print(f"  {kind}: {len(grid)} fits in {time.time() - t0:.0f}s — "
              f"best {rows[0]['valid_log_loss']} at {winners[kind]}")
        print(pd.DataFrame(rows[:5]).to_string(index=False))

    # Test, once: the shipped configuration against each family's winner.
    # "Shipped" means what the sport actually runs, including any tuned
    # hyperparameters it has adopted — otherwise a sport that already
    # took a tuning win would be compared against a model it no longer
    # uses, and the grid would look like it found the same gain twice.
    kind, params = shipped_config(sport.key)
    shipped = sport.fit_model(kind, sport.fit_for_test, **params)
    base = sport.score(shipped, sport.test)
    label = f"shipped ({kind}, {params or 'default params'})"
    report["test"].append({**base.row(label), "paired_se_vs_shipped": 0.0})
    for kind, params in winners.items():
        model = sport.fit_model(kind, sport.fit_for_test, **params)
        sc = sport.score(model, sport.test)
        report["test"].append({**sc.row(f"tuned {kind}"), "params": params,
                               "paired_se_vs_shipped": round(evaluate.paired_se(base, sc), 2)})
    print("\n  test window, once:")
    print(pd.DataFrame(report["test"]).to_string(index=False))
    return report


def shipped_config(sport_key: str) -> tuple[str, dict]:
    """What the sport ships today — learner and any hyperparameters it
    overrides — so the tuned numbers have the right baseline."""
    if sport_key == "soccer":
        from soccer.clubs.model import train
        return train.LEARNER, dict(getattr(train, "PRODUCTION_PARAMS", {}))
    if sport_key == "nfl":
        from NFL.model import advanced as nfl
        return nfl.PRODUCTION_LEARNER, dict(getattr(nfl, "PRODUCTION_PARAMS", {}))
    from CFB.model import advanced as cfb
    return cfb.PRODUCTION_LEARNER, dict(getattr(cfb, "PRODUCTION_PARAMS", {}))


# --------------------------------------------------------------------------
# calibrate
# --------------------------------------------------------------------------
def reliability(sport: Sport, model, mask, bins=10) -> list[dict]:
    """Predicted vs actual by probability bucket. For the multiclass
    sport this is the home-win probability against the home-win rate."""
    X = sport.table.loc[mask, sport.features]
    y = sport.table.loc[mask, sport.target].to_numpy()
    probs = model.predict_proba(X)
    if sport.classes is None:
        p, hit = probs[:, 1], y.astype(float)
    else:
        h = list(model.classes_).index("H")
        p, hit = probs[:, h], (y == "H").astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        out.append({"bucket": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                    "mean_predicted": round(float(p[m].mean()), 4),
                    "actual": round(float(hit[m].mean()), 4),
                    "gap": round(float(p[m].mean() - hit[m].mean()), 4)})
    return out


def calibrate(sport: Sport, seeds: int) -> dict:
    kind = learners_default(sport.key)
    print(f"\n=== {sport.key}: calibration and seed stability for the shipped {kind}")
    report = {"sport": sport.key, "learner": kind}

    # 1. Seed stability of the shipped model on the test window. A gap
    #    between two learners smaller than this spread is not a finding.
    losses = []
    for seed in range(seeds):
        model = sport.fit_model(kind, sport.fit_for_test, random_state=seed) \
            if kind != "logistic" else sport.fit_model(kind, sport.fit_for_test)
        losses.append(sport.score(model, sport.test).log_loss)
        if kind == "logistic":
            break                      # deterministic; one fit is the answer
    report["seed_stability"] = {
        "seeds": len(losses), "mean": round(float(np.mean(losses)), 5),
        "sd": round(float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0, 5),
        "min": round(float(np.min(losses)), 5), "max": round(float(np.max(losses)), 5),
    }
    print(f"  seed spread over {len(losses)} fits: "
          f"{report['seed_stability']['min']} .. {report['seed_stability']['max']} "
          f"(sd {report['seed_stability']['sd']})")

    # 2. Reliability of the shipped model, and of a calibrated wrapper
    #    fit on the validation window only.
    shipped = sport.fit_model(kind, sport.fit_for_test)
    base = sport.score(shipped, sport.test)
    report["reliability_shipped"] = reliability(sport, shipped, sport.test)
    print("\n  reliability, shipped model on the test window:")
    print(pd.DataFrame(report["reliability_shipped"]).to_string(index=False))

    # Calibration has to be judged against the *same* inner model, not
    # against the shipped one: a calibrator needs a held-out slice, so
    # its inner model trains on the pre-validation window and sees one
    # season less. Comparing it to the shipped model would charge
    # calibration for that missing season. `inner, uncalibrated` is the
    # honest baseline; `shipped` is in the table only for scale.
    inner = sport.fit_model(kind, sport.fit)          # never sees validation
    inner_sc = sport.score(inner, sport.test)
    rows = [{**base.row("shipped (fit through validation)"), "paired_se_vs_inner": None},
            {**inner_sc.row("inner, uncalibrated (fit before validation)"),
             "paired_se_vs_inner": 0.0}]
    for method in ("isotonic", "sigmoid"):
        # FrozenEstimator: calibrate the already-fitted inner model
        # rather than refitting it inside the calibrator (sklearn 1.9
        # replaced cv="prefit" with this).
        cal = CalibratedClassifierCV(FrozenEstimator(inner), method=method)
        cal.fit(sport.table.loc[sport.valid, sport.features],
                sport.table.loc[sport.valid, sport.target])
        sc = sport.score(cal, sport.test)
        rows.append({**sc.row(f"{method} (calibrated on validation)"),
                     "paired_se_vs_inner": round(evaluate.paired_se(inner_sc, sc), 2)})
        report[f"reliability_{method}"] = reliability(sport, cal, sport.test)
    report["calibration"] = rows
    print("\n  calibration on the test window (compare the calibrated rows to "
          "`inner, uncalibrated`, which trains on the same window):")
    print(pd.DataFrame(rows).to_string(index=False))
    return report


# --------------------------------------------------------------------------
@contextlib.contextmanager
def _single_run():
    """Refuse to start while another run holds the lock.

    Two reasons. The artifact write below is read-modify-write, so two
    runs finishing together lose one of them. And two runs fighting over
    the same cores is far worse than a queue: the grids are almost all
    compiled, multi-threaded code, so overlapping them does not halve the
    speed, it collapses it.

    A stale lock (a run that was killed) is taken over rather than
    honoured, so a crash never wedges the tool.
    """
    lock = OUT / ".run.lock"
    OUT.mkdir(exist_ok=True)
    if lock.exists():
        pid = lock.read_text().strip()
        alive = pid.isdigit() and Path(f"/proc/{pid}").exists()
        if alive:
            raise SystemExit(
                f"another learner_lab run is going (pid {pid}). Wait for it, or "
                f"stop it and delete {lock}. Running two at once loses one of "
                f"their artifacts and makes both far slower than running them "
                f"back to back.")
        print(f"taking over a stale lock from pid {pid or '?'}")
    lock.write_text(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["tune", "calibrate"])
    ap.add_argument("--sport", choices=list(LOADERS) + ["all"], default="all")
    ap.add_argument("--seeds", type=int, default=8,
                    help="fits per sport for the seed-stability spread (calibrate)")
    args = ap.parse_args()

    keys = list(LOADERS) if args.sport == "all" else [args.sport]
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{args.command}.json"
    # Merge into whatever is already there, so `--sport nfl` does not
    # erase yesterday's soccer run. Read-modify-write, so run one
    # invocation at a time.
    with _single_run():
        out = json.loads(path.read_text()) if path.exists() else {}
        for key in keys:
            sport = LOADERS[key]()
            out[key] = tune(sport) if args.command == "tune" else calibrate(sport, args.seeds)
        path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
