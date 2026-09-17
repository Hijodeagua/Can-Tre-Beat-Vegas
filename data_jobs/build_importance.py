#!/usr/bin/env python3
"""
What each live model's inputs are actually worth, measured rather than
asserted, for every model on the board:

    web/public/data/importance.json

and a per-model copy next to each model's own tuned parameters, so the
number sits with the artifact it describes.

Two methods, because the models are two different kinds of thing and
pretending otherwise would produce a chart that lies:

* **Permutation importance** — the club-soccer outcome model is a fitted
  multinomial logistic, so a feature can be shuffled and the damage
  measured. Each feature is permuted `REPEATS` times on the holdout
  seasons the fit never saw; the reported number is the mean increase in
  log loss, with its spread across repeats. This is importance in the
  usual sense.

* **Component ablation** — an Elo engine has no feature matrix to
  shuffle. Permuting "home advantage" is meaningless: it is a constant,
  not a column. What can be measured is what the component is *worth* —
  neutralise it (home edge to zero, no bye bonus, no season regression,
  margin ignored) and replay the whole history. The reported number is
  the increase in log loss, on the same held-out seasons each tuner keeps
  back. A component whose ablation costs nothing is not carrying its
  weight, which is exactly what a reader should be able to see.

Both are reported as an increase in log loss, so the bars are comparable
in direction and units even though they answer slightly different
questions — and `method` on each model says which question was asked.

The two are not interchangeable and the chart labels them. A parameter's
ablation folds in everything downstream of it (drop the home edge and
every subsequent rating is different), which is why an Elo ablation can
dwarf a permutation number; it is a measure of the component's
contribution to the whole replay, not of one column in a design matrix.

Six models share the file: `soccer`, `nfl`, `cfb`, `mlb` measure each
sport's Elo (or the fitted outcome model, for soccer); `nfl_second_stage`
and `cfb_second_stage` measure the Elo + adjusted-efficiency logistic
layered on top (`NFL/model/advanced.py`, `CFB/model/advanced.py`) the
same way `soccer()` measures its outcome model — permutation importance
on the clean, never-trained-on seasons the second stage's own evaluation
holds out. A sport with no second stage (soccer's advanced layer isn't
shipped; MLB has none) has no `*_second_stage` key, and the page section
just doesn't grow a second panel.

Usage:
    python -m data_jobs.build_importance [--skip soccer]
    python -m data_jobs.build_importance --only nfl nfl_second_stage

`--only` / `--skip` merge into whatever the web file already holds, so a
single model can be recomputed without redoing the soccer replay. Merging
means read-modify-write, so run one invocation at a time: two in parallel
will each write the models the other is missing.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_OUT = REPO_ROOT / "web" / "public" / "data" / "importance.json"
REPEATS = 10
SEED = 0


def _round(x) -> float:
    return round(float(x), 5)


# --------------------------------------------------------------------------
# club soccer — permutation importance on the fitted outcome model
# --------------------------------------------------------------------------
def soccer() -> dict:
    import pickle

    import pandas as pd
    from sklearn.metrics import log_loss

    from soccer.clubs.model.train import ARTIFACTS, build_table

    artifact = pickle.loads((ARTIFACTS / "outcome_model.pkl").read_bytes())
    model, features, split = artifact["model"], artifact["features"], artifact["split_season"]

    table = build_table()
    test = table[table["season"] >= split]
    # Kept as a frame with the fitted feature names, so sklearn scores it
    # the way the pipeline does rather than warning about bare arrays.
    X = test[features].astype(float)
    y = test["outcome"].to_numpy()
    labels = list(model.classes_)

    def loss(frame) -> float:
        return log_loss(y, model.predict_proba(frame), labels=labels)

    baseline = loss(X)
    rng = np.random.default_rng(SEED)
    rows = []
    for name in features:
        deltas = []
        constant = X[name].nunique(dropna=False) <= 1
        for _ in range(REPEATS):
            shuffled = X.copy()
            shuffled[name] = rng.permutation(shuffled[name].to_numpy())
            deltas.append(loss(shuffled) - baseline)
        rows.append({
            "name": name,
            "value": _round(np.mean(deltas)),
            "sd": _round(np.std(deltas)),
            "detail": (
                f"no data over this window — the column is constant, so "
                f"shuffling it cannot do damage"
                if constant else f"column shuffled {REPEATS}x on {split}+ holdout"
            ),
            "constant": bool(constant),
        })
    rows.sort(key=lambda r: -r["value"])
    return {
        "name": "Club soccer outcome model",
        "method": "permutation importance",
        "metric": "increase in holdout log loss when the feature is shuffled",
        "baseline": _round(baseline),
        "n": int(len(test)),
        "window": f"{split} onward (never fitted on)",
        "features": rows,
        "caveat": (
            "A feature already feeding zeros on every live slate cannot be "
            "damaged by shuffling, so a zero here can mean 'no signal' or "
            "'no data' — see the dormant features in docs/MODEL_FEATURES.md."
        ),
    }


# --------------------------------------------------------------------------
# the Elo engines — component ablation
# --------------------------------------------------------------------------
def _ablate(score, tuned: dict, off: dict[str, tuple[str, float, str]]) -> list[dict]:
    """`score(params) -> log loss`, run once per neutralised component.

    `off` maps a parameter to (label, neutral value, what that means).
    """
    baseline = score(tuned)
    rows = []
    for param, (label, neutral, detail) in off.items():
        if param not in tuned:
            continue
        trial = dict(tuned, **{param: neutral})
        rows.append({
            "name": label,
            "value": _round(score(trial) - baseline),
            "sd": None,
            "detail": detail,
        })
    rows.sort(key=lambda r: -r["value"])
    return baseline, rows


def nfl() -> dict:
    from NFL.elo.engine import fast_replay, load_games
    from NFL.elo.tune import HOLDOUT_FROM, replay_rows

    params = json.loads((REPO_ROOT / "NFL/elo/artifacts/tuned_params.json").read_text())
    tuned = params["params"]
    rows_in = replay_rows(load_games())

    def score(p) -> float:
        return fast_replay(rows_in, p, HOLDOUT_FROM, 9999)[0]

    baseline, rows = _ablate(score, tuned, {
        "home_advantage": ("Home advantage", 0.0, "home edge set to 0 Elo"),
        "rest_bonus": ("Rest (bye week)", 0.0, "no bonus off a bye"),
        "season_regression": ("Season regression", 0.0, "ratings carry over untouched"),
        "playoff_k_mult": ("Postseason K weight", 1.0, "playoff games update like any other"),
        "margin_cap": ("Margin of victory", 1.0, "every win counts the same, whatever the margin"),
    })
    _, n = fast_replay(rows_in, tuned, HOLDOUT_FROM, 9999)
    return {
        "name": "NFL Elo",
        "method": "component ablation",
        "metric": "increase in holdout log loss when the component is neutralised",
        "baseline": _round(baseline),
        "n": int(n),
        "window": f"{HOLDOUT_FROM} onward (never tuned on)",
        "features": rows,
        "caveat": (
            "Neutralising a component changes every rating after it, so these "
            "are contributions to the whole replay, not to one game. "
            "'Margin of victory' clamps the margin to a point rather than "
            "removing the multiplier, which also shrinks the update size — "
            "read it as 'margin ignored', not as a clean K-preserving swap."
        ),
    }


def cfb() -> dict:
    from CFB.data.teams import FBS  # noqa: F401  (imported by replay_rows)
    from CFB.model.elo import fast_replay, load_games, season_conferences
    from CFB.model.tune import HOLDOUT_FROM, replay_rows

    params = json.loads((REPO_ROOT / "CFB/model/artifacts/tuned_params.json").read_text())
    tuned = params["params"]
    games = load_games()
    rows_in = replay_rows(games)
    confs = season_conferences(games)

    def score(p) -> float:
        return fast_replay(rows_in, confs, p, HOLDOUT_FROM, 9999)[0]

    baseline, rows = _ablate(score, tuned, {
        "home_advantage": ("Home advantage", 0.0, "home edge set to 0 Elo"),
        "conf_weight": ("Conference regression", 0.0, "regress to 1500 alone, ignoring the new conference"),
        "season_regression": ("Season regression", 0.0, "ratings carry over untouched"),
        "margin_cap": ("Margin of victory", 1.0, "every win counts the same, whatever the margin"),
        "fcs_rating": ("FCS opponent rating", 1500.0, "non-FBS opponents rated as average FBS"),
        "entry_rating": ("FBS entry rating", 1500.0, "a new FBS program starts at average"),
    })
    _, n = fast_replay(rows_in, confs, tuned, HOLDOUT_FROM, 9999)
    return {
        "name": "College football Elo",
        "method": "component ablation",
        "metric": "increase in holdout log loss when the component is neutralised",
        "baseline": _round(baseline),
        "n": int(n),
        "window": f"{HOLDOUT_FROM} onward (never tuned on)",
        "features": rows,
        "caveat": (
            "Neutralising a component changes every rating after it, so these "
            "are contributions to the whole replay, not to one game. "
            "'Margin of victory' clamps the margin to a point rather than "
            "removing the multiplier, which also shrinks the update size."
        ),
    }


# --------------------------------------------------------------------------
# NFL / CFB second stage (Elo + adjusted efficiency) — permutation importance
# --------------------------------------------------------------------------
def _second_stage_permutation(*, name: str, features: list[str], baseline_model,
                              X, y, window: str, caveat: str) -> dict:
    """Shared body for the two second-stage permutation builders below:
    same recipe as `soccer()` — shuffle each feature `REPEATS` times on
    held-out rows the fit never trained on, report the mean increase in
    log loss."""
    p = np.clip(baseline_model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    y_arr = y.to_numpy()

    def loss(frame) -> float:
        pf = np.clip(baseline_model.predict_proba(frame)[:, 1], 1e-6, 1 - 1e-6)
        return float(-(y_arr * np.log(pf) + (1 - y_arr) * np.log(1 - pf)).mean())

    baseline = loss(X)
    rng = np.random.default_rng(SEED)
    rows = []
    for col in features:
        deltas = []
        constant = X[col].nunique(dropna=False) <= 1
        for _ in range(REPEATS):
            shuffled = X.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            deltas.append(loss(shuffled) - baseline)
        rows.append({
            "name": col,
            "value": _round(np.mean(deltas)),
            "sd": _round(np.std(deltas)),
            "detail": (
                f"no data over this window — the column is constant, so "
                f"shuffling it cannot do damage"
                if constant else f"column shuffled {REPEATS}x on the held-out window"
            ),
            "constant": bool(constant),
        })
    rows.sort(key=lambda r: -r["value"])
    return {
        "name": name,
        "method": "permutation importance",
        "metric": "increase in holdout log loss when the feature is shuffled",
        "baseline": _round(baseline),
        "n": int(len(X)),
        "window": window,
        "features": rows,
        "caveat": caveat,
    }


def nfl_second_stage() -> dict:
    from common import evaluate
    from NFL.model import advanced as adv
    from NFL.model.eval_advanced import CLEAN_FROM, LAST_TEST, build_table

    table = build_table()
    features = adv.PRODUCTION_FEATURES
    train = table["season"] < CLEAN_FROM
    test = (table["season"] >= CLEAN_FROM) & (table["season"] <= LAST_TEST)
    model = evaluate.make_logistic(adv.PRODUCTION_C).fit(
        table.loc[train, features], table.loc[train, "y"])
    X = table.loc[test, features].astype(float)
    return _second_stage_permutation(
        name="NFL second stage (Elo + adjusted success)",
        features=features, baseline_model=model, X=X, y=table.loc[test, "y"],
        window=f"{CLEAN_FROM}-{LAST_TEST} (clean of the Elo's own 2005-2023 tuning window)",
        caveat=(
            "This is the logistic layered on top of Elo "
            "(NFL/model/advanced.py), not the Elo engine itself — see the "
            "'NFL Elo' section above for what the rating's own components "
            "are worth. A game with no play-by-play rating for either side "
            "falls back to Elo alone and is not in this window."
        ),
    )


def cfb_second_stage() -> dict:
    from common import evaluate
    from CFB.model import advanced as adv
    from CFB.model.eval_advanced import TEST, TRAIN_FROM, build_table

    table = build_table()
    features = adv.PRODUCTION_FEATURES
    train = (table["season"] >= TRAIN_FROM) & (table["season"] < TEST[0])
    test = (table["season"] >= TEST[0]) & (table["season"] <= TEST[1])
    model = evaluate.make_logistic(adv.PRODUCTION_C).fit(
        table.loc[train, features], table.loc[train, "y"])
    X = table.loc[test, features].astype(float)
    return _second_stage_permutation(
        name="College football second stage (Elo + adjusted EPA)",
        features=features, baseline_model=model, X=X, y=table.loc[test, "y"],
        window=f"{TEST[0]}-{TEST[1]} (clean of the Elo's own 2005-2023 tuning window)",
        caveat=(
            "This is the logistic layered on top of Elo "
            "(CFB/model/advanced.py), not the Elo engine itself — see the "
            "'College football Elo' section above for what the rating's "
            "own components are worth. A game with an FCS side, or no "
            "weekly-summary rating for either side, falls back to Elo "
            "alone and is not in this window."
        ),
    )


def mlb() -> dict:
    from mlb.elo import CARRYOVER, HOME_ADVANTAGE, K, run_history

    EVAL_FROM = 2012

    def score(home_advantage: float, carryover: float, use_mov: bool) -> tuple[float, int]:
        _, hist, _ = run_history(k=K, home_advantage=home_advantage,
                                 carryover=carryover, use_mov=use_mov)
        ev = hist[hist.season >= EVAL_FROM]
        p = ev.p_home.clip(1e-9, 1 - 1e-9)
        y = ev.home_win
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        return float(ll), int(len(ev))

    baseline, n = score(HOME_ADVANTAGE, CARRYOVER, True)
    rows = [
        {"name": "Home advantage", "sd": None, "detail": "home edge set to 0 Elo",
         "value": _round(score(0.0, CARRYOVER, True)[0] - baseline)},
        {"name": "Season carryover", "sd": None, "detail": "no rating carried across seasons",
         "value": _round(score(HOME_ADVANTAGE, 0.0, True)[0] - baseline)},
        {"name": "Margin of victory", "sd": None, "detail": "run margin ignored in the update",
         "value": _round(score(HOME_ADVANTAGE, CARRYOVER, False)[0] - baseline)},
    ]
    rows.sort(key=lambda r: -r["value"])
    return {
        "name": "MLB Elo",
        "method": "component ablation",
        "metric": "increase in log loss when the component is neutralised",
        "baseline": _round(baseline),
        "n": n,
        "window": f"{EVAL_FROM} onward",
        "features": rows,
        "caveat": (
            "Only the rating engine's own components are ablated here. The "
            "starting-pitcher, rest and travel adjustments are applied by the "
            "daily pipeline rather than the replay, so their contribution is "
            "not measured yet — see research/SP-BACKTEST.md for the pitcher "
            "layer's own out-of-sample test."
        ),
    }


BUILDERS = {"soccer": soccer, "nfl": nfl, "nfl_second_stage": nfl_second_stage,
            "cfb": cfb, "cfb_second_stage": cfb_second_stage, "mlb": mlb}
# Where each model's own copy lands, next to its tuned parameters.
ARTIFACTS = {
    "soccer": REPO_ROOT / "soccer/clubs/model/artifacts/importance.json",
    "nfl": REPO_ROOT / "NFL/elo/artifacts/importance.json",
    "nfl_second_stage": REPO_ROOT / "NFL/model/artifacts/importance.json",
    "cfb": REPO_ROOT / "CFB/model/artifacts/importance.json",
    "cfb_second_stage": REPO_ROOT / "CFB/model/artifacts/importance_second_stage.json",
    "mlb": REPO_ROOT / "data/mlb/importance.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", default=[],
                        help="model keys to leave out (e.g. soccer, which replays 58k matches)")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    keys = args.only or [k for k in BUILDERS if k not in args.skip]
    models = {}
    if WEB_OUT.exists():
        # Keep whatever this run isn't recomputing, so --only stays useful.
        models = json.loads(WEB_OUT.read_text()).get("models", {})
    for key in keys:
        print(f"== {key}")
        entry = BUILDERS[key]()
        models[key] = entry
        path = ARTIFACTS[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry, indent=2) + "\n")
        top = entry["features"][0]
        print(f"   baseline log loss {entry['baseline']} over n={entry['n']}; "
              f"biggest: {top['name']} +{top['value']}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": models,
    }
    WEB_OUT.parent.mkdir(parents=True, exist_ok=True)
    WEB_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {WEB_OUT}")


if __name__ == "__main__":
    main()
