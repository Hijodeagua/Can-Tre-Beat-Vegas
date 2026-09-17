"""
Two questions about the club outcome model, answered on the same
chronological split as everything else:

1. Feature set — the shipping compact set (`BASE`) against the full set:
   home and away Elo as their own inputs (not only the gap), squad
   economics, xG and shots-on-target form, and every advanced Understat
   column (`advanced.ALL_ADVANCED`: xG/npxG form in both flavours, the
   home-attack-vs-away-defence splits, xG per shot, deep completions and
   the field-tilt proxy, PPDA, xPts form, rest, congestion, European
   ties).
2. Learner — multinomial logistic against a random forest and gradient
   boosting (`common/learners.py`), same features, same rows.

Fit on every season before TEST_FROM, score TEST_FROM onward, once. The
2023-24 season is reported too, labelled as the validation year it is.
Both scopes: every league, and the five Understat leagues alone.

The Elo-threshold question is answered directly: for the fitted boosting
model, P(home win) on a grid of (home Elo, gap) with every other input at
its training median, next to the empirical home-win rate binned the same
way. If a 100-point gap is worth more at 1700 than at 1300, both tables
show it.

Usage:
    python -m soccer.clubs.model.eval_learners
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import evaluate, learners
from soccer.clubs.model import advanced as adv
from soccer.clubs.model.eval_advanced import TOP5, build_table
from soccer.clubs.model.features import ALL_FEATURES
from soccer.clubs.model.shots import SHOT_FEATURES
from soccer.clubs.model.xg import XG_FEATURES

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
OUT = ARTIFACTS / "learners_eval.json"
VALIDATION = "2023-24"
TEST_FROM = "2024-25"
LOGISTIC_C = 0.1

RAW_ELO = ["elo_home_pre", "elo_away_pre"]
BASE = ["elo_gap"] + ALL_FEATURES + XG_FEATURES + SHOT_FEATURES
FULL = RAW_ELO + BASE + adv.ALL_ADVANCED
SETS = {"base": BASE, "full": FULL}


def _fit_score(table, feats, train_mask, test_mask, kind):
    model = learners.make(kind, C=LOGISTIC_C).fit(table.loc[train_mask, feats],
                                                  table.loc[train_mask, "outcome"])
    probs = model.predict_proba(table.loc[test_mask, feats])
    sc = evaluate.score_multiclass(table.loc[test_mask, "outcome"].to_numpy(), probs,
                                   list(model.classes_))
    return sc, model


def elo_threshold(table, model, feats, train_mask) -> dict:
    """Model-implied P(H) on a (home Elo, gap) grid, other inputs at their
    training medians; and the empirical home-win rate in the same bins."""
    med = table.loc[train_mask, feats].median(numeric_only=True)
    grid = []
    for home in (1250, 1400, 1550, 1700):
        for gap in (-100, 0, 100, 200):
            row = med.copy()
            # elo_gap includes the home advantage; back it out so the pair
            # is a real (home, away) rating pair at that venue-adjusted gap.
            adv_pts = float((table.loc[train_mask, "elo_gap"]
                             - (table.loc[train_mask, "elo_home_pre"]
                                - table.loc[train_mask, "elo_away_pre"])).median())
            row["elo_home_pre"] = home
            row["elo_away_pre"] = home + adv_pts - gap
            row["elo_gap"] = gap
            p = model.predict_proba(pd.DataFrame([row])[feats])[0]
            h = list(model.classes_).index("H")
            grid.append({"home_elo": home, "gap": gap, "away_elo": round(row["elo_away_pre"]),
                         "p_home": round(float(p[h]), 4)})
    t = table.copy()
    t["level"] = pd.cut(t["elo_home_pre"], [0, 1325, 1475, 1625, 9999],
                        labels=["<1325", "1325-1475", "1475-1625", ">1625"])
    t["gap_bin"] = pd.cut(t["elo_gap"], [-9999, -50, 50, 150, 9999],
                          labels=["<-50", "-50..50", "50..150", ">150"])
    emp = (t.groupby(["level", "gap_bin"], observed=True)
            .agg(n=("outcome", "size"), home_win=("outcome", lambda s: round(float((s == "H").mean()), 3)))
            .reset_index())
    emp["level"] = emp["level"].astype(str); emp["gap_bin"] = emp["gap_bin"].astype(str)
    return {"model_grid": grid, "empirical": emp.to_dict("records")}


def run(table: pd.DataFrame, scope: str) -> dict:
    if scope == "top5":
        table = table[table["league"].isin(TOP5)].reset_index(drop=True)
    season = table["season"]
    fit = season < TEST_FROM
    test = season >= TEST_FROM
    fit_v = season < VALIDATION
    valid = season == VALIDATION
    report = {"scope": scope, "n_fit": int(fit.sum()), "n_test": int(test.sum()),
              "n_valid": int(valid.sum()), "features": SETS}

    ship, _ = _fit_score(table, BASE, fit, test, "logistic")
    rows, models = [], {}
    for set_name, feats in SETS.items():
        for kind in learners.KINDS:
            sc, model = _fit_score(table, feats, fit, test, kind)
            models[(set_name, kind)] = model
            v, _ = _fit_score(table, feats, fit_v, valid, kind)
            rows.append({"features": set_name, "learner": kind,
                         **{k: val for k, val in sc.row("").items() if k != "model"},
                         "paired_se_vs_shipping": round(evaluate.paired_se(ship, sc), 2),
                         "validation_2023_24_log_loss": round(v.log_loss, 5)})
    report["test"] = rows
    # Per-season for the best full-set model vs the shipping model.
    best = min((r for r in rows if r["features"] == "full"), key=lambda r: r["log_loss"])
    report["best_full"] = {"learner": best["learner"], "log_loss": best["log_loss"]}
    per = []
    for s in sorted(table.loc[test, "season"].unique()):
        m = test & (season == s)
        b, _ = _fit_score(table, BASE, fit, m, "logistic")
        c, _ = _fit_score(table, FULL, fit, m, best["learner"])
        per.append({"season": s, "n": b.n, "shipping": round(b.log_loss, 5),
                    f"full_{best['learner']}": round(c.log_loss, 5),
                    "paired_se": round(evaluate.paired_se(b, c), 2)})
    report["by_season"] = per
    report["elo_threshold"] = elo_threshold(table, models[("full", "gbm")], FULL, fit)
    if ("full", "logistic") in models:
        clf = models[("full", "logistic")].named_steps["clf"]
        report["full_logistic_coefficients"] = {
            cls: dict(zip(FULL, coef.round(4).tolist()))
            for cls, coef in zip(clf.classes_, clf.coef_)}
    return report


def print_report(r: dict) -> None:
    print(f"\n[{r['scope']}] fit {r['n_fit']} · validation 2023-24 {r['n_valid']} · test ≥ {TEST_FROM} {r['n_test']}")
    print(pd.DataFrame(r["test"]).to_string(index=False))
    print(f"\nBest full-set model: {r['best_full']}")
    print(pd.DataFrame(r["by_season"]).to_string(index=False))
    print("\nElo threshold — boosting model, P(home) on a (home Elo, venue-adjusted gap) grid:")
    g = pd.DataFrame(r["elo_threshold"]["model_grid"]).pivot(index="home_elo", columns="gap", values="p_home")
    print(g.to_string())
    print("\nEmpirical home-win rate by home Elo level × gap bin (all seasons):")
    e = pd.DataFrame(r["elo_threshold"]["empirical"])
    print(e.pivot(index="level", columns="gap_bin", values="home_win").to_string())
    print(e.pivot(index="level", columns="gap_bin", values="n").to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["all", "top5", "both"], default="both")
    args = parser.parse_args()
    table = build_table()
    scopes = ["all", "top5"] if args.scope == "both" else [args.scope]
    out = {}
    for scope in scopes:
        print(f"\n==================== scope: {scope} ====================")
        out[scope] = run(table, scope)
        print_report(out[scope])
    ARTIFACTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
