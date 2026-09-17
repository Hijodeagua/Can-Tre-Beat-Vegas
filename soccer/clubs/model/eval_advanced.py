"""
Out-of-time ablation of the advanced feature groups for the club outcome
model, and the stale-xG audit.

Chronology, fixed:

    train        seasons  < VALIDATION   (2010-11 .. 2022-23)
    validation   season  == VALIDATION   (2023-24)  — picks C and the compact set
    test         seasons >= TEST_FROM    (2024-25 onward) — reported once

The Elo pools were tuned holding out 2024-25 onward, and the shipping
outcome model was trained on < 2024-25, so 2024-25+ is clean for both.
2023-24 has been used as a holdout by earlier feature tests in this repo
(`xg.py`, `shots.py`); it is used here only to choose regularisation and
the compact set, never to report a number that is called out-of-sample.

Groups (each is the shipping feature set plus one thing):

    1. base       — elo_gap + squad economics + xg_net_diff + sot_net_diff
    2. + xg       — refreshed rolling / EWMA xG and npxG for & against
    3. + matchup  — home-attack-vs-away-defence splits, xG per shot
    4. + territory— deep completions, deep share, PPDA, xPts form
    5. + rest     — rest days, 14-day congestion, European tie in the week
    6. combined   — the best compact set chosen on validation

Multinomial logistic on a standardised, median-imputed design matrix, all
fit on training rows only. Log loss and Brier are the metrics; the paired
test on per-match log loss is the bar.

The stale-xG audit answers one question: were the earlier xG results an
artefact of the feed dying on 2025-01-04? It reports per-season coverage,
the share of test rows the staleness guard zeroed, and the baseline
scored with and without `xg_net_diff` on (a) all test rows and (b) only
rows where both sides had live coverage.

Usage:
    python -m soccer.clubs.model.eval_advanced [--validation 2023-24] [--test-from 2024-25]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import evaluate
from soccer.clubs.model import advanced as adv
from soccer.clubs.model.europe import run_all_european
from soccer.clubs.model.features import attach_features
from soccer.clubs.model.shots import attach_shots
from soccer.clubs.model.train import FEATURES as BASE
from soccer.clubs.model.xg import attach_xg

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
CLASSES = ["A", "D", "H"]
VALIDATION = "2023-24"
TEST_FROM = "2024-25"
C_GRID = [0.1, 0.3, 1.0, 3.0]

GROUPS = {
    "base": [],
    "+xg_ewm": adv.XG_FEATURES,
    "+xg_rolling": adv.XG_ROLLING_FEATURES,
    "+matchup": adv.MATCHUP_FEATURES,
    "+territory": adv.TERRITORY_FEATURES,
    "+rest": adv.REST_FEATURES,
}


def build_table() -> pd.DataFrame:
    _, history = run_all_european()
    league_only = history[~history["league"].str.startswith("uefa:")].copy()
    table = attach_shots(attach_xg(attach_features(league_only)))
    return adv.attach_advanced(table)


def _fit_score(table: pd.DataFrame, features: list[str], train_mask, test_mask, C: float):
    model = evaluate.make_logistic(C).fit(table.loc[train_mask, features], table.loc[train_mask, "outcome"])
    probs = model.predict_proba(table.loc[test_mask, features])
    classes = list(model.classes_)
    return evaluate.score_multiclass(table.loc[test_mask, "outcome"].to_numpy(), probs, classes), model


TOP5 = ["epl", "la_liga", "bundesliga", "serie_a", "ligue_1"]


def run(table: pd.DataFrame, validation: str = VALIDATION, test_from: str = TEST_FROM,
        scope: str = "all") -> dict:
    """`scope="top5"` restricts every split to the five leagues the
    advanced feeds cover, so the imputer is not fit on rows where the
    features are structurally absent (second divisions, MLS)."""
    if scope == "top5":
        table = table[table["league"].isin(TOP5)].reset_index(drop=True)
    season = table["season"]
    train = season < validation
    valid = season == validation
    test = season >= test_from
    report: dict = {"scope": scope, "validation": validation, "test_from": test_from,
                    "n_train": int(train.sum()), "n_valid": int(valid.sum()), "n_test": int(test.sum())}

    # --- coverage: what each group could actually see, by season ----------
    cov = []
    for s, g in table[test | valid].groupby("season"):
        cov.append({
            "season": s, "rows": int(len(g)),
            "xg_net_diff_nonzero": round(float((g["xg_net_diff"] != 0).mean()), 3),
            "xg_ewm_covered": round(float(g[["xg_for_ewm_diff", "xg_against_ewm_diff"]].notna().all(axis=1).mean()), 3),
            "npxg_covered": round(float(g[["npxg_for_ewm_diff"]].notna().all(axis=1).mean()), 3),
            "territory_covered": round(float(g[adv.TERRITORY_FEATURES[:3]].notna().all(axis=1).mean()), 3),
            "rest_covered": round(float(g[["rest_diff"]].notna().all(axis=1).mean()), 3),
        })
    report["coverage"] = cov

    # --- validation: C per group, then the compact combined set -----------
    val_rows, chosen = [], {}
    for name, extra in GROUPS.items():
        feats = BASE + extra
        best = None
        for C in C_GRID:
            sc, _ = _fit_score(table, feats, train, valid, C)
            val_rows.append({"group": name, "C": C, **{k: v for k, v in sc.row("").items() if k != "model"}})
            if best is None or sc.log_loss < best[1]:
                best = (C, sc.log_loss)
        chosen[name] = best[0]
    report["validation_grid"] = val_rows

    # Greedy forward selection over groups on validation, starting from base.
    base_sc, _ = _fit_score(table, BASE, train, valid, chosen["base"])
    current, current_ll, remaining = [], base_sc.log_loss, [g for g in GROUPS if g != "base"]
    steps = []
    while remaining:
        trial = []
        for g in remaining:
            feats = BASE + [f for h in current + [g] for f in GROUPS[h]]
            sc, _ = _fit_score(table, feats, train, valid, 1.0)
            trial.append((sc.log_loss, g))
        ll, g = min(trial)
        steps.append({"added": g, "validation_log_loss": round(ll, 5),
                      "gain": round(current_ll - ll, 5)})
        if ll >= current_ll - 1e-4:
            break
        current.append(g); current_ll = ll; remaining.remove(g)
    combined_feats = BASE + [f for h in current for f in GROUPS[h]]
    best_C = 1.0
    for C in C_GRID:
        sc, _ = _fit_score(table, combined_feats, train, valid, C)
        if sc.log_loss < _fit_score(table, combined_feats, train, valid, best_C)[0].log_loss:
            best_C = C
    report["combined"] = {"groups": current, "C": best_C, "selection": steps}

    # --- test, once ---------------------------------------------------------
    # Retrain on train+validation for the test numbers (chronological: all
    # seasons before the test window).
    fit_mask = season < test_from
    rows, per_season = [], []
    base_test, _ = _fit_score(table, BASE, fit_mask, test, chosen["base"])
    for name, extra in GROUPS.items():
        sc, _ = _fit_score(table, BASE + extra, fit_mask, test, chosen[name])
        rows.append({**sc.row(name), "C": chosen[name],
                     "paired_se_vs_base": round(evaluate.paired_se(base_test, sc), 2)})
    sc, model = _fit_score(table, combined_feats, fit_mask, test, best_C)
    rows.append({**sc.row("combined"), "C": best_C,
                 "paired_se_vs_base": round(evaluate.paired_se(base_test, sc), 2)})
    for s in sorted(table.loc[test, "season"].unique()):
        m = test & (season == s)
        b, _ = _fit_score(table, BASE, fit_mask, m, chosen["base"])
        c, _ = _fit_score(table, combined_feats, fit_mask, m, best_C)
        per_season.append({"season": s, "n": b.n, "base_log_loss": round(b.log_loss, 5),
                           "combined_log_loss": round(c.log_loss, 5),
                           "paired_se": round(evaluate.paired_se(b, c), 2)})
    report["test"] = rows
    report["test_by_season"] = per_season
    report["combined_features"] = combined_feats

    # --- stale-xG audit -----------------------------------------------------
    no_xg = [f for f in BASE if f != "xg_net_diff"]
    # "Covered" = both sides had live xG form on the day (npxG is a
    # separate feed and may be absent without the xG audit being moot).
    covered = test & table[["xg_for_ewm_diff", "xg_against_ewm_diff"]].notna().all(axis=1)
    audit = {}
    for label, mask in (("all_test_rows", test), ("both_sides_covered", covered)):
        with_xg, _ = _fit_score(table, BASE, fit_mask, mask, chosen["base"])
        without, _ = _fit_score(table, no_xg, fit_mask, mask, chosen["base"])
        audit[label] = {"n": with_xg.n,
                        "with_xg_net_diff": round(with_xg.log_loss, 5),
                        "without": round(without.log_loss, 5),
                        "paired_se_for_xg": round(evaluate.paired_se(without, with_xg), 2)}
    guard_zeroed = table.loc[test & table["league"].isin(["epl", "la_liga", "bundesliga", "serie_a", "ligue_1"])]
    audit["test_rows_top5"] = int(len(guard_zeroed))
    audit["test_rows_top5_xg_net_diff_zero"] = int((guard_zeroed["xg_net_diff"] == 0).sum())
    report["stale_xg_audit"] = audit
    return report


def print_report(r: dict) -> None:
    print(f"\n[{r['scope']}] Train {r['n_train']} · validation {r['validation']} ({r['n_valid']}) · "
          f"test ≥ {r['test_from']} ({r['n_test']})\n")
    print("Coverage of each feature group on validation + test seasons:")
    print(pd.DataFrame(r["coverage"]).to_string(index=False))
    print("\nGreedy selection on validation:")
    print(pd.DataFrame(r["combined"]["selection"]).to_string(index=False))
    print(f"  -> combined = base + {r['combined']['groups']}  (C={r['combined']['C']})")
    print("\nTest (fit on every season before the test window):")
    print(pd.DataFrame(r["test"]).to_string(index=False))
    print("\nTest by season, base vs combined:")
    print(pd.DataFrame(r["test_by_season"]).to_string(index=False))
    a = r["stale_xg_audit"]
    print("\nStale-xG audit:")
    print(f"  top-5 test rows: {a['test_rows_top5']}, of which the staleness guard "
          f"zeroed xg_net_diff on {a['test_rows_top5_xg_net_diff_zero']}")
    for k in ("all_test_rows", "both_sides_covered"):
        v = a[k]
        print(f"  {k:20s} n={v['n']:5d}  with xg {v['with_xg_net_diff']}  "
              f"without {v['without']}  xg worth {v['paired_se_for_xg']:+.2f} SE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", default=VALIDATION)
    parser.add_argument("--test-from", default=TEST_FROM)
    parser.add_argument("--scope", choices=["all", "top5", "both"], default="both")
    args = parser.parse_args()
    table = build_table()
    ARTIFACTS.mkdir(exist_ok=True)
    scopes = ["all", "top5"] if args.scope == "both" else [args.scope]
    for scope in scopes:
        print(f"\n==================== scope: {scope} ====================")
        report = run(table, args.validation, args.test_from, scope=scope)
        print_report(report)
        out = ARTIFACTS / f"advanced_eval_{scope}.json"
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
