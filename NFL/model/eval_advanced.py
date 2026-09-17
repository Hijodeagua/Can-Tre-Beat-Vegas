"""
Out-of-time ablation of the NFL efficiency feature groups on top of the
betting-blind Elo.

Chronology, fixed:

    train        2002 .. s-1 for every test season s (walk-forward)
    selection    test seasons 2015-2023 — picks C and the compact set
    clean test   2024-2025 — reported once, never used to choose anything

The Elo parameters were tuned on 2005-2023 (`NFL/elo/tune.py`), so the
Elo-only baseline is slightly flattered on the selection window; every
candidate sits on the same Elo, so the comparison is fair, but only
2024-2025 is clean for all of them. 2026 is live and excluded.

Groups (the numbered comparison from the brief):

    1. elo            — elo_logit only (the shipping forecast, re-fit)
    2. + core EPA     — opponent-adjusted off/def EPA per play + net matchup
    3. + success      — 2 plus adjusted success rates
    4. + splits       — 2 plus dropback / rush / early-down matchups, PROE, explosive
    5. + drive        — 2 plus points & EPA per drive, drives, plays/yards per drive, series success
    6. + rz/third     — 2 plus red-zone TD rate and third-down matchups, RZ/3rd EPA, 3rd distance
    7. combined       — greedy forward selection over every group on 2015-2023

Plus two diagnostics: `elo + core_ewm` (the unadjusted EWMA twin of
group 2, to show what the opponent adjustment buys) and a LightGBM fit of
group 7's features (the repo already ships LightGBM for the NFL).

Target is home win; ties (actual 0.5) are dropped from the fit and the
score. Log loss and Brier decide; accuracy is printed because people ask.

Usage:
    python -m NFL.model.eval_advanced
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import evaluate
from NFL.elo.engine import replay
from NFL.model import advanced as adv

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
OUT = ARTIFACTS / "advanced_eval.json"
FIRST_TEST = 2015
CLEAN_FROM = 2024
LAST_TEST = 2025
C_GRID = [0.03, 0.1, 0.3, 1.0]

CORE = adv.ELO + adv.CORE_EPA
GROUPS = {
    "1_elo": adv.ELO,
    "2_core_epa": CORE,
    "3_success": CORE + adv.SUCCESS,
    "4_splits": CORE + adv.SPLITS,
    "5_drive": CORE + adv.DRIVE,
    "6_rz_third": CORE + adv.RED_ZONE_THIRD,
    "x_pace_st": CORE + adv.PACE_ST,
    "x_core_ewm": adv.ELO + adv.CORE_EWM,
}


def build_table() -> pd.DataFrame:
    _, history = replay()
    table = adv.build_game_table(history)
    table = table[(table["season"] >= 2002) & (table["home_win"] != 0.5)].copy()
    table["y"] = (table["home_win"] == 1.0).astype(int)
    return table.reset_index(drop=True)


def _lgbm_fit(X, y):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7,
                               min_child_samples=40, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=0)
    return model.fit(X, y)


def _wf(table, feats, seasons, C=1.0, fit=None):
    return evaluate.walk_forward(table, feats, "y", "season", seasons, C=C, fit=fit)


def run(table: pd.DataFrame) -> dict:
    select = list(range(FIRST_TEST, CLEAN_FROM))
    clean = list(range(CLEAN_FROM, LAST_TEST + 1))
    report = {"first_test": FIRST_TEST, "clean_from": CLEAN_FROM, "last_test": LAST_TEST,
              "n_rows": int(len(table)), "note_elo_tuning": "Elo params tuned on 2005-2023; "
              "2024-2025 is the clean window for the Elo baseline."}

    # coverage by season
    feats_all = adv.all_features()
    cov = table.groupby("season").apply(lambda g: pd.Series({
        "rows": len(g), "all_features": g[feats_all].notna().all(axis=1).mean(),
        "core_epa": g[adv.CORE_EPA].notna().all(axis=1).mean()})).reset_index()
    cov["all_features"] = cov["all_features"].round(3); cov["core_epa"] = cov["core_epa"].round(3)
    report["coverage"] = cov.to_dict("records")

    # --- selection window: C per group ------------------------------------
    chosen, grid = {}, []
    for name, feats in GROUPS.items():
        best = None
        for C in C_GRID:
            sc, _ = _wf(table, feats, select, C)
            grid.append({"group": name, "C": C, **{k: v for k, v in sc.row("").items() if k != "model"}})
            if best is None or sc.log_loss < best[1]:
                best = (C, sc.log_loss)
        chosen[name] = best[0]
    report["selection_grid"] = grid

    # --- greedy forward selection over feature groups on the selection window
    pool = {"core_epa": adv.CORE_EPA, "success": adv.SUCCESS, "splits": adv.SPLITS,
            "drive": adv.DRIVE, "rz_third": adv.RED_ZONE_THIRD, "pace_st": adv.PACE_ST,
            "core_ewm": adv.CORE_EWM}
    current, remaining, steps = [], list(pool), []
    current_ll = _wf(table, adv.ELO, select, chosen["1_elo"])[0].log_loss
    while remaining:
        trial = []
        for g in remaining:
            feats = adv.ELO + [f for h in current + [g] for f in pool[h]]
            trial.append((_wf(table, feats, select, 0.3)[0].log_loss, g))
        ll, g = min(trial)
        steps.append({"added": g, "selection_log_loss": round(ll, 5), "gain": round(current_ll - ll, 5)})
        if ll >= current_ll - 1e-4:
            break
        current.append(g); current_ll = ll; remaining.remove(g)
    combined = adv.ELO + [f for h in current for f in pool[h]]
    best_C, best_ll = None, None
    for C in C_GRID:
        ll = _wf(table, combined, select, C)[0].log_loss
        if best_ll is None or ll < best_ll:
            best_C, best_ll = C, ll
    report["combined"] = {"groups": current, "C": best_C, "selection": steps, "features": combined}

    # --- both windows, every group, paired against Elo-only ---------------
    results = {}
    for label, seasons in (("selection_2015_2023", select), ("clean_2024_2025", clean)):
        rows, per_season = [], {}
        base, base_seasons = _wf(table, adv.ELO, seasons, chosen["1_elo"])
        per_season["1_elo"] = base_seasons
        for name, feats in GROUPS.items():
            sc, ps = _wf(table, feats, seasons, chosen[name])
            rows.append({**sc.row(name), "C": chosen[name],
                         "paired_se_vs_elo": round(evaluate.paired_se(base, sc), 2)})
            per_season[name] = ps
        sc, ps = _wf(table, combined, seasons, best_C)
        rows.append({**sc.row("7_combined"), "C": best_C,
                     "paired_se_vs_elo": round(evaluate.paired_se(base, sc), 2)})
        per_season["7_combined"] = ps
        lg, ps = _wf(table, combined, seasons, fit=_lgbm_fit)
        rows.append({**lg.row("7_combined_lgbm"), "C": None,
                     "paired_se_vs_elo": round(evaluate.paired_se(base, lg), 2)})
        by_season = per_season["1_elo"][["season", "n", "log_loss"]].rename(columns={"log_loss": "elo"})
        by_season = by_season.merge(per_season["2_core_epa"][["season", "log_loss"]].rename(columns={"log_loss": "core_epa"}))
        by_season = by_season.merge(per_season["7_combined"][["season", "log_loss"]].rename(columns={"log_loss": "combined"}))
        results[label] = {"table": rows, "by_season": by_season.round(5).to_dict("records")}
    report["results"] = results

    # --- coefficient readout of the combined logistic on 2002-2023 -------
    train = table[table["season"] < CLEAN_FROM]
    model = evaluate.make_logistic(best_C).fit(train[combined], train["y"])
    coefs = dict(zip(combined, model.named_steps["clf"].coef_[0].round(4).tolist()))
    report["combined_coefficients_standardised"] = coefs
    return report


def print_report(r: dict) -> None:
    print(f"\nRows {r['n_rows']} · walk-forward test {r['first_test']}-{r['last_test']} · "
          f"clean window {r['clean_from']}-{r['last_test']}")
    print(r["note_elo_tuning"])
    print("\nCoverage by season:")
    print(pd.DataFrame(r["coverage"]).to_string(index=False))
    print("\nGreedy selection (2015-2023):")
    print(pd.DataFrame(r["combined"]["selection"]).to_string(index=False))
    print(f"  -> combined = elo + {r['combined']['groups']}  (C={r['combined']['C']})")
    for label, res in r["results"].items():
        print(f"\n{label}:")
        print(pd.DataFrame(res["table"]).to_string(index=False))
        print("\n  by season (log loss):")
        print(pd.DataFrame(res["by_season"]).to_string(index=False))
    print("\nStandardised coefficients of the combined logistic (fit 2002-2023):")
    for k, v in sorted(r["combined_coefficients_standardised"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:32s} {v:+.4f}")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    table = build_table()
    report = run(table)
    print_report(report)
    ARTIFACTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
