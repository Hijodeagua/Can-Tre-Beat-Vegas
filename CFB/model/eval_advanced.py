"""
Out-of-time ablation of the SportsDataverse feature groups on top of the
college Elo.

Chronology, fixed:

    train        2005-2022  (2004 is the first weekly season: no prior, burn-in)
    validation   2023       picks C and the compact set
    test         2024-2025  reported once
    live         2026       excluded

Contamination, stated: the Elo parameters were tuned on 2005-2023
(`CFB/model/tune.py`, holdout 2024+), so the Elo-only baseline is
in-sample on the training and validation seasons. Every candidate sits
on the same Elo, so the *relative* comparison on 2023 is fair; only the
2024-2025 test is clean for the baseline itself.

Groups:

    1. elo          — elo_logit only
    2. + core       — adjusted offence / defence EPA, matchup net, net diff
    3. + success    — 2 plus success rates
    4. + early/expl — 2 plus early-down EPA, explosive, havoc
    5. + drive      — 2 plus EPA per drive, drives per game, plays and yards per drive
    6. + situational— 2 plus red zone, third down, pass rate
    x. + splits     — 2 plus pass / rush EPA matchups
    7. combined     — greedy forward selection on validation

Reported overall and for FBS-vs-FBS games separately (the pooled FCS
side has no weekly data, so those rows are imputed on one side).

Usage:
    python -m CFB.model.eval_advanced
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import evaluate
from CFB.model import advanced as adv
from CFB.model.elo import load_games, replay

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
OUT = ARTIFACTS / "advanced_eval.json"
TRAIN_FROM, VALIDATION, TEST = 2005, 2023, (2024, 2025)
C_GRID = [0.03, 0.1, 0.3, 1.0]

CORE = adv.ELO + adv.CORE
GROUPS = {
    "1_elo": adv.ELO,
    "2_core": CORE,
    "3_success": CORE + adv.SUCCESS,
    "4_early_explosive": CORE + adv.EARLY_EXPLOSIVE,
    "5_drive": CORE + adv.DRIVE,
    "6_situational": CORE + adv.SITUATIONAL,
    "x_splits": CORE + adv.SPLITS,
}


def build_table() -> pd.DataFrame:
    games = load_games()
    _, history = replay(games)
    history = history.merge(games[["game_id", "home_id", "away_id"]], on="game_id", how="left")
    table = adv.build_game_table(history[history["season"] >= 2004])
    table = table[table["home_win"] != 0.5].copy()
    table["y"] = (table["home_win"] == 1.0).astype(int)
    table["fbs_vs_fbs"] = ~table["home_fcs"].astype(bool) & ~table["away_fcs"].astype(bool)
    return table.reset_index(drop=True)


def _fit_score(table, feats, train_mask, test_mask, C):
    model = evaluate.make_logistic(C).fit(table.loc[train_mask, feats], table.loc[train_mask, "y"])
    p = model.predict_proba(table.loc[test_mask, feats])[:, 1]
    return evaluate.score_binary(table.loc[test_mask, "y"].to_numpy(), p), model


def run(table: pd.DataFrame) -> dict:
    season = table["season"]
    train = (season >= TRAIN_FROM) & (season < VALIDATION)
    valid = season == VALIDATION
    test = (season >= TEST[0]) & (season <= TEST[1])
    fit_for_test = (season >= TRAIN_FROM) & (season < TEST[0])
    report = {"train": f"{TRAIN_FROM}-{VALIDATION - 1}", "validation": VALIDATION, "test": list(TEST),
              "n_train": int(train.sum()), "n_valid": int(valid.sum()), "n_test": int(test.sum()),
              "note": "Elo params tuned on 2005-2023 (CFB/model/tune.py); only 2024-2025 is clean "
                      "for the Elo baseline. Every candidate sits on the same Elo."}

    feats_all = adv.all_features()
    cov = table.groupby("season").apply(lambda g: pd.Series({
        "rows": len(g), "fbs_vs_fbs": g["fbs_vs_fbs"].mean(),
        "core_covered": g[adv.CORE].notna().all(axis=1).mean(),
        "all_covered": g[feats_all].notna().all(axis=1).mean(),
        "core_covered_fbs_only": g.loc[g["fbs_vs_fbs"], adv.CORE].notna().all(axis=1).mean()})).reset_index()
    report["coverage"] = cov.round(3).to_dict("records")
    wk = table[table["season"].between(2019, 2025) & (table["season_type"] == "regular")]
    report["coverage_by_week_2019_2025"] = wk.groupby("week").apply(
        lambda g: g[adv.CORE].notna().all(axis=1).mean()).round(3).to_dict()

    # validation: C per group
    chosen, grid = {}, []
    for name, feats in GROUPS.items():
        best = None
        for C in C_GRID:
            sc, _ = _fit_score(table, feats, train, valid, C)
            grid.append({"group": name, "C": C, **{k: v for k, v in sc.row("").items() if k != "model"}})
            if best is None or sc.log_loss < best[1]:
                best = (C, sc.log_loss)
        chosen[name] = best[0]
    report["validation_grid"] = grid

    pool = {"core": adv.CORE, "success": adv.SUCCESS, "early_explosive": adv.EARLY_EXPLOSIVE,
            "drive": adv.DRIVE, "situational": adv.SITUATIONAL, "splits": adv.SPLITS}
    current, remaining, steps = [], list(pool), []
    current_ll = _fit_score(table, adv.ELO, train, valid, chosen["1_elo"])[0].log_loss
    while remaining:
        trial = []
        for g in remaining:
            feats = adv.ELO + [f for h in current + [g] for f in pool[h]]
            trial.append((_fit_score(table, feats, train, valid, 0.1)[0].log_loss, g))
        ll, g = min(trial)
        steps.append({"added": g, "validation_log_loss": round(ll, 5), "gain": round(current_ll - ll, 5)})
        if ll >= current_ll - 1e-4:
            break
        current.append(g); current_ll = ll; remaining.remove(g)
    combined = adv.ELO + [f for h in current for f in pool[h]]
    best_C = min(C_GRID, key=lambda C: _fit_score(table, combined, train, valid, C)[0].log_loss)
    report["combined"] = {"groups": current, "C": best_C, "selection": steps, "features": combined}

    # test, once, overall and FBS-vs-FBS
    results = {}
    for label, scope in (("overall", test), ("fbs_vs_fbs", test & table["fbs_vs_fbs"])):
        rows = []
        base, _ = _fit_score(table, adv.ELO, fit_for_test, scope, chosen["1_elo"])
        for name, feats in GROUPS.items():
            sc, _ = _fit_score(table, feats, fit_for_test, scope, chosen[name])
            rows.append({**sc.row(name), "C": chosen[name],
                         "paired_se_vs_elo": round(evaluate.paired_se(base, sc), 2)})
        sc, model = _fit_score(table, combined, fit_for_test, scope, best_C)
        rows.append({**sc.row("7_combined"), "C": best_C,
                     "paired_se_vs_elo": round(evaluate.paired_se(base, sc), 2)})
        by_season = []
        for s in TEST:
            m = scope & (season == s)
            b, _ = _fit_score(table, adv.ELO, fit_for_test, m, chosen["1_elo"])
            c, _ = _fit_score(table, CORE, fit_for_test, m, chosen["2_core"])
            k, _ = _fit_score(table, combined, fit_for_test, m, best_C)
            by_season.append({"season": s, "n": b.n, "elo": round(b.log_loss, 5), "core": round(c.log_loss, 5),
                              "combined": round(k.log_loss, 5), "paired_se_combined": round(evaluate.paired_se(b, k), 2)})
        results[label] = {"table": rows, "by_season": by_season}
    # validation-season numbers too, labelled as such
    vrows = []
    vbase, _ = _fit_score(table, adv.ELO, train, valid, chosen["1_elo"])
    for name, feats in GROUPS.items():
        sc, _ = _fit_score(table, feats, train, valid, chosen[name])
        vrows.append({**sc.row(name), "paired_se_vs_elo": round(evaluate.paired_se(vbase, sc), 2)})
    sc, _ = _fit_score(table, combined, train, valid, best_C)
    vrows.append({**sc.row("7_combined"), "paired_se_vs_elo": round(evaluate.paired_se(vbase, sc), 2)})
    results["validation_2023_not_clean"] = {"table": vrows}
    report["results"] = results

    _, model = _fit_score(table, combined, fit_for_test, test, best_C)
    report["combined_coefficients_standardised"] = dict(zip(
        combined, model.named_steps["clf"].coef_[0].round(4).tolist()))
    return report


def print_report(r: dict) -> None:
    print(f"\nTrain {r['train']} ({r['n_train']}) · validation {r['validation']} ({r['n_valid']}) · "
          f"test {r['test']} ({r['n_test']})")
    print(r["note"])
    print("\nCoverage by season:")
    print(pd.DataFrame(r["coverage"]).to_string(index=False))
    print("\nCore coverage by regular-season week, 2019-2025:")
    print({int(k): v for k, v in r["coverage_by_week_2019_2025"].items()})
    print("\nGreedy selection on 2023:")
    print(pd.DataFrame(r["combined"]["selection"]).to_string(index=False))
    print(f"  -> combined = elo + {r['combined']['groups']}  (C={r['combined']['C']})")
    for label, res in r["results"].items():
        print(f"\n{label}:")
        print(pd.DataFrame(res["table"]).to_string(index=False))
        if "by_season" in res:
            print(pd.DataFrame(res["by_season"]).to_string(index=False))
    print("\nStandardised coefficients of the combined logistic (fit 2005-2023):")
    for k, v in sorted(r["combined_coefficients_standardised"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:28s} {v:+.4f}")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    report = run(build_table())
    print_report(report)
    ARTIFACTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
