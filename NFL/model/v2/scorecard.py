"""Full performance scorecard for the production model.

Reports the metrics that actually characterise a probabilistic sports model,
not just accuracy:

- **Accuracy / AUC** — can it rank games correctly.
- **Log loss / Brier** — are the probabilities themselves any good.
- **McFadden R2** — improvement in log-likelihood over an intercept-only model.
- **Efron R2 / Brier skill** — variance explained versus always predicting the
  base rate. For a binary target these are the honest analogues of the R2 you
  would quote for a regression; a raw R2 on a 0/1 outcome is not meaningful.
- **Calibration slope and intercept** — refit a logistic on the model's own
  logit. Slope 1.0 and intercept 0.0 means perfectly calibrated; slope below 1
  means overconfident.
- **ECE** — average gap between predicted and actual in probability bins.
- **Top-3 hit rate** — the weekly product metric.
- **ATS / ROI** — what it would have done as a bet.

Everything is walk-forward out of sample: for season S the model saw only
seasons before S-1 and calibrated on S-1.

Usage
    python3 -m NFL.model.v2.scorecard
    python3 -m NFL.model.v2.scorecard --save
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             roc_auc_score)

from .compare_models import walk_forward_model, top_k_weekly
from .dataset import build_dataset

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
OUT_DIR = ARTIFACTS / "scorecard"
RANKING = ARTIFACTS / "importance" / "uniform_ranking.csv"

EVAL_FROM = 2015
HALF_LIFE = 6.0
BREAK_EVEN = 110 / 210


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expected_calibration_error(y, p, bins: int = 10) -> float:
    idx = np.clip((np.asarray(p) * bins).astype(int), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            err += m.mean() * abs(np.asarray(y)[m].mean() - np.asarray(p)[m].mean())
    return float(err)


def kpis(y, p, label: str, weeks: pd.DataFrame | None = None) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    base = y.mean()

    ll = log_loss(y, p)
    ll_null = log_loss(y, np.full(len(y), base))
    brier = brier_score_loss(y, p)
    brier_null = float(np.mean((base - y) ** 2))

    # Calibration curve refit on the model's own logit.
    cal = LogisticRegression(C=1e6)
    cal.fit(_logit(p).reshape(-1, 1), y)

    out = {
        "model": label,
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, (p >= 0.5).astype(int))), 4),
        "auc": round(float(roc_auc_score(y, p)), 4),
        "log_loss": round(float(ll), 4),
        "brier": round(float(brier), 4),
        "mcfadden_r2": round(float(1 - ll / ll_null), 4),
        "efron_r2": round(float(1 - brier / brier_null), 4),
        "cal_slope": round(float(cal.coef_[0][0]), 3),
        "cal_intercept": round(float(cal.intercept_[0]), 3),
        "ece": round(expected_calibration_error(y, p), 4),
    }
    if weeks is not None:
        t = top_k_weekly(weeks.assign(prob=p, y=y), k=3)
        out["top3_hit"] = round(t["hit_rate"], 4)
        out["weeks_swept"] = round(t["weeks_swept"], 4)
    return out


def ats_and_roi(preds: pd.DataFrame, edge: float = 0.02) -> dict:
    """Flat-stake moneyline result at the actual closing price."""
    p, mkt, y = preds["prob"].to_numpy(), preds["market_prob"].to_numpy(), preds["y"].to_numpy()
    take_home, take_away = (p - mkt) >= edge, ((1 - p) - (1 - mkt)) >= edge
    bet = take_home | take_away
    if not bet.any():
        return {"bets": 0}
    picked_home = take_home[bet]
    won = np.where(picked_home, y[bet] == 1, y[bet] == 0)
    price = np.where(picked_home, mkt[bet], 1 - mkt[bet])
    payout = 1.0 / np.clip(price, 0.02, 0.98) - 1.0
    units = np.where(won, payout, -1.0)
    return {
        "bets": int(bet.sum()),
        "win_pct": round(float(won.mean()), 4),
        "units": round(float(units.sum()), 2),
        "roi": round(float(units.mean()), 4),
        "roi_se": round(float(units.std(ddof=1) / np.sqrt(len(units))), 4),
    }


def production_features() -> list[str]:
    """Score exactly what the weekly report ships, not a freshly-ranked list.

    The re-ranked importance table is *not* used to pick the subset. Ranking
    measures each feature's contribution on its own; it does not identify the
    best-performing combination. When the ranking was regenerated after the Elo
    change it promoted ``market_vig`` and ``travel_miles`` over
    ``home_roll_margin`` and ``away_ppg_diff_std`` — and that swap costs 0.0014
    of log loss (0.6154 vs 0.6140). The pinned list stays.
    """
    from data_jobs.reports.weekly_nfl_report import PICKS_FEATURES
    return list(PICKS_FEATURES)


def run(save: bool = False) -> pd.DataFrame:
    feats = production_features()
    print(f"production features ({len(feats)}): {', '.join(feats)}\n", flush=True)

    df_new = build_dataset(with_squad=True)                      # qb_talent Elo
    df_old = build_dataset(with_squad=True, elo_variant="base")  # plain Elo

    rows, preds_store = [], {}
    for label, df in (("production (qb+talent Elo)", df_new),
                      ("previous (base Elo)", df_old)):
        pr = walk_forward_model(df, "win", "logistic", EVAL_FROM, HALF_LIFE, features=feats)
        preds_store[label] = pr
        rows.append(kpis(pr["y"], pr["prob"], label, weeks=pr))

    ref = preds_store["production (qb+talent Elo)"]
    rows.append(kpis(ref["y"], ref["market_prob"], "market (closing line)", weeks=ref))
    # walk_forward_model only carries a fixed column set, so pull the raw Elo
    # probability back off the dataset to score it standalone.
    elo_p = ref["game_id"].map(df_new.set_index("game_id")["elo_home_prob"])
    rows.append(kpis(ref["y"], elo_p, "qb+talent Elo alone", weeks=ref))
    rows.append(kpis(ref["y"], np.full(len(ref), ref["y"].mean()),
                     "always base rate", weeks=ref))

    table = pd.DataFrame(rows)
    print("=== KPIs, walk-forward out of sample, %d-2025 ===" % EVAL_FROM)
    print(table.to_string(index=False), flush=True)

    print("\n=== Flat-stake moneyline backtest (edge >= 2 pts) ===")
    bt = pd.DataFrame([{"model": k, **ats_and_roi(v)} for k, v in preds_store.items()])
    print(bt.to_string(index=False), flush=True)

    print("\n=== By season (production) ===")
    per = []
    for s, g in ref.groupby("season"):
        per.append({"season": int(s), **{k: v for k, v in
                    kpis(g["y"], g["prob"], str(s), weeks=g).items()
                    if k in ("n", "accuracy", "auc", "brier", "top3_hit")}})
    per_df = pd.DataFrame(per)
    print(per_df.to_string(index=False), flush=True)

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUT_DIR / "kpis.csv", index=False)
        bt.to_csv(OUT_DIR / "backtest.csv", index=False)
        per_df.to_csv(OUT_DIR / "by_season.csv", index=False)
        ref.to_csv(OUT_DIR / "oos_predictions.csv", index=False)
        print(f"\nsaved -> {OUT_DIR}")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", action="store_true")
    run(ap.parse_args().save)


if __name__ == "__main__":
    main()
