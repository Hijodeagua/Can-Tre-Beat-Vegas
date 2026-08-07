"""Walk-forward training + honest backtest for the v2 NFL model.

The old trainer used a single 2024/2025 temporal split on a box-score file
that ended in week 3 of 2025, so its "test set" was 144 games and the model
had never seen a completed season. This one retrains **once per season**:
for season S the model is fit on every game before S, calibrated on S-1, and
scored on S. Every prediction it reports is out of sample.

Targets
    win   — home team wins outright        (baseline: closing moneyline)
    ats   — home team covers the spread    (baseline: coin flip / 52.4% to profit)
    total — game goes over the total        (baseline: coin flip)

Usage
    python3 -m NFL.model.v2.train --target ats
    python3 -m NFL.model.v2.train --target all --save
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .dataset import FEATURE_COLS, build_dataset, feature_matrix

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "win": "home_win",
    "ats": "home_cover",
    "total": "over",
}

# Flat -110 juice: risk 110 to win 100.
JUICE_PAYOUT = 100 / 110
BREAK_EVEN = 110 / 210  # 0.5238

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 60,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbose": -1,
    "seed": 17,
}
NUM_ROUNDS = 400


def _platt(raw: np.ndarray, y: np.ndarray) -> LogisticRegression | None:
    """Fit a 1-D logistic on the model's logit — corrects systematic over/under-confidence."""
    if len(np.unique(y)) < 2 or len(y) < 50:
        return None
    logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1 - 1e-6))
    lr = LogisticRegression(C=1.0)
    lr.fit(logit.reshape(-1, 1), y)
    return lr


def _apply_platt(cal: LogisticRegression | None, raw: np.ndarray) -> np.ndarray:
    if cal is None:
        return raw
    logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1 - 1e-6))
    return cal.predict_proba(logit.reshape(-1, 1))[:, 1]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _fit_stacker(raw: np.ndarray, mkt: np.ndarray, y: np.ndarray) -> LogisticRegression | None:
    """Blend model and market in logit space.

    This is the question the repo is actually asking: after the closing line has
    had its say, does the model carry any information left over? A near-zero
    weight on the model column means no.
    """
    if len(np.unique(y)) < 2 or len(y) < 50:
        return None
    X = np.column_stack([_logit(mkt), _logit(raw)])
    lr = LogisticRegression(C=1.0)
    lr.fit(X, y)
    return lr


def market_prob(df: pd.DataFrame, target: str) -> np.ndarray:
    """The number to beat, as a probability of the modelled event."""
    if target == "win":
        return df["market_home_prob"].to_numpy(dtype=float)
    # Books price spreads and totals at roughly a coin flip on both sides.
    return np.full(len(df), 0.5)


def walk_forward(
    df: pd.DataFrame, target: str, start_season: int = 2010
) -> pd.DataFrame:
    """Out-of-sample predictions for every season >= ``start_season``."""
    y_col = TARGETS[target]
    played = df.dropna(subset=[y_col]).copy()
    seasons = sorted(s for s in played["season"].unique() if s >= start_season)

    out = []
    for season in seasons:
        fit = played[played["season"] < season - 1]
        cal = played[played["season"] == season - 1]
        test = played[played["season"] == season]
        if len(fit) < 500 or test.empty:
            continue

        booster = lgb.train(
            LGB_PARAMS,
            lgb.Dataset(feature_matrix(fit), label=fit[y_col].astype(int)),
            num_boost_round=NUM_ROUNDS,
        )
        raw_cal = booster.predict(feature_matrix(cal))
        y_cal = cal[y_col].astype(int).to_numpy()
        platt = _platt(raw_cal, y_cal)
        stacker = _fit_stacker(raw_cal, market_prob(cal, target), y_cal) if target == "win" else None

        raw_test = booster.predict(feature_matrix(test))
        rec = test[[
            "game_id", "season", "week", "gameday", "game_type",
            "home_team", "away_team", "home_score", "away_score",
            "spread_line", "total_line", "margin", "ats_margin_home", "total_margin",
            "elo_home_prob", "market_home_prob", y_col,
        ]].copy()
        rec["prob_raw"] = raw_test
        rec["prob"] = _apply_platt(platt, raw_test)
        rec["market_prob"] = market_prob(test, target)
        if stacker is not None:
            X = np.column_stack([_logit(rec["market_prob"].to_numpy()), _logit(raw_test)])
            rec["prob_stacked"] = stacker.predict_proba(X)[:, 1]
            rec["stack_w_market"] = float(stacker.coef_[0][0])
            rec["stack_w_model"] = float(stacker.coef_[0][1])
        else:
            rec["prob_stacked"] = rec["prob"]
            rec["stack_w_market"] = np.nan
            rec["stack_w_model"] = np.nan
        rec["target"] = target
        rec["y"] = test[y_col].astype(int).to_numpy()
        out.append(rec)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def evaluate(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "model": name,
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
    }


def baseline_table(preds: pd.DataFrame, target: str) -> pd.DataFrame:
    y = preds["y"].to_numpy()
    rows = [
        evaluate("lgbm_v2", y, preds["prob"].to_numpy()),
        evaluate("lgbm_v2_uncalibrated", y, preds["prob_raw"].to_numpy()),
        evaluate("elo_only", y, preds["elo_home_prob"].to_numpy()),
    ]
    if target == "win":
        rows.append(evaluate("market_moneyline", y, preds["market_home_prob"].to_numpy()))
        rows.append(evaluate("market_plus_model", y, preds["prob_stacked"].to_numpy()))
        rows.append(evaluate("always_home", y, np.full(len(y), 0.55)))
    else:
        rows.append(evaluate("coin_flip", y, np.full(len(y), 0.5)))
        if target == "ats":
            fav_home = (preds["spread_line"] > 0).astype(float).to_numpy()
            rows.append(evaluate("home_favorite_covers", y, np.where(fav_home == 1, 0.55, 0.45)))
    return pd.DataFrame(rows)


def backtest(preds: pd.DataFrame, target: str, edge: float = 0.02) -> pd.DataFrame:
    """Flat-stake ROI. Bet the side the model likes when its edge clears ``edge``.

    Spreads and totals are priced at -110. Straight-up bets use the actual
    closing moneyline implied by ``market_home_prob`` (payout = 1/p - 1), which
    is the honest price a bettor would have got.
    """
    rows = []
    for season, grp in preds.groupby("season"):
        p = grp["prob"].to_numpy()
        mkt = grp["market_prob"].to_numpy()
        y = grp["y"].to_numpy()

        # Model's edge on the home side and the away side.
        edge_home = p - mkt
        edge_away = (1 - p) - (1 - mkt)
        take_home = edge_home >= edge
        take_away = edge_away >= edge
        bet = take_home | take_away
        if not bet.any():
            rows.append({"season": int(season), "bets": 0, "wins": 0, "win_pct": np.nan,
                         "units": 0.0, "roi": np.nan})
            continue

        picked_home = take_home[bet]
        won = np.where(picked_home, y[bet] == 1, y[bet] == 0)

        if target == "win":
            price_prob = np.where(picked_home, mkt[bet], 1 - mkt[bet])
            payout = (1.0 / np.clip(price_prob, 0.02, 0.98)) - 1.0
        else:
            payout = np.full(bet.sum(), JUICE_PAYOUT)

        units = float(np.sum(np.where(won, payout, -1.0)))
        rows.append({
            "season": int(season),
            "bets": int(bet.sum()),
            "wins": int(won.sum()),
            "win_pct": float(won.mean()),
            "units": round(units, 2),
            "roi": round(units / bet.sum(), 4),
        })

    tot = pd.DataFrame(rows)
    if tot["bets"].sum():
        tot.loc[len(tot)] = {
            "season": "ALL",
            "bets": int(tot["bets"].sum()),
            "wins": int(tot["wins"].sum()),
            "win_pct": round(tot["wins"].sum() / tot["bets"].sum(), 4),
            "units": round(tot["units"].sum(), 2),
            "roi": round(tot["units"].sum() / tot["bets"].sum(), 4),
        }
    return tot


def per_season(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, grp in preds.groupby("season"):
        y, p = grp["y"].to_numpy(), grp["prob"].to_numpy()
        rows.append({
            "season": int(season),
            "n": len(y),
            "accuracy": round(float(accuracy_score(y, (p >= 0.5).astype(int))), 4),
            "brier": round(float(brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
        })
    return pd.DataFrame(rows)


def fit_final(df: pd.DataFrame, target: str) -> tuple[lgb.Booster, LogisticRegression | None]:
    """Refit on everything played, calibrated on the most recent season."""
    y_col = TARGETS[target]
    played = df.dropna(subset=[y_col])
    last = int(played["season"].max())
    fit = played[played["season"] < last]
    cal = played[played["season"] == last]

    booster = lgb.train(
        LGB_PARAMS,
        lgb.Dataset(feature_matrix(fit), label=fit[y_col].astype(int)),
        num_boost_round=NUM_ROUNDS,
    )
    platt = _platt(booster.predict(feature_matrix(cal)), cal[y_col].astype(int).to_numpy())
    return booster, platt


def run(target: str, start_season: int, save: bool, df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = build_dataset() if df is None else df
    print(f"\n=== target: {target} ===")
    preds = walk_forward(df, target, start_season)
    if preds.empty:
        print("no predictions produced")
        return preds

    metrics = baseline_table(preds, target)
    print(f"\nOut-of-sample, {preds['season'].min()}-{preds['season'].max()} "
          f"({len(preds)} games)")
    print(metrics.to_string(index=False))

    print("\nBy season (calibrated lgbm_v2):")
    print(per_season(preds).to_string(index=False))

    bt = backtest(preds, target)
    print(f"\nFlat-stake backtest (edge >= 2 pts, break-even {BREAK_EVEN:.4f} at -110):")
    print(bt.to_string(index=False))

    if save:
        booster, platt = fit_final(df, target)
        booster.save_model(str(ARTIFACTS / f"lgbm_{target}.txt"))
        imp = pd.DataFrame({
            "feature": booster.feature_name(),
            "gain": booster.feature_importance("gain"),
            "split": booster.feature_importance("split"),
        }).sort_values("gain", ascending=False)
        imp.to_csv(ARTIFACTS / f"importance_{target}.csv", index=False)

        cal_payload = None
        if platt is not None:
            cal_payload = {"coef": float(platt.coef_[0][0]), "intercept": float(platt.intercept_[0])}
        (ARTIFACTS / f"calibration_{target}.json").write_text(json.dumps(cal_payload, indent=2))
        (ARTIFACTS / f"features_{target}.json").write_text(json.dumps(FEATURE_COLS, indent=2))

        metrics.to_csv(ARTIFACTS / f"metrics_{target}.csv", index=False)
        per_season(preds).to_csv(ARTIFACTS / f"by_season_{target}.csv", index=False)
        bt.to_csv(ARTIFACTS / f"backtest_{target}.csv", index=False)
        preds.to_csv(ARTIFACTS / f"oos_predictions_{target}.csv", index=False)
        print(f"\nsaved artifacts -> {ARTIFACTS}")
        print("\nTop features by gain:")
        print(imp.head(12).to_string(index=False))

    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="all", choices=[*TARGETS, "all"])
    ap.add_argument("--start-season", type=int, default=2010)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    df = build_dataset()
    print(f"dataset: {len(df)} games, {df['season'].min()}-{df['season'].max()}")
    targets = list(TARGETS) if args.target == "all" else [args.target]
    for t in targets:
        run(t, args.start_season, args.save, df=df)


if __name__ == "__main__":
    main()
