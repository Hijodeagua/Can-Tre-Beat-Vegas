"""Score upcoming games with the saved v2 models.

Fills the gap the roadmap flagged: the models were trained but nothing ever
wrote predictions for games that hadn't been played. Output lands in
``data/predictions/`` in the shape ``data_jobs/export_web_json.py`` expects.

    python3 -m NFL.model.v2.predict --season 2026
    python3 -m NFL.model.v2.predict --season 2026 --week 1 --write

A caveat worth keeping in view: Elo and rolling form only advance when games
are *played*. Scoring an entire future season therefore prices every week off
the same end-of-2025 ratings. Re-run it weekly once results start landing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .dataset import build_dataset, feature_matrix
from .train import TARGETS

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data" / "predictions"


def _load(target: str) -> tuple[lgb.Booster, dict | None]:
    model_path = ARTIFACTS / f"lgbm_{target}.txt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found — run `python3 -m NFL.model.v2.train "
            "--target all --save` first"
        )
    booster = lgb.Booster(model_file=str(model_path))
    cal_path = ARTIFACTS / f"calibration_{target}.json"
    cal = json.loads(cal_path.read_text()) if cal_path.exists() else None
    return booster, cal


def _calibrate(prob: np.ndarray, cal: dict | None) -> np.ndarray:
    if not cal:
        return prob
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    z = cal["coef"] * np.log(p / (1 - p)) + cal["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def predict(season: int, week: int | None = None, include_played: bool = False) -> pd.DataFrame:
    df = build_dataset()
    sel = df[df["season"] == season]
    if not include_played:
        sel = sel[sel["home_score"].isna()]
    if week is not None:
        sel = sel[sel["week"] == week]
    if sel.empty:
        raise SystemExit(f"no games to score for season={season} week={week}")

    out = sel[[
        "game_id", "season", "week", "gameday", "home_team", "away_team",
        "spread_line", "total_line", "market_home_prob", "elo_home_prob", "elo_spread",
    ]].copy()

    X = feature_matrix(sel)
    for target in TARGETS:
        booster, cal = _load(target)
        out[f"prob_{target}"] = _calibrate(booster.predict(X), cal)

    out["pick_su"] = np.where(out["prob_win"] >= 0.5, out["home_team"], out["away_team"])
    out["conf_su"] = np.where(out["prob_win"] >= 0.5, out["prob_win"], 1 - out["prob_win"])
    # Positive edge = model likes the home side more than the market does.
    out["edge_vs_market"] = out["prob_win"] - out["market_home_prob"]

    home_line = np.where(out["spread_line"] > 0, "-", "+") + out["spread_line"].abs().astype(str)
    away_line = np.where(out["spread_line"] > 0, "+", "-") + out["spread_line"].abs().astype(str)
    out["pick_ats"] = np.where(out["prob_ats"] >= 0.5,
                               out["home_team"] + " " + home_line,
                               out["away_team"] + " " + away_line)
    out["pick_total"] = np.where(out["prob_total"] >= 0.5, "Over ", "Under ") + \
        out["total_line"].astype(str)
    out["model"] = "lgbm_v2"
    return out.sort_values(["week", "gameday", "home_team"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--include-played", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    preds = predict(args.season, args.week, args.include_played)
    cols = ["week", "gameday", "away_team", "home_team", "spread_line", "total_line",
            "prob_win", "market_home_prob", "edge_vs_market", "pick_ats", "pick_total"]
    show = preds[cols].head(24).copy()
    show["gameday"] = show["gameday"].dt.date
    print(show.round(3).to_string(index=False))
    print(f"\n{len(preds)} games scored")

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        suffix = f"_week{args.week}" if args.week else ""
        path = OUT_DIR / f"nfl_{args.season}{suffix}_v2.csv"
        preds.to_csv(path, index=False)
        print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
