"""SRS benchmark for the CFB Elo (DATA_PULL_PLAN.md §3.1's "something to
lose to"), plus the flat-vs-conference-regression comparison.

CFR's standings pages are Cloudflare-blocked for automated pulls, but SRS
itself is a published algorithm — rating = average margin + average
opponent rating — so this module computes it from our own game spine
instead of downloading theirs. Weekly and walk-forward: the SRS used to
predict a week-N game is solved only on that season's games from weeks
< N, so it never sees the future. Note it also never sees *prior seasons*,
which is exactly the handicap Elo's carryover is supposed to exploit —
early-season weeks are where Elo should win.

Probability calibration: SRS predicts a point margin (rating diff + HFA);
the margin→probability logistic scale is fitted once on the 2000-2004
burn-in and frozen, same discipline as everything else here.

Usage
    python3 -m CFB.benchmark            # SRS vs Elo (flat + cluster regression)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from CFB import elo
from CFB.conferences import load_clusters

HFA_POINTS = 2.7  # the fitted Elo HFA in point terms (50 / 18.6)
SRS_ITERS = 200
MIN_WEEK = 4  # earlier weeks have too few games for a stable SRS solve


def solve_srs(games: pd.DataFrame) -> dict[str, float]:
    """rating_i = mean margin_i + mean(opponent ratings), mean-centred."""
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    margin_sum = np.zeros(n)
    count = np.zeros(n)
    opp = np.zeros((n, n))
    for r in games.itertuples(index=False):
        h, a = idx[r.home_team], idx[r.away_team]
        m = float(r.home_score) - float(r.away_score)
        margin_sum[h] += m
        margin_sum[a] -= m
        count[h] += 1
        count[a] += 1
        opp[h, a] += 1
        opp[a, h] += 1

    count[count == 0] = 1
    mean_margin = margin_sum / count
    opp = opp / count[:, None]

    r = mean_margin.copy()
    for _ in range(SRS_ITERS):
        r_new = mean_margin + opp @ r
        r_new -= r_new.mean()
        if np.abs(r_new - r).max() < 1e-9:
            r = r_new
            break
        r = r_new
    return {t: float(r[idx[t]]) for t in teams}


def srs_predictions(games: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward weekly SRS point predictions for weeks >= MIN_WEEK."""
    played = games.dropna(subset=["home_score", "away_score"])
    rows = []
    for season, sg in played.groupby("season"):
        for week in sorted(sg["week"].unique()):
            if week < MIN_WEEK:
                continue
            past = sg[sg["week"] < week]
            target = sg[sg["week"] == week]
            ratings = solve_srs(past)
            for r in target.itertuples(index=False):
                if r.home_team not in ratings or r.away_team not in ratings:
                    continue
                pred = (
                    ratings[r.home_team]
                    - ratings[r.away_team]
                    + (HFA_POINTS if str(r.location) == "Home" else 0.0)
                )
                rows.append(
                    {
                        "season": int(season),
                        "week": int(week),
                        "gameday": r.gameday,
                        "home_team": r.home_team,
                        "away_team": r.away_team,
                        "srs_pred_margin": pred,
                        "home_win": float(r.home_score > r.away_score),
                    }
                )
    return pd.DataFrame(rows)


def fit_logistic_scale(preds: pd.DataFrame) -> float:
    """1-D fit of s in P(home) = 1/(1+exp(-margin/s)) on the burn-in years."""
    burn = preds[preds["season"] <= elo.BURN_IN_THROUGH]
    y = burn["home_win"].to_numpy()
    m = burn["srs_pred_margin"].to_numpy()
    best_s, best_ll = None, np.inf
    for s in np.arange(4.0, 20.1, 0.25):
        p = np.clip(1.0 / (1.0 + np.exp(-m / s)), 1e-6, 1 - 1e-6)
        ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        if ll < best_ll:
            best_s, best_ll = float(s), ll
    return best_s


def _score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    acc = float(((p > 0.5) == y.astype(bool)).mean())
    return {"games": int(len(y)), "log_loss": round(ll, 4), "accuracy": round(acc, 4)}


def main() -> None:
    argparse.ArgumentParser().parse_args()

    games = elo.load_games(pool=True)

    # --- SRS side ---------------------------------------------------------
    preds = srs_predictions(games)
    scale = fit_logistic_scale(preds)
    print(f"SRS logistic scale (fit on burn-in): {scale} pts")
    preds["srs_prob"] = 1.0 / (1.0 + np.exp(-preds["srs_pred_margin"] / scale))

    # --- Elo side, same game subset --------------------------------------
    key = ["season", "gameday", "home_team", "away_team"]
    for label, clusters in (("flat", None), ("cluster", load_clusters())):
        df, _ = elo.walk_forward(games, clusters=clusters)
        df = df.dropna(subset=["home_score", "away_score"])
        merged = preds.merge(
            df[key + ["elo_home_prob"]], on=key, how="inner", validate="1:1"
        )
        eval_set = merged[merged["season"] >= elo.EVAL_FROM]
        y = eval_set["home_win"].to_numpy()
        if label == "flat":
            print(f"\ncommon eval set: {len(eval_set)} games, "
                  f"{elo.EVAL_FROM}+, weeks {MIN_WEEK}+")
            print("SRS        :", _score(y, eval_set["srs_prob"].to_numpy()))
        print(f"Elo {label:7}:", _score(y, eval_set["elo_home_prob"].to_numpy()))

    # --- full-schedule effect of cluster regression -----------------------
    print("\nfull schedule (all weeks), 2005+:")
    print("Elo flat   :", elo.evaluate(games))
    print("Elo cluster:", elo.evaluate(games, clusters=load_clusters()))


if __name__ == "__main__":
    main()
