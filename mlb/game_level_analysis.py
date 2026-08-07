"""
Game-level (head-to-head) test: does last season's stat level predict an
individual game's winner better or worse than the SAME season's stat level?

Two comparisons, because "same season" can mean two very different things:

  A. SAME-SEASON FINAL vs PRIOR-SEASON FINAL (oracle vs legitimate prior)
     Same-season-final stats are computed from the whole season, including
     games *after* the one being predicted -- that's look-ahead / an
     explanatory ceiling, not a real predictor. Prior-season-final has zero
     look-ahead: it's exactly what you'd know on Opening Day. This isolates
     "how much does knowing the full season's truth help vs. a genuine
     year-old prior."

  B. ROLLING IN-SEASON (no look-ahead, cumulative through the prior game)
     vs PRIOR-SEASON FINAL, evaluated at increasing games-played cutoffs.
     Both sides are legitimate (zero look-ahead) here, so this answers the
     practical question: how many games into a season until this year's
     sample beats last year's whole season?

Outputs -> data/mlb/analysis/game_level_*.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "mlb"
AN = DATA / "analysis"

# stat -> True if HIGHER is better (sign convention for the differential)
STATS = {
    "ERA": False, "ERA+": True, "WHIP": False, "FIP": False,
    "SO/BB": True, "RA/G": False, "R/Gm": True, "RunDiff/G": True,
    "OPS+": True, "OBP": True, "SLG": True, "OPS": True, "BA": True,
}

# seasons excluded from ALL prior-season comparisons: 2009 (no prior year
# in the dataset) and 2021 (prior year 2020 is the 60-game season, not a
# comparable "full season"). 2020 itself is dropped as the game season too
# (60 games is a different animal for both sides of the comparison).
VALID_SEASONS = [s for s in range(2009, 2027) if s not in (2009, 2020, 2021)]


def load_games() -> pd.DataFrame:
    g = pd.read_csv(DATA / "games_2009_2026.csv")
    g = g.sort_values(["date", "game_num", "home"]).reset_index(drop=True)
    g["gidx"] = np.arange(len(g))
    g["home_win"] = (g.home_score > g.away_score).astype(int)
    return g


def load_panel() -> pd.DataFrame:
    m = pd.read_csv(AN / "merged_panel.csv")
    return m[["Season", "franchise"] + list(STATS.keys())]


# ---------------------------------------------------------- A. oracle vs prior


def eval_feature(x: np.ndarray, y: np.ndarray) -> dict:
    r, p = sps.pointbiserialr(y, x)
    X = x.reshape(-1, 1)
    clf = LogisticRegression().fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba > 0.5).astype(int)
    return {
        "point_biserial_r": round(float(r), 4),
        "p": float(p),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "log_loss": round(float(log_loss(y, proba)), 5),
        "n": len(y),
    }


def oracle_vs_prior(games: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    g = games[games.season.isin(VALID_SEASONS)].copy()

    same_home = panel.rename(
        columns={c: f"{c}_same_home" for c in STATS} | {"franchise": "home_fr"})
    same_away = panel.rename(
        columns={c: f"{c}_same_away" for c in STATS} | {"franchise": "away_fr"})
    prior = panel.copy()
    prior["Season"] = prior["Season"] + 1  # prior stats attach to NEXT season
    prior_home = prior.rename(
        columns={c: f"{c}_prior_home" for c in STATS} | {"franchise": "home_fr"})
    prior_away = prior.rename(
        columns={c: f"{c}_prior_away" for c in STATS} | {"franchise": "away_fr"})

    d = (g.merge(same_home, left_on=["season", "home_fr"],
                right_on=["Season", "home_fr"])
          .merge(same_away, left_on=["season", "away_fr"],
                right_on=["Season", "away_fr"], suffixes=("", "_sa"))
          .merge(prior_home, left_on=["season", "home_fr"],
                right_on=["Season", "home_fr"], suffixes=("", "_ph"))
          .merge(prior_away, left_on=["season", "away_fr"],
                right_on=["Season", "away_fr"], suffixes=("", "_pa")))

    rows = []
    y = d["home_win"].to_numpy()
    for stat, higher_better in STATS.items():
        for kind in ("same", "prior"):
            hc, ac = f"{stat}_{kind}_home", f"{stat}_{kind}_away"
            diff = (d[hc] - d[ac]) if higher_better else (d[ac] - d[hc])
            res = eval_feature(diff.to_numpy(), y)
            res.update({"stat": stat, "kind": kind})
            rows.append(res)
    return pd.DataFrame(rows)


# ---------------------------------------------------------- B. rolling vs prior


def build_rolling(games: pd.DataFrame) -> pd.DataFrame:
    """No-look-ahead rolling RS/G, RA/G, RunDiff/G for each team entering
    each of its games (cumulative through the PRIOR game only)."""
    home = games[["gidx", "season", "home_fr", "home_score", "away_score"]].rename(
        columns={"home_fr": "team", "home_score": "rs", "away_score": "ra"})
    away = games[["gidx", "season", "away_fr", "away_score", "home_score"]].rename(
        columns={"away_fr": "team", "away_score": "rs", "home_score": "ra"})
    long = pd.concat([home, away], ignore_index=True).sort_values(
        ["team", "season", "gidx"])

    grp = long.groupby(["team", "season"])
    long["games_played"] = grp.cumcount()
    long["cum_rs"] = grp["rs"].cumsum() - long["rs"]
    long["cum_ra"] = grp["ra"].cumsum() - long["ra"]
    with np.errstate(invalid="ignore", divide="ignore"):
        long["rs_g"] = long["cum_rs"] / long["games_played"]
        long["ra_g"] = long["cum_ra"] / long["games_played"]
    long["rundiff_g"] = long["rs_g"] - long["ra_g"]
    return long


def rolling_vs_prior(games: pd.DataFrame, panel: pd.DataFrame,
                     cutoffs=(5, 10, 20, 30, 40, 50, 60, 81, 100, 130)) -> pd.DataFrame:
    g = games[games.season.isin(VALID_SEASONS)].copy()
    long = build_rolling(games)

    home_roll = long.rename(columns={
        "team": "home_fr", "games_played": "gp_home",
        "rundiff_g": "rundiff_g_roll_home"})[
        ["gidx", "home_fr", "gp_home", "rundiff_g_roll_home"]]
    away_roll = long.rename(columns={
        "team": "away_fr", "games_played": "gp_away",
        "rundiff_g": "rundiff_g_roll_away"})[
        ["gidx", "away_fr", "gp_away", "rundiff_g_roll_away"]]

    prior = panel[["Season", "franchise", "RunDiff/G"]].copy()
    prior["Season"] = prior["Season"] + 1
    prior_home = prior.rename(columns={
        "franchise": "home_fr", "RunDiff/G": "RunDiff/G_prior_home"})
    prior_away = prior.rename(columns={
        "franchise": "away_fr", "RunDiff/G": "RunDiff/G_prior_away"})

    d = (g.merge(home_roll, on=["gidx", "home_fr"])
          .merge(away_roll, on=["gidx", "away_fr"])
          .merge(prior_home, left_on=["season", "home_fr"],
                right_on=["Season", "home_fr"])
          .merge(prior_away, left_on=["season", "away_fr"],
                right_on=["Season", "away_fr"]))

    d["diff_prior"] = d["RunDiff/G_prior_home"] - d["RunDiff/G_prior_away"]
    d["diff_roll"] = d["rundiff_g_roll_home"] - d["rundiff_g_roll_away"]
    d["min_gp"] = np.minimum(d.gp_home, d.gp_away)

    rows = []
    for c in cutoffs:
        sub = d[d.min_gp >= c].dropna(subset=["diff_prior", "diff_roll"])
        y = sub["home_win"].to_numpy()
        r_prior = eval_feature(sub["diff_prior"].to_numpy(), y)
        r_roll = eval_feature(sub["diff_roll"].to_numpy(), y)
        X2 = sub[["diff_prior", "diff_roll"]].to_numpy()
        clf = LogisticRegression().fit(X2, y)
        proba = clf.predict_proba(X2)[:, 1]
        combo_acc = accuracy_score(y, (proba > 0.5).astype(int))
        combo_ll = log_loss(y, proba)
        rows.append({
            "games_played_cutoff": c, "n": len(sub),
            "r_prior": r_prior["point_biserial_r"],
            "acc_prior": r_prior["accuracy"], "ll_prior": r_prior["log_loss"],
            "r_roll": r_roll["point_biserial_r"],
            "acc_roll": r_roll["accuracy"], "ll_roll": r_roll["log_loss"],
            "acc_combo": round(float(combo_acc), 4),
            "ll_combo": round(float(combo_ll), 5),
            "combo_beta_prior": round(float(clf.coef_[0][0]), 4),
            "combo_beta_roll": round(float(clf.coef_[0][1]), 4),
        })
    return pd.DataFrame(rows)


def main():
    games = load_games()
    panel = load_panel()

    ovp = oracle_vs_prior(games, panel)
    ovp.to_csv(AN / "game_level_oracle_vs_prior.csv", index=False)
    print("=== A. same-season-final (oracle) vs prior-season-final ===")
    piv = ovp.pivot(index="stat", columns="kind",
                    values=["point_biserial_r", "accuracy", "log_loss"])
    print(piv.round(4).to_string())

    rvp = rolling_vs_prior(games, panel)
    rvp.to_csv(AN / "game_level_rolling_vs_prior.csv", index=False)
    print("\n=== B. rolling in-season RunDiff/G vs prior-season-final RunDiff/G ===")
    print(rvp.to_string(index=False))

    summary = {
        "era_same_vs_prior": {
            "same": ovp[(ovp.stat == "ERA") & (ovp.kind == "same")]
            .iloc[0][["point_biserial_r", "accuracy", "log_loss", "n"]].to_dict(),
            "prior": ovp[(ovp.stat == "ERA") & (ovp.kind == "prior")]
            .iloc[0][["point_biserial_r", "accuracy", "log_loss", "n"]].to_dict(),
        },
        "rolling_crossover": rvp.to_dict("records"),
    }
    with open(AN / "game_level_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
