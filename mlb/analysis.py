"""
Association testing: Baseball-Reference team season stats (2009-2026) vs
game outcomes and the betting-blind Elo model.

Produces CSV tables + summary.json under data/mlb/analysis/ that the report
generator consumes.

Era handling: every stat and outcome is z-scored WITHIN season (across the
30 teams) before pooling, so changing run environments (2019 juiced ball,
2020 short season, partial 2026) don't manufacture spurious correlations.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "mlb"
OUT = DATA / "analysis"
OUT.mkdir(exist_ok=True)

TO_FRANCHISE = {"FLA": "MIA", "OAK": "ATH"}

# ---------------------------------------------------------------- load


def load_merged() -> pd.DataFrame:
    hit = pd.read_csv(DATA / "team_hitting_2009_2026.csv")
    pit = pd.read_csv(DATA / "team_pitching_2009_2026.csv")
    adv = pd.read_csv(DATA / "team_pitching_advanced_2009_2026.csv")
    elo = pd.read_csv(DATA / "elo_team_seasons.csv")

    for df in (hit, pit, adv):
        df["franchise"] = df["Team"].map(lambda t: TO_FRANCHISE.get(t, t))

    key = ["Season", "franchise"]
    m = hit.merge(
        pit, on=key, suffixes=("", "_pit"), validate="1:1"
    ).merge(adv[key + ["PtchR", "PtchW", "WPA", "WPA/LI", "Clutch",
                       "RE24", "REW", "aLI"]],
            on=key, validate="1:1")
    m = m.merge(
        elo.rename(columns={"season": "Season"}), on=key, validate="1:1"
    )

    g = m["GP"].astype(float)
    # per-game normalizations for counting stats
    m["BtRuns/G"] = m["BtRuns"] / g          # batting runs above avg
    m["PtchR/G"] = m["PtchR"] / g            # pitching runs above avg
    m["RC/G"] = m["RC"] / g
    m["RA/G"] = m["R_pit"] / g               # runs allowed per game
    m["RunDiff/G"] = m["R/Gm"] - m["RA/G"]
    m["BB%"] = m["BB"] / m["PA"]
    m["SO%"] = m["SO"] / m["PA"]
    m["HR%"] = m["HR"] / m["PA"]
    m["XBH%"] = m["XBH"] / m["PA"]
    m["SV/G"] = m["SV"] / g
    m["WPA/G"] = m["WPA"] / g
    m["Clutch/G"] = m["Clutch"] / g
    m["RE24/G"] = m["RE24"] / g
    m["wl"] = m["WL%"]
    return m


HIT_STATS = ["BA", "OBP", "SLG", "OPS", "OPS+", "ISO", "R/Gm", "BAbip",
             "BB%", "SO%", "HR%", "XBH%", "SB%", "SBatt", "AB/SO",
             "BtRuns/G", "RC/G", "Hitting AIR"]
PIT_STATS = ["ERA", "ERA+", "FIP", "WHIP", "H9", "HR9", "BB9", "SO9",
             "SO/BB", "RA/G", "SV/G", "PtchR/G"]
ADV_STATS = ["WPA/G", "WPA/LI", "Clutch/G", "RE24/G", "REW", "aLI"]
COMBO = ["RunDiff/G"]
ALL_STATS = HIT_STATS + PIT_STATS + ADV_STATS + COMBO

GROUP = {**{s: "hitting" for s in HIT_STATS},
         **{s: "pitching" for s in PIT_STATS},
         **{s: "advanced" for s in ADV_STATS},
         **{s: "combined" for s in COMBO}}

OUTCOMES = ["wl", "elo_delta", "elo_end"]


def zscore_within_season(df: pd.DataFrame, cols) -> pd.DataFrame:
    z = df.copy()
    for c in cols:
        grp = z.groupby("Season")[c]
        z[c] = (z[c] - grp.transform("mean")) / grp.transform("std")
    return z


# ---------------------------------------------------------------- tests


def corr_table(z: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in ALL_STATS:
        for o in OUTCOMES:
            sub = z[[s, o]].dropna()
            pr, pp = sps.pearsonr(sub[s], sub[o])
            sr, sp = sps.spearmanr(sub[s], sub[o])
            rows.append({"stat": s, "group": GROUP[s], "outcome": o,
                         "pearson_r": round(pr, 4), "pearson_p": pp,
                         "spearman_r": round(sr, 4), "spearman_p": sp,
                         "n": len(sub)})
    return pd.DataFrame(rows)


def ols(zdf: pd.DataFrame, y: str, xs: list[str]) -> dict:
    sub = zdf[[y] + xs].dropna()
    X = np.column_stack([np.ones(len(sub))] + [sub[x] for x in xs])
    Y = sub[y].to_numpy()
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    dof = len(sub) - X.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    tvals = beta / se
    pvals = 2 * sps.t.sf(np.abs(tvals), dof)
    r2 = 1 - (resid @ resid) / ((Y - Y.mean()) @ (Y - Y.mean()))
    return {
        "y": y, "n": len(sub), "r2": round(float(r2), 4),
        "terms": [
            {"x": name, "beta": round(float(b), 4),
             "t": round(float(t), 2), "p": float(p)}
            for name, b, t, p in zip(
                ["intercept"] + xs, beta, tvals, pvals)
        ],
    }


def yoy_table(m: pd.DataFrame) -> pd.DataFrame:
    """Correlate year-over-year stat changes with Elo / WL% changes.
    Skips any pair involving 2020 (60-game season). 2025->2026 kept
    (rate stats only) but the partial-season caveat applies."""
    m = m.sort_values(["franchise", "Season"])
    rate_stats = [s for s in ALL_STATS if s not in ("SBatt",)]
    rows = []
    d = m.copy()
    for c in rate_stats + ["wl", "elo_end"]:
        d[f"d_{c}"] = d.groupby("franchise")[c].diff()
    d["prev_season"] = d.groupby("franchise")["Season"].shift()
    d = d[(d.prev_season == d.Season - 1)
          & (d.Season != 2020) & (d.prev_season != 2020)]
    for s in rate_stats:
        sub = d[[f"d_{s}", "d_wl", "d_elo_end"]].dropna()
        r_wl, p_wl = sps.pearsonr(sub[f"d_{s}"], sub["d_wl"])
        r_elo, p_elo = sps.pearsonr(sub[f"d_{s}"], sub["d_elo_end"])
        rows.append({"stat": s, "group": GROUP[s],
                     "r_delta_wl": round(r_wl, 4), "p_delta_wl": p_wl,
                     "r_delta_elo": round(r_elo, 4), "p_delta_elo": p_elo,
                     "n": len(sub)})
    return pd.DataFrame(rows)


def persistence_table(z: pd.DataFrame) -> pd.DataFrame:
    """Does stat_t predict next season's record beyond end-of-season Elo?
    r_next: raw corr(stat_t, wl_{t+1}); r_next_given_elo: partial corr
    controlling for elo_end_t (both residualized)."""
    z = z.sort_values(["franchise", "Season"]).copy()
    z["wl_next"] = z.groupby("franchise")["wl"].shift(-1)
    z["next_season"] = z.groupby("franchise")["Season"].shift(-1)
    z = z[(z.next_season == z.Season + 1)
          & (z.Season != 2020) & (z.next_season != 2020)]

    def partial(x, y, c):
        rx = x - np.polyval(np.polyfit(c, x, 1), c)
        ry = y - np.polyval(np.polyfit(c, y, 1), c)
        return sps.pearsonr(rx, ry)

    rows = []
    for s in ALL_STATS + ["elo_end"]:
        cols = [s, "wl_next"] + (["elo_end"] if s != "elo_end" else [])
        sub = z[cols].dropna()
        r, p = sps.pearsonr(sub[s], sub["wl_next"])
        if s == "elo_end":
            rp, pp_ = np.nan, np.nan
        else:
            rp, pp_ = partial(sub[s].to_numpy(),
                              sub["wl_next"].to_numpy(),
                              sub["elo_end"].to_numpy())
        rows.append({"stat": s, "group": GROUP.get(s, "elo"),
                     "r_next_wl": round(float(r), 4), "p_next_wl": p,
                     "r_next_wl_given_elo":
                         None if s == "elo_end" else round(float(rp), 4),
                     "p_next_wl_given_elo":
                         None if s == "elo_end" else float(pp_),
                     "n": len(sub)})
    return pd.DataFrame(rows)


def calibration_table() -> pd.DataFrame:
    hist = pd.read_csv(DATA / "elo_game_history.csv")
    ev = hist[hist.season >= 2012].copy()
    ev["bin"] = pd.cut(ev.p_home, np.arange(0.3, 0.775, 0.025))
    g = ev.groupby("bin", observed=True).agg(
        p_pred=("p_home", "mean"),
        p_actual=("home_win", "mean"),
        n=("home_win", "size"),
    ).reset_index(drop=True)
    return g


# ---------------------------------------------------------------- main


def main():
    m = load_merged()
    print(f"merged panel: {len(m)} team-seasons, "
          f"{m.Season.nunique()} seasons")

    z = zscore_within_season(m, ALL_STATS + OUTCOMES)

    ct = corr_table(z)
    ct.to_csv(OUT / "correlations.csv", index=False)

    models = [
        ols(z, "wl", ["OPS+", "ERA+"]),
        ols(z, "wl", ["OPS+", "ERA+", "SV/G", "SB%", "BAbip"]),
        ols(z, "wl", ["RunDiff/G"]),
        ols(z, "elo_delta", ["OPS+", "ERA+"]),
        ols(z, "elo_delta", ["RunDiff/G"]),
        ols(z, "wl", ["OBP", "SLG", "WHIP", "SO9", "HR9"]),
    ]
    with open(OUT / "ols_models.json", "w") as fh:
        json.dump(models, fh, indent=2)

    yt = yoy_table(m)
    yt.to_csv(OUT / "yoy_deltas.csv", index=False)

    pt = persistence_table(z)
    pt.to_csv(OUT / "persistence.csv", index=False)

    cal = calibration_table()
    cal.to_csv(OUT / "calibration.csv", index=False)

    # convenience: merged panel for charts
    m.to_csv(OUT / "merged_panel.csv", index=False)

    with open(DATA / "elo_params.json") as fh:
        params = json.load(fh)["best"]

    summary = {
        "n_team_seasons": len(m),
        "elo_params": params,
        "top_wl_correlates": ct[ct.outcome == "wl"]
        .reindex(ct[ct.outcome == "wl"].pearson_r.abs()
                 .sort_values(ascending=False).index)
        .head(12)[["stat", "group", "pearson_r"]]
        .to_dict("records"),
        "top_elo_delta_correlates": ct[ct.outcome == "elo_delta"]
        .reindex(ct[ct.outcome == "elo_delta"].pearson_r.abs()
                 .sort_values(ascending=False).index)
        .head(12)[["stat", "group", "pearson_r"]]
        .to_dict("records"),
    }
    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2)[:2000])


if __name__ == "__main__":
    main()
