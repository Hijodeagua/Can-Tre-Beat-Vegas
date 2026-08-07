"""
Extended association analysis:
  1. Additional stats not covered in the first pass (RBI, TB, GIDP, HBP,
     SH, SF, IBB, LOB, CS%, CG, SHO, BK, WP, BF, PtchW, plus per-game rates).
  2. Prior-season effects: does season t-1's stat LEVEL predict season t's
     outcome, and does it add anything beyond elo_start_t (the Elo a team
     actually enters the season with, i.e. last year's rating after
     carryover)? This is the "does the past season matter" question, asked
     directly rather than via deltas.
  3. Rank vs. value: is a team's within-season ORDINAL rank on a stat a
     better predictor than the stat's raw magnitude? Pearson-on-ranks
     (=Spearman) vs Pearson-on-values, plus a quartile-bucket linearity
     check for the three headline stats.

Outputs land in data/mlb/analysis/ alongside the first-pass tables.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO = Path(__file__).resolve().parent.parent
AN = REPO / "data" / "mlb" / "analysis"

OUTCOMES = ["wl", "elo_delta", "elo_end"]

# ---------------------------------------------------------------- load


def load_panel() -> pd.DataFrame:
    m = pd.read_csv(AN / "merged_panel.csv")
    g = m["GP"].astype(float)

    m["RBI/G"] = m["RBI"] / g
    m["TB/G"] = m["TB"] / g
    m["GIDP/G"] = m["GIDP"] / g
    m["HBP%"] = m["HBP"] / m["PA"]
    m["SH/G"] = m["SH"] / g
    m["SF/G"] = m["SF"] / g
    m["IBB_bat%"] = m["IBB"] / m["PA"]
    m["LOB/G"] = m["LOB"] / g
    m["CS%"] = m["CS"] / (m["SB"] + m["CS"]).replace(0, np.nan)
    m["CG/G"] = m["CG"] / g
    m["SHO/G"] = m["SHO"] / g
    m["BK/G"] = m["BK"] / g
    m["WP/G"] = m["WP"] / g
    m["BF/G"] = m["BF"] / g
    m["PtchW/G"] = m["PtchW"] / g
    return m


EXTRA_STATS = ["RBI/G", "TB/G", "GIDP/G", "HBP%", "SH/G", "SF/G",
                "IBB_bat%", "LOB/G", "CS%", "CG/G", "SHO/G", "BK/G",
                "WP/G", "BF/G", "PtchW/G"]
EXTRA_GROUP = {
    "RBI/G": "hitting", "TB/G": "hitting", "GIDP/G": "hitting",
    "HBP%": "hitting", "SH/G": "hitting", "SF/G": "hitting",
    "IBB_bat%": "hitting", "LOB/G": "hitting", "CS%": "hitting",
    "CG/G": "pitching", "SHO/G": "pitching", "BK/G": "pitching",
    "WP/G": "pitching", "BF/G": "pitching", "PtchW/G": "advanced",
}

# core stats carried over from the first pass, used for the rank-vs-value
# and prior-season tests
CORE_STATS = ["BA", "OBP", "SLG", "OPS", "OPS+", "ISO", "R/Gm", "BAbip",
              "BB%", "SO%", "HR%", "XBH%", "SB%", "AB/SO", "BtRuns/G",
              "RC/G", "ERA", "ERA+", "FIP", "WHIP", "H9", "HR9", "BB9",
              "SO9", "SO/BB", "RA/G", "SV/G", "PtchR/G", "RunDiff/G"]


def zscore_within_season(df: pd.DataFrame, cols) -> pd.DataFrame:
    z = df.copy()
    for c in cols:
        grp = z.groupby("Season")[c]
        z[c] = (z[c] - grp.transform("mean")) / grp.transform("std")
    return z


# ---------------------------------------------------------------- 1. extra


def extra_stats_table(m: pd.DataFrame) -> pd.DataFrame:
    z = zscore_within_season(m, EXTRA_STATS + OUTCOMES)
    rows = []
    for s in EXTRA_STATS:
        for o in OUTCOMES:
            sub = z[[s, o]].dropna()
            pr, pp = sps.pearsonr(sub[s], sub[o])
            sr, sp = sps.spearmanr(sub[s], sub[o])
            rows.append({"stat": s, "group": EXTRA_GROUP[s], "outcome": o,
                         "pearson_r": round(pr, 4), "pearson_p": pp,
                         "spearman_r": round(sr, 4), "spearman_p": sp,
                         "n": len(sub)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. lag


def prior_season_table(m: pd.DataFrame) -> pd.DataFrame:
    """For each stat, does its value in season t-1 predict outcome in
    season t, and does it survive controlling for elo_start_t (the Elo
    the team actually carries into season t)?"""
    all_stats = CORE_STATS + EXTRA_STATS
    z = zscore_within_season(m, all_stats + ["elo_start"])
    z = z.sort_values(["franchise", "Season"])

    prev = z[["franchise", "Season"] + all_stats].copy()
    prev["Season"] = prev["Season"] + 1
    prev = prev.rename(columns={s: f"prev_{s}" for s in all_stats})

    cur = z[["franchise", "Season", "wl", "elo_end", "elo_delta",
              "elo_start"]]
    d = cur.merge(prev, on=["franchise", "Season"], validate="1:1")
    # drop pairs that cross the 2020 season either direction
    prev_season_col = z[["franchise", "Season"]].copy()
    prev_season_col["had_prev"] = True
    valid_prev = set(zip(m.franchise, m.Season + 1))
    valid_prev_season = set(m.Season)
    d = d[d.apply(lambda r: (r.Season - 1) in valid_prev_season, axis=1)]
    d = d[(d.Season != 2021) & (d.Season != 2020)]  # 2020 on either side

    def partial(x, y, c):
        rx = x - np.polyval(np.polyfit(c, x, 1), c)
        ry = y - np.polyval(np.polyfit(c, y, 1), c)
        return sps.pearsonr(rx, ry)

    rows = []
    for s in all_stats:
        col = f"prev_{s}"
        sub = d[[col, "wl", "elo_start"]].dropna()
        r, p = sps.pearsonr(sub[col], sub["wl"])
        rp, pp_ = partial(sub[col].to_numpy(), sub["wl"].to_numpy(),
                          sub["elo_start"].to_numpy())
        grp = EXTRA_GROUP.get(s, "hitting" if s in (
            "BA", "OBP", "SLG", "OPS", "OPS+", "ISO", "R/Gm", "BAbip",
            "BB%", "SO%", "HR%", "XBH%", "SB%", "AB/SO", "BtRuns/G",
            "RC/G") else "pitching")
        if s == "RunDiff/G":
            grp = "combined"
        if s == "PtchR/G":
            grp = "pitching"
        rows.append({"stat": s, "group": grp,
                     "r_prevstat_vs_curwl": round(float(r), 4),
                     "p_prevstat_vs_curwl": p,
                     "r_given_elo_start": round(float(rp), 4),
                     "p_given_elo_start": float(pp_),
                     "n": len(sub)})
    out = pd.DataFrame(rows)

    # benchmark: elo_start alone vs current wl
    sub = d[["elo_start", "wl"]].dropna()
    r, p = sps.pearsonr(sub["elo_start"], sub["wl"])
    bench = pd.DataFrame([{"stat": "elo_start", "group": "elo",
                           "r_prevstat_vs_curwl": round(float(r), 4),
                           "p_prevstat_vs_curwl": p,
                           "r_given_elo_start": None,
                           "p_given_elo_start": None, "n": len(sub)}])
    return pd.concat([bench, out], ignore_index=True)


# ---------------------------------------------------------------- 3. rank


def rank_vs_value_table(m: pd.DataFrame) -> pd.DataFrame:
    stats = CORE_STATS + EXTRA_STATS
    z = zscore_within_season(m, stats + OUTCOMES)
    rows = []
    for s in stats:
        for o in ["wl", "elo_delta"]:
            sub = z[[s, o]].dropna()
            pr, _ = sps.pearsonr(sub[s], sub[o])
            sr, _ = sps.spearmanr(sub[s], sub[o])  # = pearson on ranks
            rows.append({"stat": s, "outcome": o,
                         "pearson_value_r": round(pr, 4),
                         "pearson_rank_r": round(sr, 4),
                         "gap": round(sr - pr, 4)})
    return pd.DataFrame(rows)


def quartile_buckets(m: pd.DataFrame) -> pd.DataFrame:
    """Average win% and elo_delta by within-season quartile, for the three
    headline stats -- a visual/tabular linearity check."""
    m = m.copy()
    rows = []
    for s in ["RunDiff/G", "OPS+", "ERA+"]:
        m[f"{s}_q"] = m.groupby("Season")[s].transform(
            lambda x: pd.qcut(x, 4, labels=[1, 2, 3, 4]))
        g = m.groupby(f"{s}_q", observed=True).agg(
            wl=("wl", "mean"), elo_delta=("elo_delta", "mean"),
            n=("wl", "size")).reset_index()
        g.insert(0, "stat", s)
        g = g.rename(columns={f"{s}_q": "quartile"})
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- main


def main():
    m = load_panel()

    ex = extra_stats_table(m)
    ex.to_csv(AN / "extra_stat_correlations.csv", index=False)

    ps = prior_season_table(m)
    ps.to_csv(AN / "prior_season.csv", index=False)

    rv = rank_vs_value_table(m)
    rv.to_csv(AN / "rank_vs_value.csv", index=False)

    qb = quartile_buckets(m)
    qb.to_csv(AN / "quartile_buckets.csv", index=False)

    print("=== extra stats: top |r| with wl ===")
    e_wl = ex[ex.outcome == "wl"].reindex(
        ex[ex.outcome == "wl"].pearson_r.abs().sort_values(
            ascending=False).index)
    print(e_wl[["stat", "group", "pearson_r", "pearson_p"]]
          .to_string(index=False))

    print("\n=== prior season stat -> current wl, raw vs given elo_start ===")
    print(ps.reindex(ps.r_prevstat_vs_curwl.abs()
                     .sort_values(ascending=False).index)
          .head(15).to_string(index=False))

    print("\n=== rank vs value gap (|gap| sorted) ===")
    rv_wl = rv[rv.outcome == "wl"].copy()
    rv_wl["absgap"] = rv_wl.gap.abs()
    print(rv_wl.sort_values("absgap", ascending=False).head(12)
          .drop(columns="absgap").to_string(index=False))

    print("\n=== quartile buckets ===")
    print(qb.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
