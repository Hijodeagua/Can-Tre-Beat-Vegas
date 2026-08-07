"""Where does Top 100 talent actually matter — by position, by rank, at QB?

Three questions, all answered off the honors data plus the schedule:

1. **Position groups.** Split each roster's Top 100 count by position group
   (QB, OL, DL, DB, LB, WR, RB, TE) and ask which group's *advantage* over the
   opponent moves the result. Also tests rank weighting, since the list is
   ordered and a #3 player presumably outweighs a #97.
2. **QB head-to-head.** When both starting quarterbacks are on the list, does
   the higher-ranked one win more? Ordered by rank gap.
3. **Top 100 QB vs not.** The base rate everything else is measured against.

Everything is descriptive *and* market-relative: a raw win rate conflates
"good players win" (obvious) with "the market misprices good players" (the
only version that pays). Each table therefore reports the result against the
closing spread as well as straight up.

Usage
    python3 -m NFL.model.v2.top100_analysis            # tables to stdout
    python3 -m NFL.model.v2.top100_analysis --save     # + CSVs for the writeup
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
NFLVERSE_DIR = REPO_ROOT / "data" / "rosters" / "nflverse"
TOP100_CSV = REPO_ROOT / "data" / "rosters" / "awards" / "top100.csv"
GAMES_PATH = REPO_ROOT / "data" / "schedules" / "nflverse_games.csv"
OUT_DIR = REPO_ROOT / "NFL" / "model" / "v2" / "artifacts" / "top100"

FIRST_SEASON = 2011  # the list does not exist before this
POSITION_GROUPS = ["QB", "OL", "DL", "DB", "LB", "WR", "RB", "TE"]

# Rank weighting schemes. The list is ordered, so a #3 should not count the
# same as a #97 — but how much more is an open question, hence three curves.
WEIGHTS = {
    "flat": lambda r: np.ones_like(r, dtype=float),
    "linear": lambda r: (101.0 - r) / 100.0,
    "log": lambda r: 1.0 / np.log2(r + 1.0),
}


def load_top100_with_positions() -> pd.DataFrame:
    t = pd.read_csv(TOP100_CSV)
    p = pd.read_csv(NFLVERSE_DIR / "players.csv", low_memory=False,
                    usecols=["pfr_id", "gsis_id", "position", "position_group",
                             "display_name"])
    m = t.merge(p.drop_duplicates("pfr_id"), on="pfr_id", how="left")
    m["position_group"] = m["position_group"].where(
        m["position_group"].isin(POSITION_GROUPS), "OTHER")
    for name, fn in WEIGHTS.items():
        m[f"w_{name}"] = fn(m["rank"].to_numpy())
    return m


def load_active_rosters(seasons) -> pd.DataFrame:
    frames = []
    for s in seasons:
        path = NFLVERSE_DIR / f"roster_weekly_{s}.csv"
        if not path.exists():
            continue
        r = pd.read_csv(path, low_memory=False,
                        usecols=lambda c: c in {"season", "team", "week",
                                                "status", "gsis_id"})
        frames.append(r[r["status"].astype(str).str.upper() == "ACT"])
    out = pd.concat(frames, ignore_index=True)
    from .squad import _norm_team
    out["team"] = _norm_team(out["team"])
    return out.dropna(subset=["gsis_id"])


def team_week_by_position(top100: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Top 100 count and weighted score per team-week, per position group."""
    t = top100.dropna(subset=["gsis_id"])
    j = rosters.merge(t[["season", "gsis_id", "rank", "position_group",
                         *[f"w_{k}" for k in WEIGHTS]]],
                      on=["season", "gsis_id"], how="inner")
    counts = (j.pivot_table(index=["season", "team", "week"],
                            columns="position_group", values="rank",
                            aggfunc="size", fill_value=0)
                .reset_index())
    counts.columns.name = None
    for g in POSITION_GROUPS:
        if g not in counts.columns:
            counts[g] = 0
        counts = counts.rename(columns={g: f"t100_{g}"})

    for name in WEIGHTS:
        w = (j.groupby(["season", "team", "week"])[f"w_{name}"].sum()
              .rename(f"t100_score_{name}").reset_index())
        counts = counts.merge(w, on=["season", "team", "week"], how="left")

    counts["t100_total"] = counts[[f"t100_{g}" for g in POSITION_GROUPS]].sum(axis=1)
    return counts.fillna(0)


def attach_to_games(games: pd.DataFrame, tw: pd.DataFrame) -> pd.DataFrame:
    g = games[games["season"] >= FIRST_SEASON].dropna(subset=["home_score"]).copy()
    g["margin"] = g["home_score"] - g["away_score"]
    g["home_win"] = (g["margin"] > 0).astype(int)
    ats = g["margin"] - g["spread_line"]
    g["home_cover"] = np.where(ats > 0, 1.0, np.where(ats < 0, 0.0, np.nan))

    cols = [c for c in tw.columns if c.startswith("t100")]
    for side in ("home", "away"):
        s = tw.rename(columns={"team": f"{side}_team",
                               **{c: f"{side}_{c}" for c in cols}})
        g = g.merge(s[["season", f"{side}_team", "week"] +
                      [f"{side}_{c}" for c in cols]],
                    on=["season", f"{side}_team", "week"], how="left")
    for c in cols:
        g[f"{c}_diff"] = g[f"home_{c}"] - g[f"away_{c}"]
    return g


# --------------------------------------------------------------------------
# 1. position groups
# --------------------------------------------------------------------------

def _rate_block(d: pd.DataFrame, label: str) -> dict:
    cov = d["home_cover"].dropna()
    return {
        "bucket": label,
        "games": len(d),
        "home_win": round(float(d["home_win"].mean()), 3) if len(d) else np.nan,
        "home_cover": round(float(cov.mean()), 3) if len(cov) else np.nan,
        "avg_margin": round(float(d["margin"].mean()), 2) if len(d) else np.nan,
        "avg_spread": round(float(d["spread_line"].mean()), 2) if len(d) else np.nan,
    }


def position_group_impact(g: pd.DataFrame) -> pd.DataFrame:
    """For each position group: results when the home team has more, fewer, or
    the same number of Top 100 players at that group."""
    rows = []
    for grp in POSITION_GROUPS:
        col = f"t100_{grp}_diff"
        d = g.dropna(subset=[col])
        adv, dis = d[d[col] > 0], d[d[col] < 0]
        if len(adv) < 30 or len(dis) < 30:
            continue
        a, b = _rate_block(adv, "advantage"), _rate_block(dis, "deficit")
        rows.append({
            "position_group": grp,
            "games_adv": a["games"], "games_def": b["games"],
            "win_adv": a["home_win"], "win_def": b["home_win"],
            "win_gap": round(a["home_win"] - b["home_win"], 3),
            "cover_adv": a["home_cover"], "cover_def": b["home_cover"],
            # The market-relative number: does the edge survive the spread?
            "cover_gap": round(a["home_cover"] - b["home_cover"], 3),
            "spread_adv": a["avg_spread"], "spread_def": b["avg_spread"],
        })
    return pd.DataFrame(rows).sort_values("cover_gap", ascending=False)


def rank_weight_comparison(g: pd.DataFrame) -> pd.DataFrame:
    """Do rank-weighted talent scores separate outcomes better than raw counts?"""
    rows = []
    for name in ["count", *WEIGHTS]:
        col = "t100_total_diff" if name == "count" else f"t100_score_{name}_diff"
        d = g.dropna(subset=[col])
        if not len(d):
            continue
        # Correlation with margin, and with margin-against-the-spread.
        ats = d["margin"] - d["spread_line"]
        rows.append({
            "weighting": name,
            "games": len(d),
            "corr_with_margin": round(float(np.corrcoef(d[col], d["margin"])[0, 1]), 4),
            "corr_with_ats_margin": round(float(np.corrcoef(d[col], ats)[0, 1]), 4),
            "corr_with_spread": round(float(np.corrcoef(d[col], d["spread_line"])[0, 1]), 4),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2 & 3. quarterbacks
# --------------------------------------------------------------------------

def qb_table(games: pd.DataFrame, top100: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game with the starter's Top 100 rank (NaN if unlisted)."""
    t = top100[top100["position"] == "QB"][["season", "display_name", "rank"]]
    t = t.dropna(subset=["display_name"]).drop_duplicates(["season", "display_name"])

    rows = []
    for side in ("home", "away"):
        d = games[["game_id", "season", "week", f"{side}_team", f"{side}_qb_name"]].copy()
        d.columns = ["game_id", "season", "week", "team", "qb"]
        d["side"] = side
        rows.append(d)
    qb = pd.concat(rows, ignore_index=True)
    qb = qb.merge(t.rename(columns={"display_name": "qb", "rank": "qb_rank"}),
                  on=["season", "qb"], how="left")
    return qb


def qb_matchup_frame(g: pd.DataFrame, qb: pd.DataFrame) -> pd.DataFrame:
    h = qb[qb["side"] == "home"][["game_id", "qb", "qb_rank"]].rename(
        columns={"qb": "home_qb", "qb_rank": "home_qb_rank"})
    a = qb[qb["side"] == "away"][["game_id", "qb", "qb_rank"]].rename(
        columns={"qb": "away_qb", "qb_rank": "away_qb_rank"})
    m = g.merge(h, on="game_id", how="left").merge(a, on="game_id", how="left")
    m["home_listed"] = m["home_qb_rank"].notna()
    m["away_listed"] = m["away_qb_rank"].notna()
    return m


def qb_listed_breakdown(m: pd.DataFrame) -> pd.DataFrame:
    """Top 100 QB vs non-Top 100 QB — the headline split."""
    rows = []
    scenarios = {
        "both listed": m["home_listed"] & m["away_listed"],
        "home only": m["home_listed"] & ~m["away_listed"],
        "away only": ~m["home_listed"] & m["away_listed"],
        "neither": ~m["home_listed"] & ~m["away_listed"],
    }
    for label, mask in scenarios.items():
        rows.append(_rate_block(m[mask], label))
    return pd.DataFrame(rows)


def qb_listed_team_perspective(m: pd.DataFrame) -> pd.DataFrame:
    """Same question from the listed QB's own side, home/away folded together."""
    edge = m[m["home_listed"] != m["away_listed"]].copy()
    listed_is_home = edge["home_listed"]
    su = np.where(listed_is_home, edge["home_win"] == 1, edge["home_win"] == 0)
    ats = edge["margin"] - edge["spread_line"]
    cover = np.where(listed_is_home, ats, -ats)
    spread_for = np.where(listed_is_home, edge["spread_line"], -edge["spread_line"])
    valid = cover != 0
    return pd.DataFrame([{
        "scenario": "Top 100 QB vs unlisted QB",
        "games": len(edge),
        "listed_qb_SU": round(float(su.mean()), 3),
        "listed_qb_ATS": round(float((cover[valid] > 0).mean()), 3),
        "avg_spread_for_listed": round(float(spread_for.mean()), 2),
        "avg_margin_for_listed": round(
            float(np.where(listed_is_home, edge["margin"], -edge["margin"]).mean()), 2),
    }])


def qb_head_to_head(m: pd.DataFrame, bins=(0, 10, 25, 50, 100)) -> pd.DataFrame:
    """Both QBs listed: does the better-ranked one win? Bucketed by rank gap."""
    both = m[m["home_listed"] & m["away_listed"]].copy()
    if both.empty:
        return pd.DataFrame()
    # Rank 1 is best, so the *lower* rank number is the better player.
    both["gap"] = (both["home_qb_rank"] - both["away_qb_rank"]).abs()
    higher_is_home = both["home_qb_rank"] < both["away_qb_rank"]
    both["higher_won"] = np.where(higher_is_home, both["home_win"] == 1,
                                  both["home_win"] == 0)
    ats = both["margin"] - both["spread_line"]
    both["higher_cover"] = np.where(higher_is_home, ats, -ats)
    both["spread_for_higher"] = np.where(higher_is_home, both["spread_line"],
                                         -both["spread_line"])

    rows = [{
        "rank_gap": "all",
        "games": len(both),
        "higher_ranked_SU": round(float(both["higher_won"].mean()), 3),
        "higher_ranked_ATS": round(float((both.loc[both["higher_cover"] != 0,
                                                   "higher_cover"] > 0).mean()), 3),
        "avg_spread_for_higher": round(float(both["spread_for_higher"].mean()), 2),
    }]
    for lo, hi in zip(bins[:-1], bins[1:]):
        d = both[(both["gap"] >= lo) & (both["gap"] < hi)]
        if len(d) < 20:
            continue
        cov = d.loc[d["higher_cover"] != 0, "higher_cover"]
        rows.append({
            "rank_gap": f"{lo}-{hi}",
            "games": len(d),
            "higher_ranked_SU": round(float(d["higher_won"].mean()), 3),
            "higher_ranked_ATS": round(float((cov > 0).mean()), 3),
            "avg_spread_for_higher": round(float(d["spread_for_higher"].mean()), 2),
        })
    return pd.DataFrame(rows)


def qb_by_rank_tier(m: pd.DataFrame) -> pd.DataFrame:
    """Do top-10 QBs outperform 90th-ranked ones, against the number?"""
    rows = []
    for side in ("home", "away"):
        d = m[m[f"{side}_listed"]].copy()
        ats = d["margin"] - d["spread_line"]
        rows.append(pd.DataFrame({
            "rank": d[f"{side}_qb_rank"],
            "su": (d["home_win"] == 1) if side == "home" else (d["home_win"] == 0),
            "ats": ats if side == "home" else -ats,
            "spread": d["spread_line"] if side == "home" else -d["spread_line"],
        }))
    q = pd.concat(rows, ignore_index=True)
    q["tier"] = pd.cut(q["rank"], [0, 10, 25, 50, 100],
                       labels=["1-10", "11-25", "26-50", "51-100"])
    out = q.groupby("tier", observed=True).apply(lambda d: pd.Series({
        "games": len(d),
        "SU": round(float(d["su"].mean()), 3),
        "ATS": round(float((d.loc[d["ats"] != 0, "ats"] > 0).mean()), 3),
        "avg_spread": round(float(d["spread"].mean()), 2),
        "avg_margin_vs_spread": round(float(d["ats"].mean()), 2),
    }), include_groups=False).reset_index()
    return out


def build_all() -> dict[str, pd.DataFrame]:
    games = pd.read_csv(GAMES_PATH)
    games["gameday"] = pd.to_datetime(games["gameday"])
    seasons = sorted(s for s in games["season"].unique() if s >= FIRST_SEASON)

    top100 = load_top100_with_positions()
    rosters = load_active_rosters(seasons)
    tw = team_week_by_position(top100, rosters)
    g = attach_to_games(games, tw)

    qb = qb_table(g, top100)
    m = qb_matchup_frame(g, qb)

    return {
        "position_impact": position_group_impact(g),
        "rank_weighting": rank_weight_comparison(g),
        "qb_listed_breakdown": qb_listed_breakdown(m),
        "qb_listed_team": qb_listed_team_perspective(m),
        "qb_head_to_head": qb_head_to_head(m),
        "qb_by_tier": qb_by_rank_tier(m),
        "_games": g,
        "_matchups": m,
        "_top100": top100,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    res = build_all()
    for name, df in res.items():
        if name.startswith("_"):
            continue
        print(f"\n=== {name} ===")
        print(df.to_string(index=False))

    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, df in res.items():
            if not name.startswith("_"):
                df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        res["_matchups"].to_csv(OUT_DIR / "qb_matchups.csv", index=False)
        print(f"\nsaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
