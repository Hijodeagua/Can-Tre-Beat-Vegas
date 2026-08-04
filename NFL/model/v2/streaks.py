"""Hot streaks, cold streaks, and QB bounce-backs — does the market misprice form?

The question that matters is **not** "do hot teams win more". They do, and the
spread already knows it: a team that has scored 35 a game will be laying more
points next week. Asking whether they win is asking whether the line is roughly
right, and it is.

The tradable question is whether the market **over- or under-adjusts**. So every
test here is against the spread, on the team that carries the streak, and the
null is 50% (or 52.38% once you pay the vig). If bettors chase recent scoring
the way the behavioural literature says they do, hot teams should be *overpriced*
and their cover rate should sit below 50%.

## The trap this module is built to avoid

Testing 15 form hypotheses at the 5% level produces a 5%-significant result about
half the time by construction. Everything below therefore reports:

- **block bootstrap** CIs resampling whole season-weeks, not games
- a **season-level t**, because a strategy is deployed once a year (established
  in `early_season.py`, where a game-level p of 0.009 became a season-level 0.058)
- **Benjamini-Hochberg FDR** and Bonferroni across the whole family of tests

A result that survives only the raw p-value is reported as "not surviving
correction" rather than as a finding.

## QB bounce-back: two different questions

1. *Does the QB play better after a bad game?* Almost certainly yes, and it is
   statistically uninteresting — a bad game is partly bad luck, and luck does not
   repeat. This is regression to the mean and shows up in `qb_bounce_back`'s
   `next_epa_per_att` column.
2. *Does the market price that reversion correctly?* That is the ATS column, and
   it is the only one worth money.

Reporting (1) as though it were (2) is the classic form-betting error.

Usage
    python3 -m NFL.model.v2.streaks
    python3 -m NFL.model.v2.streaks --save
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .dataset import load_games

ARTIFACTS = Path(__file__).resolve().parent / "artifacts" / "streaks"
ROSTER_DIR = Path("data/rosters/nflverse")

BREAK_EVEN = 110 / 210
MIN_GAMES = 80          # below this a cover rate is unreadable
SEED = 17

# nflverse's schedule keeps the franchise's contemporaneous code while the
# player-stats feed uses the current one. Joining them raw silently drops every
# pre-relocation Raiders, Chargers and Rams game — ~5% of team-games, and not a
# random 5%. Normalise both sides to the modern code.
# The Rams are the awkward one: the schedule says STL then LAR, the stats feed
# says LA throughout. Collapse all three spellings onto one code.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}


def normalize_team(s: pd.Series) -> pd.Series:
    return s.replace(TEAM_ALIASES)


# --------------------------------------------------------------------------
# team-game spine
# --------------------------------------------------------------------------

def team_games(seasons_from: int = 2006) -> pd.DataFrame:
    """One row per team per game, with the ATS result and prior-game history.

    Every streak column is built from games strictly *before* the row's own
    game, so nothing here can see its own result.
    """
    g = load_games()
    g = g[(g["season"] >= seasons_from) & g["home_score"].notna()
          & g["spread_line"].notna()].copy()
    if "game_type" in g:
        g = g[g["game_type"] == "REG"]

    rows = []
    for side in ("home", "away"):
        opp = "away" if side == "home" else "home"
        d = pd.DataFrame({
            "game_id": g["game_id"], "season": g["season"], "week": g["week"],
            "gameday": pd.to_datetime(g["gameday"]),
            "team": g[f"{side}_team"], "opp": g[f"{opp}_team"],
            "pf": g[f"{side}_score"], "pa": g[f"{opp}_score"],
            "is_home": side == "home",
        })
        # Team-oriented spread: negative means this team is favoured by that much.
        d["team_spread"] = g["spread_line"] if side == "home" else -g["spread_line"]
        d["margin"] = d["pf"] - d["pa"]
        rows.append(d)

    t = pd.concat(rows, ignore_index=True)
    t["team"] = normalize_team(t["team"])
    t["opp"] = normalize_team(t["opp"])
    t["ats_margin"] = t["margin"] - t["team_spread"]
    t = t[t["ats_margin"] != 0]                 # pushes are not a bet
    t["covered"] = t["ats_margin"] > 0
    t["won"] = t["margin"] > 0
    t = t.sort_values(["team", "season", "week"]).reset_index(drop=True)

    grp = t.groupby(["team", "season"], sort=False)
    for k in (1, 2, 3):
        t[f"pf_lag{k}"] = grp["pf"].shift(k)
        t[f"margin_lag{k}"] = grp["margin"].shift(k)
        t[f"cov_lag{k}"] = grp["covered"].shift(k)
        t[f"won_lag{k}"] = grp["won"].shift(k)
    # transform keeps the roll inside each team-season; a plain shift().rolling()
    # would slide across the team boundary.
    t["pf_roll3"] = grp["pf"].transform(lambda s: s.shift(1).rolling(3).mean())
    t["pf_season_mean"] = grp["pf"].transform(lambda s: s.shift(1).expanding().mean())
    t["blk"] = t["season"].astype(str) + "_" + t["week"].astype(str)
    return t


# --------------------------------------------------------------------------
# testing machinery
# --------------------------------------------------------------------------

def block_bootstrap(d: pd.DataFrame, col: str = "covered",
                    reps: int = 5000, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    keys = d["blk"].unique()
    vals = np.array([d[d["blk"].isin(rng.choice(keys, len(keys)))][col].mean()
                     for _ in range(reps)])
    return tuple(np.percentile(vals, [2.5, 97.5]))


def season_t(d: pd.DataFrame, col: str = "covered") -> tuple[float, int]:
    """t on the per-season cover rate against 50%. The deployable unit."""
    per = d.groupby("season")[col].mean()
    per = per[d.groupby("season")[col].size() >= 4]
    if len(per) < 5:
        return float("nan"), len(per)
    return float((per.mean() - .5) / (per.std(ddof=1) / np.sqrt(len(per)))), len(per)


def evaluate(d: pd.DataFrame, label: str, col: str = "covered") -> dict:
    """One hypothesis: how does the flagged team do against the spread next?"""
    n = len(d)
    if n < MIN_GAMES:
        return {"hypothesis": label, "n": n, "cover": np.nan, "note": "too few games"}
    r = float(d[col].mean())
    lo, hi = block_bootstrap(d, col)
    # Two-sided binomial against a fair 50% market.
    p = float(stats.binomtest(int(d[col].sum()), n, 0.5).pvalue)
    t, ns = season_t(d, col)
    return {"hypothesis": label, "n": n, "cover": round(r, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "p_raw": round(p, 4), "season_t": round(t, 2), "seasons": ns,
            "roi_at_110": round(r * (100 / 110) - (1 - r), 4)}


def bh_fdr(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg: which of a family of tests survive at FDR alpha."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), bool)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m = len(idx)
    keep = 0
    for rank, i in enumerate(idx, start=1):
        if p[i] <= alpha * rank / m:
            keep = rank
    out[idx[:keep]] = True
    return out.tolist()


# --------------------------------------------------------------------------
# team form hypotheses
# --------------------------------------------------------------------------

def team_hypotheses(t: pd.DataFrame) -> pd.DataFrame:
    hot2 = (t["pf_lag1"] >= 30) & (t["pf_lag2"] >= 30)
    hot3 = hot2 & (t["pf_lag3"] >= 30)
    cold2 = (t["pf_lag1"] <= 17) & (t["pf_lag2"] <= 17)
    blow2 = (t["margin_lag1"] >= 14) & (t["margin_lag2"] >= 14)
    lost_big2 = (t["margin_lag1"] <= -14) & (t["margin_lag2"] <= -14)
    ats_hot3 = (t["cov_lag1"] == True) & (t["cov_lag2"] == True) & (t["cov_lag3"] == True)
    ats_cold3 = (t["cov_lag1"] == False) & (t["cov_lag2"] == False) & (t["cov_lag3"] == False)
    won3 = (t["won_lag1"] == True) & (t["won_lag2"] == True) & (t["won_lag3"] == True)
    lost3 = (t["won_lag1"] == False) & (t["won_lag2"] == False) & (t["won_lag3"] == False)
    surge = t["pf_roll3"] >= t["pf_season_mean"] + 10
    slump = t["pf_roll3"] <= t["pf_season_mean"] - 10

    specs = [
        ("scored 30+ in each of last 2", hot2),
        ("scored 30+ in each of last 3", hot3),
        ("scored <=17 in each of last 2", cold2),
        ("won last 2 by 14+", blow2),
        ("lost last 2 by 14+", lost_big2),
        ("covered last 3", ats_hot3),
        ("failed to cover last 3", ats_cold3),
        ("won last 3 outright", won3),
        ("lost last 3 outright", lost3),
        ("last-3 scoring 10+ above season avg", surge),
        ("last-3 scoring 10+ below season avg", slump),
    ]
    rows = [evaluate(t[m.fillna(False)], name) for name, m in specs]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# QB bounce-back
# --------------------------------------------------------------------------

def qb_weeks(seasons_from: int = 2006) -> pd.DataFrame:
    """Primary passer per team-game, with EPA per attempt."""
    files = sorted(glob.glob(str(ROSTER_DIR / "stats_player_week_*.csv")))
    keep = ["season", "week", "team", "player_id", "player_display_name",
            "position", "attempts", "passing_epa", "passing_interceptions",
            "season_type"]
    frames = []
    for f in files:
        yr = int(f.rsplit("_", 1)[1].split(".")[0])
        if yr < seasons_from:
            continue
        d = pd.read_csv(f, usecols=lambda c: c in keep, low_memory=False)
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    q = pd.concat(frames, ignore_index=True)
    q = q[(q.get("position") == "QB") & (q["attempts"].fillna(0) >= 15)]
    if "season_type" in q:
        q = q[q["season_type"].isin(["REG", "Regular Season"])]
    # One QB per team-game: the one who threw the most.
    q = (q.sort_values("attempts", ascending=False)
           .drop_duplicates(["season", "week", "team"]))
    q["team"] = normalize_team(q["team"])
    q["epa_per_att"] = q["passing_epa"] / q["attempts"]
    return q.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def qb_bounce_back(t: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """After a bad start, does the QB rebound — and does the market know?"""
    if q.empty:
        return pd.DataFrame()
    g = q.groupby(["player_id", "season"], sort=False)
    q = q.copy()
    q["prev_epa"] = g["epa_per_att"].shift(1)
    q["prev_int"] = g["passing_interceptions"].shift(1)
    q["prev_same_qb"] = g["player_id"].shift(1).notna()

    m = t.merge(q[["season", "week", "team", "player_display_name", "epa_per_att",
                   "prev_epa", "prev_int", "prev_same_qb"]],
                on=["season", "week", "team"], how="inner")
    m = m[m["prev_same_qb"].fillna(False) & m["prev_epa"].notna()]

    # Thresholds from the pooled distribution of QB starts, not hand-picked.
    lo_q, hi_q = m["prev_epa"].quantile([.20, .80])
    specs = [
        ("QB after a bottom-20% game", m["prev_epa"] <= lo_q),
        ("QB after a top-20% game", m["prev_epa"] >= hi_q),
        ("QB after 2+ interceptions", m["prev_int"] >= 2),
        ("QB after 3+ interceptions", m["prev_int"] >= 3),
        ("QB after negative total EPA", m["prev_epa"] < 0),
    ]
    rows = []
    for name, mask in specs:
        sub = m[mask.fillna(False)]
        r = evaluate(sub, name)
        # The reversion itself, to separate "plays better" from "beats the line".
        r["next_epa_per_att"] = (round(float(sub["epa_per_att"].mean()), 4)
                                 if len(sub) else np.nan)
        r["prior_epa_per_att"] = (round(float(sub["prev_epa"].mean()), 4)
                                  if len(sub) else np.nan)
        r["baseline_epa_per_att"] = round(float(m["epa_per_att"].mean()), 4)
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------

def top100_qb_flags() -> pd.DataFrame:
    """Map each season's NFL Top 100 to nflverse player ids.

    The list is published pre-season, so season S's list is legal information
    for season S games — the same rule `squad.py` uses. Coverage starts 2011.
    """
    t100 = Path("data/rosters/awards/top100.csv")
    players = ROSTER_DIR / "players.csv"
    if not t100.exists() or not players.exists():
        return pd.DataFrame()
    t = pd.read_csv(t100)
    p = pd.read_csv(players, usecols=["gsis_id", "pfr_id"], low_memory=False)
    p = p.dropna(subset=["gsis_id", "pfr_id"]).drop_duplicates("pfr_id")
    m = t.merge(p, on="pfr_id", how="inner")
    return m[["season", "rank", "gsis_id"]].rename(columns={"gsis_id": "player_id"})


def qb_bounce_by_tier(t: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """Does an elite QB bounce back differently — and does the line know?

    Two mechanisms pull in opposite directions. A Top 100 QB has a higher true
    mean, so a bad game is more likely to be luck and should revert *further*.
    But he is also the more heavily bet name, so any market overreaction should
    be *larger* against him. The first shows up in EPA, the second in ATS.
    """
    if q.empty:
        return pd.DataFrame()
    flags = top100_qb_flags()
    if flags.empty:
        return pd.DataFrame()

    g = q.groupby(["player_id", "season"], sort=False)
    q = q.copy()
    q["prev_epa"] = g["epa_per_att"].shift(1)
    q["prev_int"] = g["passing_interceptions"].shift(1)

    q = q.merge(flags, on=["season", "player_id"], how="left")
    q["is_top100"] = q["rank"].notna()
    # Top 100 coverage starts in 2011; earlier seasons are unknown, not "no".
    q = q[q["season"] >= 2011]

    m = t.merge(q[["season", "week", "team", "epa_per_att", "prev_epa",
                   "prev_int", "is_top100", "rank"]],
                on=["season", "week", "team"], how="inner")
    m = m[m["prev_epa"].notna()]
    lo_q = m["prev_epa"].quantile(.20)

    rows = []
    for tier_name, tier in (("Top 100 QB", m["is_top100"]),
                            ("not Top 100", ~m["is_top100"])):
        base = m[tier]
        for cond_name, cond in (("after a bottom-20% game", base["prev_epa"] <= lo_q),
                                ("after 2+ interceptions", base["prev_int"] >= 2),
                                ("all starts", pd.Series(True, index=base.index))):
            sub = base[cond.fillna(False)]
            r = evaluate(sub, f"{tier_name} — {cond_name}")
            r["prior_epa"] = round(float(sub["prev_epa"].mean()), 4) if len(sub) else np.nan
            r["next_epa"] = round(float(sub["epa_per_att"].mean()), 4) if len(sub) else np.nan
            r["tier_baseline_epa"] = round(float(base["epa_per_att"].mean()), 4)
            rows.append(r)

    # Within the Top 100, does where he ranks matter?
    elite = m[m["is_top100"] & (m["prev_epa"] <= lo_q)]
    for name, sub in (("Top 100 rank 1-25", elite[elite["rank"] <= 25]),
                      ("Top 100 rank 26-100", elite[elite["rank"] > 25])):
        r = evaluate(sub, f"{name} — after a bottom-20% game")
        r["prior_epa"] = round(float(sub["prev_epa"].mean()), 4) if len(sub) else np.nan
        r["next_epa"] = round(float(sub["epa_per_att"].mean()), 4) if len(sub) else np.nan
        r["tier_baseline_epa"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def hot_cold_sign_test(team: pd.DataFrame) -> dict:
    """Every hot bucket landed under 50% and every cold one over it. Is that luck?

    The individual tests are underpowered, but their *signs* carry information
    the per-test p-values throw away. Treated as one sign test — with the caveat
    that the buckets overlap, so this is directional evidence, not a clean p.
    """
    hot = ["scored 30+ in each of last 2", "scored 30+ in each of last 3",
           "won last 2 by 14+", "covered last 3", "won last 3 outright",
           "last-3 scoring 10+ above season avg"]
    cold = ["scored <=17 in each of last 2", "lost last 2 by 14+",
            "failed to cover last 3", "lost last 3 outright",
            "last-3 scoring 10+ below season avg"]
    h = team[team["hypothesis"].isin(hot)]["cover"]
    c = team[team["hypothesis"].isin(cold)]["cover"]
    agree = int((h < .5).sum() + (c > .5).sum())
    total = len(h) + len(c)
    return {"hot_buckets_below_50": int((h < .5).sum()), "hot_total": len(h),
            "cold_buckets_above_50": int((c > .5).sum()), "cold_total": len(c),
            "directionally_consistent": f"{agree}/{total}",
            "sign_test_p_if_independent": round(
                float(stats.binomtest(agree, total, 0.5).pvalue), 4),
            "caveat": "buckets overlap; treat as directional, not a clean p"}


def composite_fade(t: pd.DataFrame) -> pd.DataFrame:
    """One pre-specified strategy: fade any team arriving hot, back any arriving cold.

    Collapsing the family into a single rule is the honest way to use the sign
    pattern — it is one test, not sixteen, and it is the thing you would actually
    bet. Split in half by era so the second block is a genuine hold-out.
    """
    hot = (((t["pf_lag1"] >= 30) & (t["pf_lag2"] >= 30))
           | ((t["margin_lag1"] >= 14) & (t["margin_lag2"] >= 14))
           | ((t["cov_lag1"] == True) & (t["cov_lag2"] == True) & (t["cov_lag3"] == True)))
    cold = (((t["pf_lag1"] <= 17) & (t["pf_lag2"] <= 17))
            | ((t["margin_lag1"] <= -14) & (t["margin_lag2"] <= -14))
            | ((t["cov_lag1"] == False) & (t["cov_lag2"] == False) & (t["cov_lag3"] == False)))

    d = t.copy()
    d["bet_hits"] = np.nan
    d.loc[hot.fillna(False), "bet_hits"] = (~d.loc[hot.fillna(False), "covered"]).astype(float)
    d.loc[cold.fillna(False) & ~hot.fillna(False), "bet_hits"] = \
        d.loc[cold.fillna(False) & ~hot.fillna(False), "covered"].astype(float)
    bets = d[d["bet_hits"].notna()].copy()
    bets["covered"] = bets["bet_hits"].astype(bool)

    mid = int(bets["season"].median())
    rows = []
    for label, g in (("all seasons", bets),
                     (f"discovery (<= {mid})", bets[bets.season <= mid]),
                     (f"hold-out (> {mid})", bets[bets.season > mid])):
        r = evaluate(g, label)
        r["hypothesis"] = f"fade hot / back cold — {label}"
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-season", type=int, default=2006)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    t = team_games(args.from_season)
    print(f"team-games {t.season.min()}-{t.season.max()}: {len(t)} "
          f"(pushes dropped)\nbaseline cover rate: {t['covered'].mean():.3%} "
          f"| break-even {BREAK_EVEN:.2%}\n")

    print("=== team form: how does the STREAKING team do ATS next game? ===")
    team = team_hypotheses(t)
    print(team.to_string(index=False))

    q = qb_weeks(args.from_season)
    qb = qb_bounce_back(t, q)
    print(f"\n=== QB bounce-back ({len(q)} qualifying starts) ===")
    if not qb.empty:
        print(qb[["hypothesis", "n", "prior_epa_per_att", "next_epa_per_att",
                  "baseline_epa_per_att", "cover", "ci_lo", "ci_hi",
                  "p_raw", "season_t"]].to_string(index=False))

    tier = qb_bounce_by_tier(t, q)
    print("\n=== QB bounce-back split by NFL Top 100 status (2011-2025) ===")
    if not tier.empty:
        print(tier[["hypothesis", "n", "prior_epa", "next_epa", "tier_baseline_epa",
                    "cover", "ci_lo", "ci_hi", "p_raw", "season_t"]].to_string(index=False))

    both = pd.concat([team, qb, tier], ignore_index=True)
    tested = both[both["cover"].notna()].copy()
    tested["survives_fdr"] = bh_fdr(tested["p_raw"].tolist())
    tested["survives_bonferroni"] = tested["p_raw"] < 0.05 / len(tested)

    print(f"\n=== multiple-testing correction across all {len(tested)} tests ===")
    print(tested[["hypothesis", "n", "cover", "p_raw", "season_t",
                  "survives_fdr", "survives_bonferroni"]].to_string(index=False))
    winners = tested[tested["survives_fdr"]]
    print(f"\nsurviving BH-FDR at 5%: "
          f"{', '.join(winners['hypothesis']) if len(winners) else 'none'}")

    print("\n=== but the SIGNS are consistent — is that luck? ===")
    st = hot_cold_sign_test(team)
    for k, v in st.items():
        print(f"  {k}: {v}")

    print("\n=== collapse the family into one pre-specified strategy ===")
    comp = composite_fade(t)
    print(comp[["hypothesis", "n", "cover", "ci_lo", "ci_hi", "p_raw",
                "season_t", "roi_at_110"]].to_string(index=False))

    if args.save:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        team.to_csv(ARTIFACTS / "team_form.csv", index=False)
        qb.to_csv(ARTIFACTS / "qb_bounce_back.csv", index=False)
        tested.to_csv(ARTIFACTS / "all_tests_corrected.csv", index=False)
        comp.to_csv(ARTIFACTS / "composite_fade.csv", index=False)
        pd.DataFrame([st]).to_csv(ARTIFACTS / "sign_test.csv", index=False)
        tier.to_csv(ARTIFACTS / "qb_top100_tiers.csv", index=False)
        print(f"\nsaved -> {ARTIFACTS}")


if __name__ == "__main__":
    main()
