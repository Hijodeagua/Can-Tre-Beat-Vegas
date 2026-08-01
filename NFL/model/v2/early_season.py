"""Early-season ATS: the one place the market looks beatable, and why it is thin.

Backing every underdog in weeks 1-4 covered **54.6%** from 2010-2025, against a
-110 break-even of 52.38%. The effect decays monotonically through the season
and is gone by week 9, which is the shape you would expect if the mechanism is
real: opening-week numbers lean on preseason priors, and the market needs
results to correct them.

That is the bull case, and it survives the obvious robustness checks. The bear
case is the one that decides whether to bet it, and it comes down to the unit of
analysis.

## Games are not the unit of risk; seasons are

Bootstrapping whole season-weeks over the pooled 990 games puts the cover rate's
2.5th percentile at 52.8% — just above break-even, p = 0.009. But you do not
deploy this 990 times. You deploy it **once a year**, and the season-level
record is 10 winners and 6 losers with an ROI standard deviation of 10.3%. That
gives ``t = 1.67`` on 16 observations, which does not clear the bar.

Both numbers are computed here because the gap between them *is* the finding:
week-clustered game-level resampling still understates the risk, because an
entire season can be a regime. The most recent season, 2025, was the worst in
the sample at -16.5%.

## What the models do with it

Our ATS classifier does not capture the effect at all — 49.6% in weeks 1-4, no
better than its full-season rate. The market-blind spread model appears to, at
53.3%, but `attribution` shows why that is not a discovery: the blind line sits
on the underdog side of the close 59% of the time in *every* part of the season.
It has a constant dog tilt, and the tilt pays only in the weeks when the
market's dog bias exists. The naive rule beats it (54.2% vs 53.3%).

Usage
    python3 -m NFL.model.v2.early_season
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ARTIFACTS = Path(__file__).resolve().parent / "artifacts" / "early_season"
ATS_PREDS = Path(__file__).resolve().parent / "artifacts" / "oos_predictions_ats.csv"
SPREAD_PREDS = Path(__file__).resolve().parent / "artifacts" / "spread" / "preds_extratrees.csv"

BREAK_EVEN = 110 / 210
EARLY_WEEKS = 4


def roi_at_110(cover_rate: float) -> float:
    """Flat-stake ROI at -110 for a given cover rate."""
    return cover_rate * (100 / 110) - (1 - cover_rate)


def prepare(path: Path = ATS_PREDS) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d["game_type"] == "REG"].dropna(subset=["ats_margin_home"]).copy()
    fav_cov = np.where(d["spread_line"] > 0,
                       d["ats_margin_home"] > 0, d["ats_margin_home"] < 0)
    d["dog_cov"] = ~fav_cov
    d["blk"] = d["season"].astype(str) + "_" + d["week"].astype(str)
    d["home_dog"] = d["spread_line"] < 0
    if "prob" in d:
        d["model_cov"] = (d["prob"] >= .5) == (d["ats_margin_home"] > 0)
    return d


def block_bootstrap(d: pd.DataFrame, col: str = "dog_cov",
                    reps: int = 10_000, seed: int = 17) -> tuple[float, float, float]:
    """Resample whole season-weeks. Returns (lo, hi, P(rate <= break-even))."""
    rng = np.random.default_rng(seed)
    keys = d["blk"].unique()
    vals = np.array([d[d["blk"].isin(rng.choice(keys, len(keys)))][col].mean()
                     for _ in range(reps)])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), float((vals <= BREAK_EVEN).mean())


def by_week(d: pd.DataFrame, upto: int = 10) -> pd.DataFrame:
    """Per-week dog cover rate, no cumulation — shows the decay shape."""
    rows = []
    for k in range(1, upto + 1):
        g = d[d["week"] == k]
        if g.empty:
            continue
        r = float(g["dog_cov"].mean())
        rows.append({"week": k, "games": len(g), "dog_cover": round(r, 4),
                     "se": round(float(np.sqrt(.25 / len(g))), 4),
                     "roi_at_110": round(roi_at_110(r), 4)})
    return pd.DataFrame(rows)


def cutoff_sensitivity(d: pd.DataFrame, upto: int = 9) -> pd.DataFrame:
    """Was "weeks 1-4" cherry-picked? Every cumulative cutoff, same test."""
    rows = []
    for k in range(1, upto + 1):
        g = d[d["week"] <= k]
        r = float(g["dog_cov"].mean())
        lo, _, p = block_bootstrap(g)
        rows.append({"weeks_1_to": k, "games": len(g), "dog_cover": round(r, 4),
                     "boot_lo": round(lo, 4), "p_le_break_even": round(p, 4),
                     "roi_at_110": round(roi_at_110(r), 4)})
    return pd.DataFrame(rows)


def season_level(d: pd.DataFrame, weeks: int = EARLY_WEEKS) -> dict:
    """The conservative test: one observation per season, because you bet yearly."""
    e = d[d["week"] <= weeks]
    per = e.groupby("season")["dog_cov"].mean()
    roi = per.apply(roi_at_110)
    t = float(roi.mean() / (roi.std(ddof=1) / np.sqrt(len(roi))))
    # One-sided on purpose: the decision is "bet this or not", so a t of -3 is
    # not a discovery, it is a loss. A two-sided test would call both "significant".
    p_one_sided = float(stats.t.sf(t, df=len(roi) - 1))
    return {"seasons": len(roi), "mean_roi": round(float(roi.mean()), 4),
            "median_roi": round(float(roi.median()), 4),
            "sd_roi": round(float(roi.std(ddof=1)), 4),
            "t_stat": round(t, 2), "p_one_sided": round(p_one_sided, 4),
            "winning_seasons": int((roi > 0).sum()),
            "worst_season": int(roi.idxmin()), "worst_roi": round(float(roi.min()), 4),
            "profitable_at_5pct": bool(p_one_sided < 0.05)}


def out_of_sample(d: pd.DataFrame, split: int = 2018,
                  weeks: int = EARLY_WEEKS) -> pd.DataFrame:
    """Discover the rule on the early era, bet it on the later one."""
    rows = []
    for label, g in (("in-sample (< %d)" % split, d[(d.season < split) & (d.week <= weeks)]),
                     ("out-of-sample (>= %d)" % split, d[(d.season >= split) & (d.week <= weeks)])):
        r = float(g["dog_cov"].mean())
        rows.append({"era": label, "games": len(g), "dog_cover": round(r, 4),
                     "roi_at_110": round(roi_at_110(r), 4)})
    return pd.DataFrame(rows)


def attribution(weeks: int = EARLY_WEEKS) -> pd.DataFrame:
    """Does the blind spread model find this, or does it just always like dogs?"""
    if not SPREAD_PREDS.exists():
        return pd.DataFrame()
    s = pd.read_csv(SPREAD_PREDS).dropna(subset=["spread_line", "margin"])
    s["dis"] = s["pred_margin"] - s["spread_line"]
    s["on_dog"] = np.where(s["spread_line"] > 0, s["dis"] < 0, s["dis"] > 0)
    rows = []
    for lo, hi, name in ((1, weeks, f"weeks 1-{weeks}"), (5, 9, "weeks 5-9"),
                         (10, 18, "weeks 10+")):
        g = s[(s["week"] >= lo) & (s["week"] <= hi)]
        if g.empty:
            continue
        r = g["margin"] - g["spread_line"]
        graded = r != 0
        model = np.where(g["dis"][graded] > 0, r[graded] > 0, r[graded] < 0)
        dog = np.where(g["spread_line"][graded] > 0, r[graded] < 0, r[graded] > 0)
        rows.append({"slice": name, "games": len(g),
                     "our_line_on_dog": round(float(g["on_dog"].mean()), 4),
                     "model_ats": round(float(model.mean()), 4),
                     "naive_dog_ats": round(float(dog.mean()), 4)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weeks", type=int, default=EARLY_WEEKS)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    d = prepare()
    e = d[d["week"] <= args.weeks]
    late = d[d["week"] > args.weeks]

    print(f"=== back every underdog, {d.season.min()}-{d.season.max()} ===")
    for name, g in ((f"weeks 1-{args.weeks}", e), (f"weeks {args.weeks + 1}+", late)):
        lo, hi, p = block_bootstrap(g)
        r = float(g["dog_cov"].mean())
        print(f"  {name:>12}: n={len(g):4d}  dogs cover {r:.1%}  "
              f"week-block CI [{lo:.1%},{hi:.1%}]  P(<=BE)={p:.3f}  "
              f"ROI {roi_at_110(r):+.2%}")

    print("\n--- decay shape, per week ---")
    print(by_week(d).to_string(index=False))
    print("\n--- cutoff sensitivity (was the window cherry-picked?) ---")
    print(cutoff_sensitivity(d).to_string(index=False))
    print("\n--- out-of-sample era split ---")
    print(out_of_sample(d, weeks=args.weeks).to_string(index=False))
    print("\n--- season-level test (the one that decides it) ---")
    sl = season_level(d, args.weeks)
    print(pd.DataFrame([sl]).to_string(index=False))
    print(f"\n  -> {'clears' if sl['profitable_at_5pct'] else 'DOES NOT clear'} "
          f"the one-sided 5% bar on {sl['seasons']} seasons "
          f"(t={sl['t_stat']}, p={sl['p_one_sided']})")

    print("\n--- our own models on this window ---")
    mc = float(e["model_cov"].mean())
    print(f"  ATS classifier, weeks 1-{args.weeks}: {mc:.1%} "
          f"(full season {float(d['model_cov'].mean()):.1%}) — does not capture it")
    att = attribution(args.weeks)
    if not att.empty:
        print(att.to_string(index=False))

    if args.save:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        by_week(d).to_csv(ARTIFACTS / "by_week.csv", index=False)
        cutoff_sensitivity(d).to_csv(ARTIFACTS / "cutoff_sensitivity.csv", index=False)
        out_of_sample(d, weeks=args.weeks).to_csv(ARTIFACTS / "era_split.csv", index=False)
        pd.DataFrame([sl]).to_csv(ARTIFACTS / "season_level.csv", index=False)
        att.to_csv(ARTIFACTS / "attribution.csv", index=False)
        print(f"\nsaved -> {ARTIFACTS}")


if __name__ == "__main__":
    main()
