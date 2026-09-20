"""
This week's fixtures against the clubs' own history.

The weekly email already says what the model expects. This says *why it
looks different from usual*: for every club playing this week, how far
its current form sits from the level that club has historically played
at, and which of this week's matches are the ones running furthest from
trend.

The comparison is deliberately each club against **itself**, not against
its league. A league-average baseline would report that Manchester City
creates more chances than average, which is not news and not a trend; a
club-against-itself baseline reports that a club is creating more than
*it* usually does, which is the thing that changes a match.

Two readings come out of it:

- **Aggregate** — per metric, the mean of this week's sides' current form
  against the mean of those same clubs' baselines. Answers "is this a
  high-chance week, a pressing week, a tired week" for the slate as a
  whole.
- **Movers** — the individual club-metric pairs furthest from their own
  baseline, scaled by how much clubs differ on that metric in the first
  place, so a 0.4 xG swing and a 3-point PPDA swing can be ranked against
  each other.

The baseline is every match a club played in a **completed prior
season**. Using the current season would compare form against a window
that mostly contains the same matches, which measures nothing. A club
with no prior-season coverage — promoted, newly covered by the feed — has
no baseline and is skipped rather than compared against someone else's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from soccer.clubs.model import advanced as adv

# (per-side form column on the slate, raw per-match metric it is a
# rolling mean of, label, whether a higher number is the good one,
# decimals to publish it at).
#
# Only the EWM form columns appear: r10 and the venue splits measure the
# same quantity through a different window, and three readings of one
# metric would pad the table without adding a fact.
#
# The decimals matter more than they look. xG per shot and deep share
# live between 0 and 1, so at two decimals a real move prints as "+0.00"
# with an arrow beside it — a rounding artifact wearing the clothes of a
# finding. Each metric is published at the precision its own scale needs,
# and a delta that rounds to nothing at that precision is reported as
# level rather than given a direction.
TREND_METRICS: list[tuple[str, str, str, bool, int]] = [
    ("xg_for_ewm", "xg_for", "xG created", True, 2),
    ("xg_against_ewm", "xg_against", "xG conceded", False, 2),
    ("npxg_for_ewm", "npxg_for", "npxG created", True, 2),
    ("npxg_against_ewm", "npxg_against", "npxG conceded", False, 2),
    ("xpts_for_ewm", "xpts_for", "Expected points", True, 2),
    ("xg_per_shot_ewm", "xg_per_shot", "xG per shot", True, 3),
    ("ppda_for_ewm", "ppda_for", "PPDA (pressing)", False, 2),
    ("deep_for_ewm", "deep_for", "Deep completions", True, 2),
    ("deep_against_ewm", "deep_against", "Deep completions allowed", False, 2),
    ("deep_share_ewm", "deep_share", "Deep share", True, 3),
]

# A club needs this many prior-season matches before its baseline is
# treated as a level rather than a small sample.
MIN_BASELINE_MATCHES = 20
# How many club-metric pairs the movers list names.
TOP_MOVERS = 8


def club_baselines(before_season: str,
                   matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (league, team, metric): the club's mean in completed seasons
    before `before_season`, and how many matches that rests on."""
    matches = adv.match_metrics() if matches is None else matches
    rows = adv.team_rows(matches)
    past = rows[rows["season"].notna() & (rows["season"] < before_season)]
    raw = [raw_col for _, raw_col, _, _, _ in TREND_METRICS]
    g = past.groupby(["league", "team"])
    out = g[raw].mean()
    out["matches"] = g.size()
    return out.reset_index()


def _scale(values: pd.Series) -> float:
    """How much clubs differ on a metric, used to make swings in
    different units comparable. Standard deviation across club baselines;
    0 or NaN (a metric with one club, or none) disables the scaling
    rather than dividing by it."""
    s = float(values.std(ddof=0)) if len(values) > 1 else 0.0
    return s if np.isfinite(s) and s > 0 else 0.0


def week_trends(slate: pd.DataFrame, season: str,
                matches: pd.DataFrame | None = None) -> dict:
    """Compare the week's fixtures with the clubs' own history.

    `slate` is a frame carrying the per-side form columns (the wide frame
    `predict.build_slate` returns). Returns the aggregate table, the
    biggest movers, and what the comparison rests on.
    """
    if slate.empty:
        return {"season": season, "fixtures": 0, "clubs": 0,
                "aggregate": [], "movers": [], "baseline": {}}

    base = club_baselines(season, matches=matches)
    base = base[base["matches"] >= MIN_BASELINE_MATCHES]
    keyed = base.set_index(["league", "team"])

    # One row per side of every fixture: the club, its current form, and
    # the baseline to judge it against.
    sides = []
    for r in slate.to_dict(orient="records"):
        for side in ("home", "away"):
            team = r[f"{side}_team"]
            try:
                b = keyed.loc[(r["league"], team)]
            except KeyError:
                continue
            sides.append({"league": r["league"], "team": team, "side": side,
                          "match": f"{r['home_team']} v {r['away_team']}",
                          "date": r["date"], "row": r, "base": b})

    aggregate, movers = [], []
    for form_col, raw_col, label, higher_better, decimals in TREND_METRICS:
        scale = _scale(base[raw_col])
        cur_vals, base_vals = [], []
        for s in sides:
            cur = s["row"].get(f"{s['side']}_{form_col}")
            ref = s["base"][raw_col]
            if cur is None or not np.isfinite(cur) or not np.isfinite(ref):
                continue
            cur_vals.append(float(cur))
            base_vals.append(float(ref))
            delta = float(cur) - float(ref)
            movers.append({
                "metric": label,
                "key": form_col,
                "team": s["team"],
                "match": s["match"],
                "date": s["date"],
                "league": s["league"],
                "current": round(float(cur), decimals),
                "baseline": round(float(ref), decimals),
                "delta": round(delta, decimals),
                "decimals": decimals,
                "z": round(delta / scale, 2) if scale else None,
                # Whether the club is running *better* than its own norm,
                # which is not the same as the number going up: conceding
                # more xG is a bigger number and a worse team. `level` is
                # a move too small to show at this metric's precision, and
                # is neither better nor worse.
                "level": round(delta, decimals) == 0,
                "better": (delta > 0) == higher_better,
            })
        if not cur_vals:
            continue
        mean_cur = float(np.mean(cur_vals))
        mean_base = float(np.mean(base_vals))
        aggregate.append({
            "metric": label,
            "key": form_col,
            "sides": len(cur_vals),
            "current": round(mean_cur, decimals),
            "baseline": round(mean_base, decimals),
            "delta": round(mean_cur - mean_base, decimals),
            "decimals": decimals,
            "z": round((mean_cur - mean_base) / scale, 2) if scale else None,
            "level": round(mean_cur - mean_base, decimals) == 0,
            "better": (mean_cur - mean_base > 0) == higher_better,
        })

    movers.sort(key=lambda m: abs(m["z"]) if m["z"] is not None else 0,
                reverse=True)
    return {
        "season": season,
        "fixtures": int(len(slate)),
        "clubs": len({s["team"] for s in sides}),
        "aggregate": sorted(
            aggregate,
            key=lambda a: abs(a["z"]) if a["z"] is not None else 0,
            reverse=True),
        "movers": movers[:TOP_MOVERS],
        "baseline": {
            "seasons_before": season,
            "min_matches": MIN_BASELINE_MATCHES,
            "clubs_with_baseline": int(len(base)),
            "note": (
                "Each club is compared with its own mean in completed "
                "seasons before this one, not with a league average; the "
                "z column scales a swing by how much clubs differ on that "
                "metric, so swings in different units can be ranked "
                "against each other."
            ),
        },
    }
