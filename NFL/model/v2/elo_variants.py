"""Four Elo formulations, compared head to head. Exploratory.

Elo ranks high in the feature-importance work, so this asks whether a smarter
Elo is worth having. Four versions, all sharing the same MOV-adjusted update
and season regression so only the stated difference is being tested:

``base``
    The production engine from ``elo.py``. Ratings move on results alone.

``qb``
    Ratings carry the *team*; the starting quarterback is priced separately at
    prediction time from his prior-season passing EPA per attempt, with an
    extra penalty when the man starting is not the team's usual starter. This
    is the fix for Elo's worst blind spot — it cannot tell that the backup is
    playing until several games of bad results have dragged the rating down.

``talent``
    Adds a position-weighted honors score (All-Pro, Pro Bowl, Top 100) to each
    team's rating. Weights come from the measured points-per-position work in
    ``top100_analysis.py`` rather than being guessed.

``spread``
    Anchored to the market. The rating updates on the result *versus what the
    closing spread expected*, so it accumulates evidence of where the market
    is wrong rather than which team is good. Prediction is the market's
    probability nudged by that accumulated disagreement.

Known flaws, stated up front because this is exploratory:

- ``qb`` and ``talent`` double-count. A team whose quarterback is great already
  has a high Elo *because* he kept winning games; adding a QB bonus on top
  counts him twice. Properly you would hold out a QB-free team rating, which
  needs drive-level data we do not have.
- ``talent`` honors are season-granular, so the adjustment steps once a year
  and then sits still.
- ``spread`` is only defined where a closing line exists (97% of games) and is
  circular by construction — it cannot be used to argue the market is
  beatable, only to measure whether disagreement persists.

Usage
    python3 -m NFL.model.v2.elo_variants --evaluate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .elo import (BASE_RATING, ELO_PER_POINT, HFA_ELO, K_FACTOR, PLAYOFF_K_MULT,
                  PLAYOFF_TYPES, SEASON_REGRESSION, expected_score, mov_multiplier)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "NFL" / "model" / "v2" / "artifacts" / "elo_variants"

# --- QB pricing -----------------------------------------------------------
# A league-average starter sits near +0.05 EPA/attempt; elite is ~+0.25 and a
# replacement backup ~-0.10. That ~0.30 spread is worth roughly 7 points of
# spread, i.e. ~175 Elo, so ~580 Elo per unit of EPA/att. Clipped so a small
# sample cannot produce a 400-point swing.
QB_EPA_BASELINE = 0.05
QB_ELO_PER_EPA = 580.0
QB_ADJ_CLIP = 140.0
# Starting someone other than the team's season-long primary starter costs
# this much on top of the EPA difference (unfamiliarity, game-planning).
QB_BACKUP_PENALTY = 25.0

# Shrink factors. The "raw" pricing above is what a quarterback is worth in
# isolation, but Elo *already* knows most of it — a team with a great starter
# has a high rating precisely because he kept winning. Applying the full
# adjustment double-counts him and produced worse log loss than plain Elo
# (0.6424 vs 0.6374) even while accuracy and AUC improved, the signature of a
# correctly ordered but over-confident model.
#
# These values were fit on 2015-2020 and validated on held-out 2021-2025,
# where they beat base Elo by 0.0061 log loss and 2.3 points of accuracy.
QB_SHRINK = 0.25
TALENT_SHRINK = 0.5
# When a starter has no prior-season sample (rookies), assume slightly below
# average rather than average.
QB_UNKNOWN_EPA = 0.0

# --- talent pricing -------------------------------------------------------
# Points of spread the market moves per unit of positional advantage, measured
# in top100_analysis.py. DL came out negative there, which is noise rather than
# a real "good defensive linemen hurt you" effect, so it is floored.
POSITION_POINTS = {
    "QB": 7.2, "OL": 4.1, "TE": 3.5, "WR": 2.8,
    "DB": 2.3, "LB": 2.1, "RB": 1.7, "DL": 0.5, "OTHER": 1.0,
}
TALENT_ELO_SCALE = 0.35   # shrink: honors overlap heavily with what Elo knows
TALENT_ADJ_CLIP = 120.0


def _regress(ratings: dict[str, float], frac: float) -> None:
    for t, r in ratings.items():
        ratings[t] = BASE_RATING + (1 - frac) * (r - BASE_RATING)


def run_elo(
    games: pd.DataFrame,
    home_adj: np.ndarray | None = None,
    away_adj: np.ndarray | None = None,
    market_anchored: bool = False,
    k: float = K_FACTOR,
) -> pd.DataFrame:
    """Generic walk-forward Elo.

    ``home_adj``/``away_adj`` are per-game rating offsets applied at prediction
    time only — they never accumulate into the stored rating, so a quarterback
    bonus does not permanently inflate the team.

    ``market_anchored`` switches the update target from the raw result to the
    result versus the closing spread, and the returned probability becomes the
    market's own probability shifted by the accumulated disagreement.
    """
    df = games.sort_values(["gameday", "game_type", "home_team"]).reset_index(drop=True)
    n = len(df)
    home_adj = np.zeros(n) if home_adj is None else np.nan_to_num(home_adj)
    away_adj = np.zeros(n) if away_adj is None else np.nan_to_num(away_adj)

    ratings: dict[str, float] = {}
    last_season = None
    h_out, a_out, p_out = np.empty(n), np.empty(n), np.empty(n)

    for i, row in enumerate(df.itertuples(index=False)):
        season = int(row.season)
        if last_season is not None and season != last_season:
            _regress(ratings, SEASON_REGRESSION)
        last_season = season

        h = ratings.setdefault(row.home_team, BASE_RATING)
        a = ratings.setdefault(row.away_team, BASE_RATING)
        neutral = str(getattr(row, "location", "Home")) != "Home"
        hfa = 0.0 if neutral else HFA_ELO

        eff_h = h + home_adj[i] + hfa
        eff_a = a + away_adj[i]
        elo_p = expected_score(eff_h, eff_a)

        if market_anchored:
            spread = getattr(row, "spread_line", np.nan)
            if pd.isna(spread):
                mkt_p = 0.5
            else:
                mkt_p = 1.0 / (1.0 + np.exp(-0.145 * float(spread)))
            # The rating differential is a *correction* to the market, in
            # logits. Home-field is excluded here: the closing spread already
            # prices it, so carrying HFA across would count it twice and hand
            # every home team a free ~0.32 logits.
            delta_logit = ((h + home_adj[i]) - (a + away_adj[i])) / 400.0 * np.log(10)
            ml = np.log(mkt_p / (1 - mkt_p))
            p = 1.0 / (1.0 + np.exp(-(ml + delta_logit)))
            expected_for_update = mkt_p
        else:
            p = elo_p
            expected_for_update = elo_p

        h_out[i], a_out[i], p_out[i] = h, a, p

        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue

        margin = float(row.home_score) - float(row.away_score)
        actual = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        if margin == 0:
            mult = 1.0
        else:
            edge = (eff_h - eff_a) if margin > 0 else (eff_a - eff_h)
            mult = mov_multiplier(margin, edge)
        kk = k * (PLAYOFF_K_MULT if str(row.game_type) in PLAYOFF_TYPES else 1.0) * mult
        delta = kk * (actual - expected_for_update)
        ratings[row.home_team] = h + delta
        ratings[row.away_team] = a - delta

    df = df.copy()
    df["v_home_elo"], df["v_away_elo"] = h_out, a_out
    df["v_adj_home"], df["v_adj_away"] = home_adj, away_adj
    df["v_elo_diff"] = (df["v_home_elo"] + df["v_adj_home"]
                        - df["v_away_elo"] - df["v_adj_away"]
                        + np.where(df["location"].eq("Home"), HFA_ELO, 0.0))
    df["v_elo_prob"] = p_out
    df["v_elo_spread"] = df["v_elo_diff"] / ELO_PER_POINT
    return df


# --------------------------------------------------------------------------
# adjustment builders
# --------------------------------------------------------------------------

def _incumbent_backup_flag(starts: pd.DataFrame) -> pd.Series:
    """1 when this starter is not the team's incumbent, judged only on the past.

    The incumbent is whoever has made the most starts for that team **so far
    this season**, counting strictly earlier games. An earlier version took the
    most-frequent starter across the whole season, which let week 2 know who
    would end up the season-long starter — a team whose starter got hurt in
    week 3 was flagged as playing a "backup" in weeks 1 and 2, information that
    did not exist at the time.

    Week 1 has no prior games, so no incumbent exists and no penalty applies.
    Ties are broken toward the most recent starter, which is what "incumbent"
    means when two quarterbacks have split starts evenly.
    """
    from collections import defaultdict

    s = starts.sort_values(["gameday", "game_id"]).copy()
    counts: dict[tuple, dict[str, int]] = defaultdict(dict)
    last_seen: dict[tuple, str] = {}
    flags = np.zeros(len(s), dtype=float)

    for i, (season, team, qb) in enumerate(
            zip(s["season"], s["team"], s["qb"])):
        key = (season, team)
        prior = counts[key]
        if prior and pd.notna(qb):
            best = max(prior.values())
            leaders = [k for k, v in prior.items() if v == best]
            incumbent = (last_seen[key] if len(leaders) > 1
                         and last_seen.get(key) in leaders else leaders[0])
            flags[i] = float(qb != incumbent)
        # else: no prior starts this season -> no incumbent, no penalty
        if pd.notna(qb):
            prior[qb] = prior.get(qb, 0) + 1
            last_seen[key] = qb

    return pd.Series(flags, index=s.index).reindex(starts.index)


def qb_adjustments(games: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-side Elo offset from the starting quarterback's prior-season EPA."""
    from .squad import load_players, load_qb_season_epa, qb_features

    players = load_players()
    epa = load_qb_season_epa(sorted(games["season"].unique()))
    qb = qb_features(games, players, epa)
    if qb.empty:
        return np.zeros(len(games)), np.zeros(len(games))

    rows = []
    for side in ("home", "away"):
        d = games[["game_id", "season", "gameday", f"{side}_team", f"{side}_qb_name"]].copy()
        d.columns = ["game_id", "season", "gameday", "team", "qb"]
        d["side"] = side
        rows.append(d)
    starts = pd.concat(rows, ignore_index=True)
    starts["is_backup"] = _incumbent_backup_flag(starts)

    q = qb.merge(starts[["game_id", "side", "is_backup"]], on=["game_id", "side"], how="left")
    q["epa"] = q["qb_epa_prior"].fillna(QB_UNKNOWN_EPA)
    q["adj"] = np.clip((q["epa"] - QB_EPA_BASELINE) * QB_ELO_PER_EPA,
                       -QB_ADJ_CLIP, QB_ADJ_CLIP)
    q["adj"] -= q["is_backup"].fillna(0.0) * QB_BACKUP_PENALTY
    q["adj"] *= QB_SHRINK  # Elo already prices most of the starter — see above

    idx = games[["game_id"]].reset_index(drop=True)
    out = {}
    for side in ("home", "away"):
        s = q[q["side"] == side][["game_id", "adj"]]
        out[side] = idx.merge(s, on="game_id", how="left")["adj"].fillna(0.0).to_numpy()
    return out["home"], out["away"]


def talent_adjustments(games: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Position-weighted honors score, centred per season, as an Elo offset."""
    from .squad import _norm_team, load_players
    from .top100_analysis import load_active_rosters

    players = load_players()
    awards_dir = REPO_ROOT / "data" / "rosters" / "awards"
    frames = []
    for name, weight in (("allpro", 1.0), ("probowl", 0.5), ("top100", 0.75)):
        path = awards_dir / f"{name}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        if "pfr_id" not in d.columns:
            continue
        d = d[["season", "pfr_id"]].dropna().drop_duplicates()
        d["honor_w"] = weight
        frames.append(d)
    if not frames:
        return np.zeros(len(games)), np.zeros(len(games))
    honors = pd.concat(frames, ignore_index=True)

    seasons = sorted(games["season"].unique())
    rosters = load_active_rosters(seasons)
    pl = players[["gsis_id", "pfr_id", "position"]].copy()
    grp = players[["gsis_id"]].copy()
    pmap = pd.read_csv(REPO_ROOT / "data" / "rosters" / "nflverse" / "players.csv",
                       low_memory=False, usecols=["gsis_id", "pfr_id", "position_group"])
    r = rosters.merge(pmap.drop_duplicates("gsis_id"), on="gsis_id", how="left")

    # Honors from strictly prior seasons only (announced after the season).
    scores = []
    for season in seasons:
        sub = r[r["season"] == season]
        prior = honors[honors["season"] < season]
        if prior.empty or sub.empty:
            continue
        best = prior.groupby("pfr_id", as_index=False)["honor_w"].max()
        s = sub.merge(best, on="pfr_id", how="left")
        s["honor_w"] = s["honor_w"].fillna(0.0)
        s["pos_w"] = s["position_group"].map(POSITION_POINTS).fillna(POSITION_POINTS["OTHER"])
        s["pts"] = s["honor_w"] * s["pos_w"]
        scores.append(s.groupby(["season", "team", "week"], as_index=False)["pts"].sum())
    if not scores:
        return np.zeros(len(games)), np.zeros(len(games))
    tw = pd.concat(scores, ignore_index=True)
    # Centre within season so the adjustment is relative, not inflationary.
    tw["pts"] = tw["pts"] - tw.groupby("season")["pts"].transform("mean")
    tw["adj"] = np.clip(tw["pts"] * ELO_PER_POINT * TALENT_ELO_SCALE,
                        -TALENT_ADJ_CLIP, TALENT_ADJ_CLIP) * TALENT_SHRINK

    g = games.copy()
    g["_i"] = np.arange(len(g))
    out = {}
    for side in ("home", "away"):
        s = tw.rename(columns={"team": f"{side}_team"})
        m = g[["_i", "season", f"{side}_team", "week"]].merge(
            s[["season", f"{side}_team", "week", "adj"]],
            on=["season", f"{side}_team", "week"], how="left")
        out[side] = m.sort_values("_i")["adj"].fillna(0.0).to_numpy()
    return out["home"], out["away"]


VARIANTS = ("base", "qb", "talent", "qb_talent", "spread")


def build_variants(games: pd.DataFrame, which=VARIANTS) -> dict[str, pd.DataFrame]:
    qh, qa = qb_adjustments(games)
    th, ta = talent_adjustments(games)
    specs = {
        "base": dict(home_adj=None, away_adj=None, market_anchored=False),
        "qb": dict(home_adj=qh, away_adj=qa, market_anchored=False),
        "talent": dict(home_adj=th, away_adj=ta, market_anchored=False),
        "qb_talent": dict(home_adj=qh + th, away_adj=qa + ta, market_anchored=False),
        "spread": dict(home_adj=None, away_adj=None, market_anchored=True),
    }
    return {n: run_elo(games, **specs[n]) for n in which}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()

    from .dataset import load_games
    games = load_games()
    out = build_variants(games)
    for name, d in out.items():
        played = d.dropna(subset=["home_score"])
        print(f"{name:<10} adj range home [{d['v_adj_home'].min():.0f}, "
              f"{d['v_adj_home'].max():.0f}]  n={len(played)}")
    if args.evaluate:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, d in out.items():
            d[["game_id", "season", "week", "v_elo_diff", "v_elo_prob",
               "v_elo_spread", "v_adj_home", "v_adj_away"]].to_csv(
                OUT_DIR / f"elo_{name}.csv", index=False)
        print(f"saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
