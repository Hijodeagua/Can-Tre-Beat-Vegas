"""Squad-quality features: draft pedigree, honors, QB quality, coaching churn.

Built per team **per week** off nflverse weekly rosters, so a team that loses
its starting quarterback in week 9 looks different in week 10 — unlike
season-level roster metrics, which would leak the back half of the year into
the front half.

## Leakage rules

Each feature states what it may see:

- **Draft round** — fixed when the player enters the league, so a player's
  round is legal at every game of his career.
- **Honors (All-Pro / Pro Bowl)** — only selections from seasons *strictly
  before* the game's season. All-Pro teams are announced in January, after
  the season they describe, so counting the current year would be reading the
  future.
- **NFL Top 100** — published in the pre-season, so the current season's list
  is legal from week 1. Exists 2011 onward only.
- **QB quality** — the starter's passing EPA per attempt from *prior* seasons
  only.

## Data dependencies

Required (free, automated by ``data_jobs.rosters.fetch_nflverse``):
``players.csv``, ``roster_weekly_{season}.csv``, ``stats_player_week_{season}.csv``.

Optional (``data_jobs.rosters.scrape_awards``, run locally):
``data/rosters/awards/{allpro,probowl,top100}.csv``. When absent, the honor
columns are still emitted but hold NaN, and the models simply treat them as
missing — nothing downstream breaks.

PFF grades are deliberately not wired in: they are paywalled and the project
does not have a subscription. ``qb_epa_prior`` is the open-data stand-in for
the "how good is this quarterback, really" signal PFF grades would have given.

Usage
    python3 -m NFL.model.v2.squad --build      # writes the aggregate table
    python3 -m NFL.model.v2.squad --season 2025 --week 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
NFLVERSE_DIR = REPO_ROOT / "data" / "rosters" / "nflverse"
AWARDS_DIR = REPO_ROOT / "data" / "rosters" / "awards"
OUT_PATH = REPO_ROOT / "data" / "rosters" / "squad_features.csv"
GAMES_PATH = REPO_ROOT / "data" / "schedules" / "nflverse_games.csv"

FIRST_SEASON = 2002
ACTIVE_STATUSES = ("ACT",)  # exclude practice squad (DEV), IR (RES), inactive (INA)

# Weekly rosters use a few legacy codes the schedule file never used.
ROSTER_TEAM_FIXES = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
    "SL": "STL", "LAR": "LA",
}

# Honor recency bands -> weight. A player scores the *max* band he qualifies
# for, not the sum: a selection last year is worth 1.0, not 1.0+0.75+0.5.
ALLPRO_BANDS = ((3, 1.0), (5, 0.75), (99, 0.5))
PROBOWL_LOOKBACK = 5

SQUAD_FEATURES = [
    "n_first_rounders", "n_top2_rounders", "pct_drafted",
    "allpro_score", "n_probowlers", "n_top100",
    "qb_epa_prior", "qb_quality_drop", "is_interim_coach",
]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _norm_team(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().replace(ROSTER_TEAM_FIXES)


def load_players() -> pd.DataFrame:
    path = NFLVERSE_DIR / "players.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `python3 -m data_jobs.rosters.fetch_nflverse`")
    p = pd.read_csv(path, low_memory=False,
                    usecols=["gsis_id", "pfr_id", "display_name", "position",
                             "draft_year", "draft_round", "draft_pick"])
    return p.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")


def load_weekly_rosters(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = NFLVERSE_DIR / f"roster_weekly_{season}.csv"
        if not path.exists():
            continue
        r = pd.read_csv(path, low_memory=False,
                        usecols=lambda c: c in {"season", "team", "week", "status",
                                                "gsis_id", "position", "years_exp"})
        r = r[r["status"].astype(str).str.upper().isin(ACTIVE_STATUSES)]
        frames.append(r)
    if not frames:
        raise FileNotFoundError(
            f"no weekly rosters in {NFLVERSE_DIR} — run the fetcher first")
    out = pd.concat(frames, ignore_index=True)
    out["team"] = _norm_team(out["team"])
    return out.dropna(subset=["gsis_id"])


def load_awards() -> dict[str, pd.DataFrame]:
    """Optional honor tables. Missing files yield empty frames, not errors."""
    out = {}
    for name in ("allpro", "probowl", "top100"):
        path = AWARDS_DIR / f"{name}.csv"
        out[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return out


def load_qb_season_epa(seasons: list[int]) -> pd.DataFrame:
    """Passing EPA per attempt, per QB per season (regular season only)."""
    frames = []
    for season in seasons:
        path = NFLVERSE_DIR / f"stats_player_week_{season}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path, low_memory=False,
                        usecols=lambda c: c in {"player_id", "season", "week",
                                                "season_type", "attempts",
                                                "passing_epa"})
        if "season_type" in d.columns:
            d = d[d["season_type"] == "REG"]
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["gsis_id", "season", "qb_epa"])

    st = pd.concat(frames, ignore_index=True)
    st = st[st["attempts"].fillna(0) > 0]
    agg = st.groupby(["player_id", "season"], as_index=False).agg(
        attempts=("attempts", "sum"), epa=("passing_epa", "sum"))
    # A meaningful sample only; a 3-attempt cameo is not a rating.
    agg = agg[agg["attempts"] >= 50]
    agg["qb_epa"] = agg["epa"] / agg["attempts"]
    return agg.rename(columns={"player_id": "gsis_id"})[["gsis_id", "season", "qb_epa"]]


# --------------------------------------------------------------------------
# per-team-week features
# --------------------------------------------------------------------------

def draft_features(rosters: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    r = rosters.merge(players[["gsis_id", "draft_round"]], on="gsis_id", how="left")
    r["is_r1"] = (r["draft_round"] == 1).astype(int)
    r["is_r12"] = r["draft_round"].isin([1, 2]).astype(int)
    r["drafted"] = r["draft_round"].notna().astype(int)
    return r.groupby(["season", "team", "week"], as_index=False).agg(
        n_first_rounders=("is_r1", "sum"),
        n_top2_rounders=("is_r12", "sum"),
        pct_drafted=("drafted", "mean"),
        roster_size=("gsis_id", "size"),
    )


def _honor_weight(gap: int) -> float:
    """Seasons since the selection -> weight, taking the best matching band."""
    for max_gap, weight in ALLPRO_BANDS:
        if gap <= max_gap:
            return weight
    return 0.0


def honor_features(rosters: pd.DataFrame, players: pd.DataFrame,
                   awards: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """All-Pro score, Pro Bowl count, Top 100 count per team-week.

    Returns NaN columns when the awards tables are absent, so the rest of the
    pipeline is unaffected by whether the local scrape has been run.
    """
    keys = rosters[["season", "team", "week"]].drop_duplicates()
    allpro, probowl, top100 = awards["allpro"], awards["probowl"], awards["top100"]
    if allpro.empty and probowl.empty and top100.empty:
        for col in ("allpro_score", "n_probowlers", "n_top100"):
            keys[col] = np.nan
        return keys

    r = rosters.merge(players[["gsis_id", "pfr_id"]], on="gsis_id", how="left")
    seasons = sorted(r["season"].unique())

    def _by_player(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise an awards table to (season, pfr_id)."""
        if df.empty or "season" not in df.columns:
            return pd.DataFrame(columns=["season", "pfr_id"])
        key = "pfr_id" if "pfr_id" in df.columns else None
        if key is None:
            return pd.DataFrame(columns=["season", "pfr_id"])
        return df[["season", "pfr_id"]].dropna().drop_duplicates()

    ap, pb = _by_player(allpro), _by_player(probowl)

    rows = []
    for season in seasons:
        sub = r[r["season"] == season]
        # Strictly prior seasons only — honors are announced after the fact.
        ap_prior = ap[ap["season"] < season].copy()
        if len(ap_prior):
            ap_prior["gap"] = season - ap_prior["season"]
            ap_prior["w"] = ap_prior["gap"].map(_honor_weight)
            best = ap_prior.groupby("pfr_id", as_index=False)["w"].max()
        else:
            best = pd.DataFrame(columns=["pfr_id", "w"])

        pb_prior = pb[(pb["season"] < season) & (pb["season"] >= season - PROBOWL_LOOKBACK)]
        pb_ids = set(pb_prior["pfr_id"])

        s = sub.merge(best, on="pfr_id", how="left")
        s["w"] = s["w"].fillna(0.0)
        s["pb"] = s["pfr_id"].isin(pb_ids).astype(int)

        # Top 100 is published pre-season, so the current year's list is legal.
        if not top100.empty and "pfr_id" in top100.columns:
            t100_ids = set(top100.loc[top100["season"] == season, "pfr_id"].dropna())
        else:
            t100_ids = set()
        s["t100"] = s["pfr_id"].isin(t100_ids).astype(int) if t100_ids else np.nan

        agg = s.groupby(["season", "team", "week"], as_index=False).agg(
            allpro_score=("w", "sum"),
            n_probowlers=("pb", "sum"),
            n_top100=("t100", "sum"),
        )
        # Each honor is missing independently — the Top 100 scrape can land
        # before the PFR one, and the Top 100 list doesn't exist before 2011.
        # A sum over nothing is 0.0, which would assert "this roster had no
        # All-Pros" rather than "we don't know". Restore NaN per column.
        if ap.empty:
            agg["allpro_score"] = np.nan
        if pb.empty:
            agg["n_probowlers"] = np.nan
        if not t100_ids:
            agg["n_top100"] = np.nan
        rows.append(agg)

    return pd.concat(rows, ignore_index=True)


def qb_features(games: pd.DataFrame, players: pd.DataFrame,
                qb_epa: pd.DataFrame) -> pd.DataFrame:
    """Starter quality and the drop from the previous starter.

    Answers the review question about magnitude: ``qb_change`` only said the
    name changed, so Burrow -> Browning and Burrow -> Flacco looked identical.
    ``qb_quality_drop`` is the prior starter's rating minus this one's, so a
    downgrade is positive and a promotion is negative.
    """
    if qb_epa.empty:
        return pd.DataFrame(columns=["game_id", "side", "qb_epa_prior", "qb_quality_drop"])

    name_to_id = (players.dropna(subset=["display_name"])
                  .drop_duplicates("display_name")
                  .set_index("display_name")["gsis_id"])

    rows = []
    for side in ("home", "away"):
        d = games[["game_id", "season", "week", f"{side}_team", f"{side}_qb_name"]].copy()
        d.columns = ["game_id", "season", "week", "team", "qb_name"]
        d["side"] = side
        rows.append(d)
    qb = pd.concat(rows, ignore_index=True)
    qb["gsis_id"] = qb["qb_name"].map(name_to_id)

    # Prior-season rating: the most recent season before this one with >=50 att.
    epa = qb_epa.sort_values("season")
    merged = qb.merge(epa, on="gsis_id", how="left", suffixes=("", "_epa"))
    merged = merged[merged["season_epa"] < merged["season"]] if "season_epa" in merged else merged
    best_prior = (merged.sort_values("season_epa")
                  .groupby(["game_id", "side"], as_index=False)
                  .agg(qb_epa_prior=("qb_epa", "last")))
    qb = qb.merge(best_prior, on=["game_id", "side"], how="left")

    # Drop vs the team's previous game's starter.
    qb = qb.sort_values(["season", "week"])
    qb["prev_qb_epa"] = qb.groupby(["team", "season"])["qb_epa_prior"].shift(1)
    qb["qb_quality_drop"] = qb["prev_qb_epa"] - qb["qb_epa_prior"]
    return qb[["game_id", "side", "qb_epa_prior", "qb_quality_drop"]]


def interim_coach_flags(games: pd.DataFrame) -> pd.DataFrame:
    """1 when a team's coach differs from the one it opened the season with.

    Measured at ~zero marginal signal beyond the spread (see
    SQUAD_QUALITY_PLAN.md), but it costs nothing and the tree models may find
    interactions with rest and QB churn.
    """
    rows = []
    for side in ("home", "away"):
        d = games[["game_id", "season", "gameday", f"{side}_team", f"{side}_coach"]].copy()
        d.columns = ["game_id", "season", "gameday", "team", "coach"]
        d["side"] = side
        rows.append(d)
    tc = pd.concat(rows, ignore_index=True).sort_values("gameday")
    opener = tc.groupby(["team", "season"])["coach"].first().rename("opening_coach")
    tc = tc.join(opener, on=["team", "season"])
    tc["is_interim_coach"] = (tc["coach"] != tc["opening_coach"]).astype(int)
    return tc[["game_id", "side", "is_interim_coach"]]


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def build_team_week_table(seasons: list[int] | None = None) -> pd.DataFrame:
    games = pd.read_csv(GAMES_PATH)
    seasons = seasons or sorted(s for s in games["season"].unique() if s >= FIRST_SEASON)

    players = load_players()
    rosters = load_weekly_rosters(seasons)
    awards = load_awards()

    draft = draft_features(rosters, players)
    honors = honor_features(rosters, players, awards)
    return draft.merge(honors, on=["season", "team", "week"], how="left")


def add_squad_features(games: pd.DataFrame,
                       team_week: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach home_/away_/diff squad columns to a games frame.

    Missing pieces degrade to NaN rather than raising, so ``dataset.py`` can
    call this unconditionally.
    """
    out = games.copy()
    if team_week is None:
        team_week = (pd.read_csv(OUT_PATH) if OUT_PATH.exists()
                     else pd.DataFrame(columns=["season", "team", "week"]))

    roster_cols = [c for c in ("n_first_rounders", "n_top2_rounders", "pct_drafted",
                               "allpro_score", "n_probowlers", "n_top100")
                   if c in team_week.columns]
    for side in ("home", "away"):
        if roster_cols:
            tw = team_week.rename(columns={"team": f"{side}_team",
                                           **{c: f"{side}_{c}" for c in roster_cols}})
            out = out.merge(
                tw[["season", f"{side}_team", "week"] + [f"{side}_{c}" for c in roster_cols]],
                on=["season", f"{side}_team", "week"], how="left")
        else:
            for c in ("n_first_rounders", "n_top2_rounders", "pct_drafted",
                      "allpro_score", "n_probowlers", "n_top100"):
                out[f"{side}_{c}"] = np.nan

    # QB + coach are game-level, keyed by side.
    players = load_players() if (NFLVERSE_DIR / "players.csv").exists() else pd.DataFrame()
    qb_epa = load_qb_season_epa(sorted(games["season"].unique())) if len(players) \
        else pd.DataFrame()
    qb = qb_features(games, players, qb_epa) if len(qb_epa) else pd.DataFrame()
    coach = interim_coach_flags(games)

    for side in ("home", "away"):
        if len(qb):
            q = qb[qb["side"] == side].drop(columns="side").rename(columns={
                "qb_epa_prior": f"{side}_qb_epa_prior",
                "qb_quality_drop": f"{side}_qb_quality_drop"})
            out = out.merge(q, on="game_id", how="left")
        else:
            out[f"{side}_qb_epa_prior"] = np.nan
            out[f"{side}_qb_quality_drop"] = np.nan
        c = coach[coach["side"] == side].drop(columns="side").rename(
            columns={"is_interim_coach": f"{side}_is_interim_coach"})
        out = out.merge(c, on="game_id", how="left")

    # Differentials — the form the models actually learn from.
    for col in ("n_first_rounders", "n_top2_rounders", "allpro_score",
                "n_probowlers", "n_top100", "qb_epa_prior"):
        h, a = f"home_{col}", f"away_{col}"
        if h in out.columns and a in out.columns:
            out[f"{col}_diff"] = out[h] - out[a]
    return out


SQUAD_FEATURE_COLS = [
    "n_first_rounders_diff", "n_top2_rounders_diff", "allpro_score_diff",
    "n_probowlers_diff", "n_top100_diff", "qb_epa_prior_diff",
    "home_qb_epa_prior", "away_qb_epa_prior",
    "home_qb_quality_drop", "away_qb_quality_drop",
    "home_is_interim_coach", "away_is_interim_coach",
    "home_pct_drafted", "away_pct_drafted",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="write squad_features.csv")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    args = ap.parse_args()

    tw = build_team_week_table()
    print(f"team-week rows: {len(tw)} "
          f"({tw['season'].min()}-{tw['season'].max()})")

    if args.season:
        sub = tw[tw["season"] == args.season]
        if args.week:
            sub = sub[sub["week"] == args.week]
        print(sub.sort_values("n_first_rounders", ascending=False).head(15).to_string(index=False))

    if args.build:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tw.to_csv(OUT_PATH, index=False)
        print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
