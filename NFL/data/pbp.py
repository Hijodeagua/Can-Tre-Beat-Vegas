"""
nflverse play-by-play -> one row of offensive and defensive efficiency
per team per game (`data/nfl/team_games.csv`).

Source: the annual nflfastR play-by-play Parquet releases,
`https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet`
(no key, ~20 MB a season). Raw files are cached under `data/nfl/raw/pbp/`
(gitignored); a season already on disk is not refetched unless it is the
current one, which is always refreshed. What is committed is the
aggregate table: ~530 rows a season, one per (game, team).

Play filter, applied once and used for every EPA/success figure:

    play_type in ("pass", "run")   and   epa is not null

`no_play` rows (pre-snap penalties, timeouts) are excluded even though
nflfastR flags many of them `pass = 1` from the intended play; kneels
and spikes carry neither flag and fall out on their own. Dropbacks are
`qb_dropback == 1` (sacks and scrambles included, which is why
`qb_dropback` and not `pass` splits the two), designed rushes are
`rush == 1 & qb_dropback == 0`.

Definitions the columns depend on, so nobody has to reverse them:

- **Neutral situation** — quarters 1-3 and the score within one
  possession (|score differential| <= 8). No win-probability gate, so the
  definition doesn't depend on someone else's model.
- **Neutral pace** — mean game-clock seconds between consecutive plays of
  the same drive in neutral situations, first play of each drive
  excluded. Lower is faster.
- **Explosive** — a run of >= 12 yards, a pass play of >= 20 yards; the
  overall rate is both over all plays.
- **Drive** — nflfastR `fixed_drive`. Points per drive are the offence's
  own score change across the drive (`posteam_score_post` at the last
  play minus `posteam_score` at the first), so extra points and two-point
  tries count exactly; when those columns are missing, Touchdown = 7 and
  Field goal = 3 from `fixed_drive_result`.
- **Red-zone trip** — a drive with any snap at or inside the opponent's
  20. TDs per trip use `fixed_drive_result == "Touchdown"`.
- **Third down** — `down == 3` pass/run plays; a conversion is
  nflfastR's `first_down == 1`; short is <= 3 to go, long is >= 7.
- **Successful series rate** — mean of `series_success` over the
  offence's distinct series.
- **Special teams EPA (net)** — EPA of special-teams plays with the team
  in possession minus EPA of those with the team defending, per game.
  nflfastR credits EPA to `posteam` on every special-teams play, so this
  is symmetric; it is noisy and ships only for the ablation to judge.
- **Pass rate over expected** — mean of nflfastR `pass_oe` (percentage
  points above the expected pass rate given situation) over dropback and
  rush plays with a value. Raw pass rate is kept for analysis only.

Defensive columns (`def_*`) are the same aggregates computed on the plays
where the team was `defteam` — what it allowed.

    python -m NFL.data.pbp --seasons 2002 2026     # backfill
    python -m NFL.data.pbp --current               # this season only
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "nfl" / "raw" / "pbp"
OUT_CSV = REPO_ROOT / "data" / "nfl" / "team_games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
FIRST_SEASON = 2002   # the 32-team, 8-division alignment; EPA exists from 1999

COLUMNS = [
    "game_id", "season", "week", "season_type", "game_date", "home_team", "away_team",
    "posteam", "defteam", "play_type", "qb_dropback", "pass", "rush", "sack",
    "complete_pass", "epa", "success", "xpass", "pass_oe", "yards_gained",
    "down", "ydstogo", "yardline_100", "qtr", "game_seconds_remaining",
    "score_differential", "fixed_drive", "fixed_drive_result", "series",
    "series_success", "special_teams_play", "first_down", "posteam_score",
    "posteam_score_post", "home_score", "away_score",
]

EXPLOSIVE_RUSH = 12
EXPLOSIVE_PASS = 20
NEUTRAL_MAX_QTR = 3
NEUTRAL_MAX_DIFF = 8
THIRD_SHORT = 3
THIRD_LONG = 7


def current_season(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 3 else today.year - 1


def fetch_season(season: int, force: bool = False, timeout: int = 600) -> Path | None:
    """Download one season's Parquet into the raw cache (skipped when
    cached, unless `force`). Returns the path, or None on failure."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"play_by_play_{season}.parquet"
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    try:
        resp = requests.get(PBP_URL.format(season=season), timeout=timeout, stream=True)
        resp.raise_for_status()
        tmp = path.with_suffix(".part")
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(path)
    except requests.RequestException as exc:
        print(f"  ! {season}: fetch failed ({exc})")
        return None
    return path


def load_season(path: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq
    names = set(pq.ParquetFile(path).schema.names)
    cols = [c for c in COLUMNS if c in names]
    df = pq.read_table(path, columns=cols).to_pandas()
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def _rate(num, den):
    return float(num) / den if den else np.nan


def _drive_table(plays: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, offence, fixed_drive) with EPA, points, yards, plays."""
    g = plays.groupby(["game_id", "posteam", "fixed_drive"], sort=False)
    first = g.first()
    last = g.last()
    drives = pd.DataFrame({
        "epa": g["epa"].sum(),
        "yards": g["yards_gained"].sum(),
        "plays": g.size(),
        "result": first["fixed_drive_result"],
        "red_zone": g["yardline_100"].min() <= 20,
    })
    if plays["posteam_score_post"].notna().any():
        drives["points"] = (last["posteam_score_post"] - first["posteam_score"]).clip(lower=0)
    else:
        drives["points"] = drives["result"].map({"Touchdown": 7, "Field goal": 3}).fillna(0)
    return drives.reset_index()


def _offense(plays: pd.DataFrame, st: pd.DataFrame, team_col: str) -> pd.DataFrame:
    """Aggregates keyed by (game_id, team) for the side named by team_col
    ('posteam' -> what the team did; 'defteam' -> what it allowed)."""
    p = plays
    is_db = p["qb_dropback"] == 1
    is_rush = (p["rush"] == 1) & ~is_db
    early = p["down"].isin([1, 2])
    neutral = (p["qtr"] <= NEUTRAL_MAX_QTR) & (p["score_differential"].abs() <= NEUTRAL_MAX_DIFF)
    third = p["down"] == 3
    expl = ((p["play_type"] == "run") & (p["yards_gained"] >= EXPLOSIVE_RUSH)) | \
           ((p["play_type"] == "pass") & (p["yards_gained"] >= EXPLOSIVE_PASS))

    key = ["game_id", team_col]
    g = p.groupby(key, sort=False)

    def gmean(mask, col):
        return p[col].where(mask).groupby([p["game_id"], p[team_col]], sort=False).mean()

    def gsum(mask):
        return mask.astype(int).groupby([p["game_id"], p[team_col]], sort=False).sum()

    out = pd.DataFrame({
        "plays": g.size(),
        "epa": g["epa"].mean(),
        "success": g["success"].mean(),
        "dropbacks": gsum(is_db),
        "db_epa": gmean(is_db, "epa"),
        "db_success": gmean(is_db, "success"),
        "sack_rate": gsum(is_db & (p["sack"] == 1)) / gsum(is_db).replace(0, np.nan),
        "rushes": gsum(is_rush),
        "rush_epa": gmean(is_rush, "epa"),
        "rush_success": gmean(is_rush, "success"),
        "early_epa": gmean(early, "epa"),
        "early_success": gmean(early, "success"),
        "pass_rate": gmean(is_db | is_rush, "pass"),
        "proe": gmean(is_db | is_rush, "pass_oe"),
        "proe_neutral": gmean((is_db | is_rush) & neutral, "pass_oe"),
        "explosive_rush_rate": gsum(expl & (p["play_type"] == "run")) / gsum(p["play_type"] == "run").replace(0, np.nan),
        "explosive_pass_rate": gsum(expl & (p["play_type"] == "pass")) / gsum(p["play_type"] == "pass").replace(0, np.nan),
        "explosive_rate": gsum(expl) / g.size(),
        "third_att": gsum(third),
        "third_conv": gsum(third & (p["first_down"] == 1)) / gsum(third).replace(0, np.nan),
        "third_epa": gmean(third, "epa"),
        "third_dist": gmean(third, "ydstogo"),
        "third_short_conv": gsum(third & (p["ydstogo"] <= THIRD_SHORT) & (p["first_down"] == 1))
                            / gsum(third & (p["ydstogo"] <= THIRD_SHORT)).replace(0, np.nan),
        "third_long_conv": gsum(third & (p["ydstogo"] >= THIRD_LONG) & (p["first_down"] == 1))
                           / gsum(third & (p["ydstogo"] >= THIRD_LONG)).replace(0, np.nan),
        "rz_epa_sum": gmean(p["yardline_100"] <= 20, "epa") * gsum(p["yardline_100"] <= 20),
    })

    # Pace: seconds between consecutive neutral plays of the same drive.
    q = p[neutral].sort_values(["game_id", "fixed_drive", "game_seconds_remaining"], ascending=[True, True, False])
    gap = q.groupby(["game_id", team_col, "fixed_drive"], sort=False)["game_seconds_remaining"].diff(-1)
    out["neutral_pace"] = gap.groupby([q["game_id"], q[team_col]], sort=False).mean()

    # Drives (offence-keyed; for the defence view we key by defteam, which
    # is the same drive seen from the other sideline).
    drives = _drive_table(p.assign(**{team_col: p[team_col]}).rename(columns={}) if team_col == "posteam"
                          else p.rename(columns={"posteam": "_off", "defteam": "posteam"}))
    dg = drives.groupby(["game_id", "posteam"], sort=False)
    rz = drives[drives["red_zone"]].groupby(["game_id", "posteam"], sort=False)
    dtab = pd.DataFrame({
        "drives": dg.size(),
        "epa_per_drive": dg["epa"].mean(),
        "points_per_drive": dg["points"].mean(),
        "yards_per_drive": dg["yards"].mean(),
        "plays_per_drive": dg["plays"].mean(),
        "rz_trips": rz.size(),
        "rz_td_per_trip": rz["result"].apply(lambda s: (s == "Touchdown").mean()),
        "rz_pts_per_trip": rz["points"].mean(),
    })
    dtab.index.names = ["game_id", team_col]
    out = out.join(dtab, how="left")
    out["rz_trips"] = out["rz_trips"].fillna(0)
    out["rz_epa"] = out["rz_epa_sum"] / out["rz_trips"].replace(0, np.nan)
    out = out.drop(columns=["rz_epa_sum"])

    # Series success: one value per distinct series.
    ser = p.dropna(subset=["series"]).drop_duplicates(["game_id", team_col, "series"])
    out["series_success"] = ser.groupby(["game_id", team_col], sort=False)["series_success"].mean()

    # Special teams, net.
    st_for = st.groupby(["game_id", "posteam"], sort=False)["epa"].sum()
    st_against = st.groupby(["game_id", "defteam"], sort=False)["epa"].sum()
    st_for.index.names = st_against.index.names = ["game_id", team_col]
    out["st_epa_net"] = st_for.reindex(out.index).fillna(0) - st_against.reindex(out.index).fillna(0)
    return out


def team_game_aggregates(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team): context + off_* + def_* columns."""
    pbp = pbp[pbp["posteam"].notna() & pbp["defteam"].notna()].copy()
    plays = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()].copy()
    st = pbp[(pbp["special_teams_play"] == 1) & pbp["epa"].notna()]

    off = _offense(plays, st, "posteam")
    off.columns = [c if c == "st_epa_net" else f"off_{c}" for c in off.columns]
    off.index.names = ["game_id", "team"]
    dfn = _offense(plays, st, "defteam")
    dfn.columns = [f"def_{c}" for c in dfn.columns]
    dfn.index.names = ["game_id", "team"]
    # Net ST is defined once from the team's own perspective.
    dfn = dfn.drop(columns=["def_st_epa_net"])

    ctx = pbp.groupby("game_id", sort=False).agg(
        season=("season", "first"), week=("week", "first"), season_type=("season_type", "first"),
        date=("game_date", "first"), home_team=("home_team", "first"), away_team=("away_team", "first"),
        home_score=("home_score", "max"), away_score=("away_score", "max"),
    )
    rows = []
    for gid, c in ctx.iterrows():
        for team, opp, is_home, pf, pa in ((c.home_team, c.away_team, 1, c.home_score, c.away_score),
                                           (c.away_team, c.home_team, 0, c.away_score, c.home_score)):
            rows.append({"game_id": gid, "team": team, "opponent": opp, "is_home": is_home,
                         "season": int(c.season), "week": int(c.week), "season_type": c.season_type,
                         "date": str(c.date)[:10], "points_for": pf, "points_against": pa})
    table = pd.DataFrame(rows).set_index(["game_id", "team"]).join(off, how="left").join(dfn, how="left")
    return table.reset_index().sort_values(["season", "week", "game_id", "is_home"]).reset_index(drop=True)


def build(seasons: list[int], force_current: bool = True) -> pd.DataFrame:
    frames = []
    this = current_season()
    for season in seasons:
        path = fetch_season(season, force=(season >= this and force_current))
        if path is None:
            continue
        agg = team_game_aggregates(load_season(path))
        print(f"  + {season}: {len(agg)} team-games")
        frames.append(agg)
    if not frames:
        return pd.DataFrame()
    new = pd.concat(frames, ignore_index=True)
    if OUT_CSV.exists():
        old = pd.read_csv(OUT_CSV)
        old = old[~old["season"].isin(new["season"].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["season", "week", "game_id", "is_home"]).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(OUT_CSV, index=False, float_format="%.5f")
    return new


def load_team_games(path: Path = OUT_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs=2, type=int, metavar=("FIRST", "LAST"))
    parser.add_argument("--current", action="store_true")
    args = parser.parse_args()
    if args.current:
        seasons = [current_season()]
    elif args.seasons:
        seasons = list(range(args.seasons[0], args.seasons[1] + 1))
    else:
        seasons = list(range(FIRST_SEASON, current_season() + 1))
    print(f"Building team-game aggregates for {seasons[0]}-{seasons[-1]}…")
    out = build(seasons)
    if out.empty:
        print("Nothing built; committed table untouched.")
        sys.exit(1)
    print(f"Wrote {len(out)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
