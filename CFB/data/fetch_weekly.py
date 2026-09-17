"""
SportsDataverse weekly team summaries -> data/college_football/team_weeks.csv

Source (no key, one Parquet per season, rebuilt upstream during the
season):

    https://github.com/sportsdataverse/sportsdataverse-data/releases/download/
        cfb_team_summaries_weekly/cfb_team_summaries_weekly_{YEAR}.parquet

One row per (season, team_id, through_week): the team's season-to-date
aggregates *including* week `through_week`. That word matters for
leakage: the snapshot with `through_week == W` contains the week-W games,
so a week-W game must be featured from the `W-1` snapshot. `snapshot_for`
does that and `tests/test_cfb_advanced.py` asserts it. Postseason games
(the spine restarts `week` at 1 with `season_type == "postseason"`) use
the season's last snapshot, which predates every bowl.

Kept columns are the ones the second stage evaluates, with
SportsDataverse's own names so a reader can check a definition against
the publisher. Their meaning, per the cfbfastR/sportsdataverse
`calculate_team_summaries` code:

    adj_off_epa / adj_def_epa   opponent-adjusted EPA per play (ridge on
                                offence + defence + home effects); net = off − def
    EPAplay_off/_def            raw EPA per play, for / allowed
    success_off/_def            share of plays with EPA > 0
    early_down_EPA_off/_def     EPA per play on downs 1-2
    explosive_off/_def          share of plays with EPA above the explosive cut
    havoc_off/_def              TFL + forced fumbles + INT + PBU per play (havoc_def
                                is the team's defence causing it)
    EPAdrive_off/_def           EPA per drive
    drivesgame_off/_def         drives per game
    playsdrive_off/_def         plays per drive
    yardsdrive_off/_def         yards per drive
    passrate_off / rushrate_off pass / rush share of plays
    red_zone_success_off/_def   success rate inside the 20
    third_down_success_off/_def third-down success rate (EPA > 0), and
    third_down_distance_off/_def mean yards to go on third down
    _pass / _rush suffixes      the same, split by play type

Raw Parquet files are cached under data/college_football/raw/weekly/ and
are not committed; the processed CSV is.

Usage:
    python -m CFB.data.fetch_weekly              # last + current season
    python -m CFB.data.fetch_weekly --all        # 2004 → current
    python -m CFB.data.fetch_weekly --seasons 2019 2021
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from CFB.data.fetch_schedule import current_season

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "college_football"
RAW_DIR = DATA_DIR / "raw" / "weekly"
OUT_CSV = DATA_DIR / "team_weeks.csv"
URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
       "cfb_team_summaries_weekly/cfb_team_summaries_weekly_{season}.parquet")
FIRST_SEASON = 2004

KEY = ["season", "team_id", "team", "through_week"]
BASE_METRICS = [
    "adj_off_epa", "adj_def_epa", "net_adj_epa",
    "EPAplay_off", "EPAplay_def", "success_off", "success_def",
    "early_down_EPA_off", "early_down_EPA_def", "explosive_off", "explosive_def",
    "havoc_off", "havoc_def", "EPAdrive_off", "EPAdrive_def",
    "drivesgame_off", "drivesgame_def", "playsdrive_off", "playsdrive_def",
    "yardsdrive_off", "yardsdrive_def", "passrate_off", "rushrate_off",
    "red_zone_success_off", "red_zone_success_def",
    "third_down_success_off", "third_down_success_def",
    "third_down_distance_off", "third_down_distance_def",
    "plays_off", "plays_def",
]
SPLIT_METRICS = [
    "EPAplay_off_pass", "EPAplay_off_rush", "EPAplay_def_pass", "EPAplay_def_rush",
    "success_off_pass", "success_off_rush", "success_def_pass", "success_def_rush",
    "explosive_off_pass", "explosive_off_rush", "explosive_def_pass", "explosive_def_rush",
]
METRICS = BASE_METRICS + SPLIT_METRICS
COLUMNS = KEY + ["division", "conference"] + METRICS


def fetch_season(season: int, force: bool = False, timeout: int = 300) -> Path | None:
    """Download one season's Parquet into the raw cache (skipped when
    cached, unless `force`). None on 404 or a failed request."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"cfb_team_summaries_weekly_{season}.parquet"
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    try:
        resp = requests.get(URL.format(season=season), timeout=timeout, stream=True)
        if resp.status_code == 404:
            return None
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


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """One season's Parquet -> processed rows (KEY + METRICS). Columns
    the upstream lacks come through as NaN so the schema is stable."""
    df = raw.copy()
    out = pd.DataFrame({
        "season": df["season"].astype(int),
        "team_id": pd.to_numeric(df["team_id"], errors="coerce").astype("Int64"),
        "team": df["pos_team"].astype(str).str.strip(),
        "through_week": df["through_week"].astype(int),
        "division": df.get("division"),
        "conference": df.get("conference"),
    })
    for m in METRICS:
        out[m] = pd.to_numeric(df[m], errors="coerce") if m in df.columns else float("nan")
    out = out[COLUMNS].dropna(subset=["team_id"])
    out = out.drop_duplicates(["season", "team_id", "through_week"], keep="last")
    return out.sort_values(["season", "through_week", "team_id"]).reset_index(drop=True)


def load_team_weeks(path: Path = OUT_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def build(seasons: list[int], force_current: bool = True) -> pd.DataFrame:
    this = current_season()
    frames = []
    for season in seasons:
        path = fetch_season(season, force=(season >= this and force_current))
        if path is None:
            print(f"  {season}: not available upstream")
            continue
        rows = normalize(pd.read_parquet(path))
        print(f"  + {season}: {len(rows)} team-weeks, through week {rows['through_week'].max()}")
        frames.append(rows)
    if not frames:
        return pd.DataFrame()
    new = pd.concat(frames, ignore_index=True)
    old = load_team_weeks()
    if len(old):
        old = old[~old["season"].isin(new["season"].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["season", "through_week", "team_id"]).reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    new.to_csv(OUT_CSV, index=False, float_format="%.5f")
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--all", action="store_true", help=f"backfill {FIRST_SEASON} → current")
    parser.add_argument("--seasons", nargs=2, type=int, metavar=("FIRST", "LAST"))
    args = parser.parse_args()
    this = current_season()
    if args.all:
        seasons = list(range(FIRST_SEASON, this + 1))
    elif args.seasons:
        seasons = list(range(args.seasons[0], args.seasons[1] + 1))
    else:
        seasons = [this - 1, this]
    print(f"Weekly team summaries {seasons[0]}-{seasons[-1]}…")
    out = build(seasons)
    if out.empty:
        print("Nothing built; committed table untouched.")
        sys.exit(1)
    print(f"Wrote {len(out)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
