"""Normalise raw Sports Reference CFB exports into tidy aggregate files.

Three raw shapes come out of CFR's *Share & Export -> Get table as CSV*:

- **Schedule pages** (``{YEAR}games.csv`` or ``cfb_schedule_{year}.csv``):
  Winner/Loser format — the winning team is listed first, with a location
  marker between the two teams (blank = winner was home, ``@`` = winner was
  on the road, ``N`` = neutral site). This module reconstructs home/away,
  which is the transform `CFB/DATA_PULL_PLAN.md` §2.1 says to test before
  anything else.
- **Scoring pages** (``pts_stats*.csv``): per team-season offensive scoring —
  TDs, XPs, FGs, points for/against.
- **Defense pages** (``def_stats*.csv``): the same schema mirrored — TDs and
  kicks *allowed*.

Outputs, all under ``data/college_football/agg/``:

- ``cfb_games.csv`` — one row per game, home perspective, same column names
  as the NFL spine (``data/schedules/nflverse_games.csv``) so the Elo engine
  runs on either sport unchanged.
- ``cfb_offense_team_season.csv``
- ``cfb_defense_team_season.csv``

Usage
    python3 -m CFB.ingest            # reads data/college_football/raw/
    python3 -m CFB.ingest --validate # also print the home-win-rate check
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "college_football" / "raw"
AGG_DIR = REPO_ROOT / "data" / "college_football" / "agg"

# "(4) Ohio State" -> rank 4, "Ohio State". Rank is the AP rank at kickoff.
RANK_RE = re.compile(r"^\((\d+)\)\s+(.*)$")


def _split_rank(name: str) -> tuple[float, str]:
    m = RANK_RE.match(str(name).strip())
    if m:
        return float(m.group(1)), m.group(2)
    return np.nan, str(name).strip()


def _season_of(date: pd.Timestamp) -> int:
    """Bowl games in January belong to the prior calendar year's season."""
    return date.year if date.month >= 6 else date.year - 1


def _game_type(note: str | float) -> str:
    if pd.isna(note) or not str(note).strip():
        return "REG"
    note = str(note)
    if "Championship" in note:
        return "CCG"
    return "BOWL"


def parse_schedule(path: Path) -> pd.DataFrame:
    """One CFR schedule export -> home-perspective games."""
    df = pd.read_csv(path)
    # The unnamed column between the two teams is the location marker.
    loc_col = next(c for c in df.columns if c.startswith("Unnamed"))
    df = df.rename(
        columns={loc_col: "loc", "Pts": "w_pts", "Pts.1": "l_pts"}
    )
    # CFR exports repeat the header row mid-table; drop those and any
    # cancelled games that carry no score.
    df = df[df["Rk"].astype(str) != "Rk"]
    df = df.dropna(subset=["w_pts", "l_pts"])

    rows = []
    for r in df.itertuples(index=False):
        winner_rank, winner = _split_rank(r.Winner)
        loser_rank, loser = _split_rank(r.Loser)
        w_pts, l_pts = float(r.w_pts), float(r.l_pts)

        loc = "" if pd.isna(r.loc) else str(r.loc).strip()
        neutral = loc == "N"
        if neutral or loc == "":  # blank = the winner was at home
            home, away = winner, loser
            home_rank, away_rank = winner_rank, loser_rank
            home_score, away_score = w_pts, l_pts
        else:  # '@' — the winner was the road team
            home, away = loser, winner
            home_rank, away_rank = loser_rank, winner_rank
            home_score, away_score = l_pts, w_pts

        date = pd.to_datetime(str(r.Date), format="%b %d %Y")
        rows.append(
            {
                "season": _season_of(date),
                "week": int(r.Wk),
                "gameday": date.date().isoformat(),
                "game_type": _game_type(r.Notes),
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "home_rank": home_rank,
                "away_rank": away_rank,
                "location": "Neutral" if neutral else "Home",
                "notes": r.Notes if pd.notna(r.Notes) else "",
            }
        )

    out = pd.DataFrame(rows)
    out["game_id"] = (
        out["season"].astype(str)
        + "_"
        + out["week"].astype(str).str.zfill(2)
        + "_"
        + out["away_team"].str.replace(r"\W", "", regex=True)
        + "_"
        + out["home_team"].str.replace(r"\W", "", regex=True)
    )
    return out


def _read_positional(path: Path) -> pd.DataFrame:
    """Read a CFR scoring/defense export by position, not by name.

    Both files carry two columns literally named ``TD``; pandas mangles the
    second to ``TD.1``. Reading positionally sidesteps that and the schema
    drift between the two pages (the offense page has XP%, FG%, 2PM extra).
    """
    df = pd.read_csv(path)
    df = df[df["Rk"].astype(str) != "Rk"]  # embedded repeat headers
    return df


def parse_scoring(path: Path, side: str) -> pd.DataFrame:
    """side='off' for the points page, 'def' for the defense page."""
    df = _read_positional(path)
    suffix = "" if side == "off" else "_allowed"
    out = pd.DataFrame(
        {
            "season": df["Season"].astype(int),
            "team": df["Team"].astype(str).str.strip(),
            "games": df["G"].astype(int),
            "wins": df["W"].astype(int),
            "losses": df["L"].astype(int),
            "win_pct": df["W-L%"].astype(float),
            "pts_for": df["Pts"].astype(float),
            "pts_against": df["PtsO"].astype(float),
            "pt_diff": df["PtDif"].astype(float),
            f"td{suffix}": df["TD"].astype(float),
            f"xpa{suffix}": df["XPA"].astype(float),
            f"xpm{suffix}": df["XPM"].astype(float),
            f"fga{suffix}": df["FGA"].astype(float),
            f"fgm{suffix}": df["FGM"].astype(float),
            f"safeties{suffix}": df["Sfty"].astype(float),
        }
    )
    # Per-game rates — the seasons in these pulls range from 11 to 16 games,
    # so totals are not comparable across rows.
    out[f"td{suffix}_pg"] = (out[f"td{suffix}"] / out["games"]).round(2)
    if side == "off":
        out["pts_pg"] = (out["pts_for"] / out["games"]).round(2)
    else:
        out["pts_allowed_pg"] = (out["pts_against"] / out["games"]).round(2)

    # Known corruption in the CFR export: 2015 TCU shows XPA=4 against
    # XPM=60 (an XP% of 1500). Null the pair rather than guessing.
    bad = (out[f"xpm{suffix}"] > out[f"xpa{suffix}"]) & (out[f"xpa{suffix}"] > 0)
    if bad.any():
        out.loc[bad, [f"xpa{suffix}", f"xpm{suffix}"]] = np.nan
    return out.sort_values(["season", "team"]).reset_index(drop=True)


def build_all(validate: bool = False) -> dict[str, pd.DataFrame]:
    AGG_DIR.mkdir(parents=True, exist_ok=True)

    sched_files = sorted(
        p for p in RAW_DIR.glob("*.csv") if "games" in p.name or "schedule" in p.name
    )
    games = pd.concat([parse_schedule(p) for p in sched_files], ignore_index=True)
    games = games.sort_values(["gameday", "home_team"]).reset_index(drop=True)
    games.to_csv(AGG_DIR / "cfb_games.csv", index=False)

    off_files = sorted(RAW_DIR.glob("*pts_stats*.csv"))
    off = pd.concat([parse_scoring(p, "off") for p in off_files], ignore_index=True)
    off.to_csv(AGG_DIR / "cfb_offense_team_season.csv", index=False)

    def_files = sorted(RAW_DIR.glob("*def_stats*.csv"))
    dfn = pd.concat([parse_scoring(p, "def") for p in def_files], ignore_index=True)
    dfn.to_csv(AGG_DIR / "cfb_defense_team_season.csv", index=False)

    if validate:
        non_neutral = games[games["location"] == "Home"]
        hw = (non_neutral["home_score"] > non_neutral["away_score"]).mean()
        print(f"games: {len(games)}  ({games['season'].min()}-{games['season'].max()})")
        print(f"neutral-site games: {(games['location'] == 'Neutral').sum()}")
        print(f"home win rate (non-neutral): {hw:.3f}  <- expect 0.57-0.60")
        print(f"offense rows: {len(off)}, defense rows: {len(dfn)}")

    return {"games": games, "offense": off, "defense": dfn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    build_all(validate=args.validate)


if __name__ == "__main__":
    main()
