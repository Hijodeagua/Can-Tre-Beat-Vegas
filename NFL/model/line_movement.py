"""Aggregate the per-snapshot odds CSVs into per-game opening vs closing
spread/total/h2h.

The snapshot files in `data/odds_api_data_*.csv` have evolved through
several schemas (different column counts, with/without a League column).
This module reads only the small subset of columns that have remained
stable and is tolerant of either schema.

Output (one row per game):
    game_id_odds, sport, game_date, home_team, away_team,
    open_spread_home, close_spread_home, spread_move,
    open_total, close_total, total_move,
    open_h2h_home, close_h2h_home, h2h_move_home,
    num_snapshots, first_seen, last_seen
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_GLOB = str(REPO_ROOT / "data" / "odds_api_data_*.csv")

CORE_COLS = {
    "Game ID": "game_id_odds",
    "Timestamp Pulled": "ts",
    "Home Team": "home_team",
    "Away Team": "away_team",
    "Avg Home Spread Points": "spread_home",
    "Avg Total Points": "total",
    "Avg Home H2H Odds": "h2h_home",
    "Avg Away H2H Odds": "h2h_away",
}
DATE_CANDIDATES = ["Date of Game (ET)", "Date of Game"]


def _read_snapshot(path: str) -> pd.DataFrame | None:
    try:
        head = pd.read_csv(path, nrows=0)
    except Exception:
        return None
    cols = list(head.columns)
    needed = [c for c in CORE_COLS if c in cols]
    if "Game ID" not in needed or "Timestamp Pulled" not in needed:
        return None
    date_col = next((c for c in DATE_CANDIDATES if c in cols), None)
    usecols = needed + ([date_col] if date_col else []) + (["League"] if "League" in cols else [])
    df = pd.read_csv(path, usecols=usecols)
    df = df.rename(columns=CORE_COLS)
    if date_col:
        df = df.rename(columns={date_col: "game_date"})
    else:
        df["game_date"] = pd.NaT
    if "League" not in df.columns:
        df["League"] = "NFL"  # legacy files are NFL-only
    return df


def load_all_snapshots() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(SNAPSHOT_GLOB)):
        d = _read_snapshot(path)
        if d is not None and len(d):
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out.dropna(subset=["ts", "game_id_odds"])
    return out


def aggregate_movement(snapshots: pd.DataFrame, league: str = "NFL") -> pd.DataFrame:
    df = snapshots[snapshots["League"] == league].copy()
    df = df.sort_values(["game_id_odds", "ts"])

    grouped = df.groupby("game_id_odds", as_index=False)
    first = grouped.first().rename(columns={
        "spread_home": "open_spread_home",
        "total": "open_total",
        "h2h_home": "open_h2h_home",
        "h2h_away": "open_h2h_away",
        "ts": "first_seen",
    })
    last = grouped.last().rename(columns={
        "spread_home": "close_spread_home",
        "total": "close_total",
        "h2h_home": "close_h2h_home",
        "h2h_away": "close_h2h_away",
        "ts": "last_seen",
    })
    counts = grouped.size().rename(columns={"size": "num_snapshots"})

    merged = first[["game_id_odds", "home_team", "away_team", "game_date",
                    "open_spread_home", "open_total", "open_h2h_home",
                    "open_h2h_away", "first_seen"]].merge(
        last[["game_id_odds", "close_spread_home", "close_total",
              "close_h2h_home", "close_h2h_away", "last_seen"]],
        on="game_id_odds",
    ).merge(counts, on="game_id_odds")

    merged["spread_move"] = merged["close_spread_home"] - merged["open_spread_home"]
    merged["total_move"] = merged["close_total"] - merged["open_total"]
    merged["h2h_move_home"] = merged["close_h2h_home"] - merged["open_h2h_home"]
    return merged


# Map full team name -> PFR abbrev (only the 32 NFL teams).
NAME_TO_PFR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GNB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KAN", "Las Vegas Raiders": "LVR",
    "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NWE", "New Orleans Saints": "NOR",
    "New York Giants": "NYG", "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SFO", "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TAM", "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


def add_pfr_abbrevs(movement: pd.DataFrame) -> pd.DataFrame:
    movement = movement.copy()
    movement["home_pfr"] = movement["home_team"].map(NAME_TO_PFR)
    movement["away_pfr"] = movement["away_team"].map(NAME_TO_PFR)
    return movement


if __name__ == "__main__":
    snaps = load_all_snapshots()
    print(f"Loaded {len(snaps)} snapshot rows from "
          f"{snaps['ts'].min()} to {snaps['ts'].max()}")
    move = aggregate_movement(snaps)
    move = add_pfr_abbrevs(move)
    print(f"{len(move)} unique NFL games aggregated")
    print(move[["game_id_odds", "home_pfr", "away_pfr", "game_date",
                "open_spread_home", "close_spread_home", "spread_move",
                "open_total", "close_total", "total_move",
                "num_snapshots"]].head(10).to_string())
    print()
    print("Movement summary:")
    print(move[["spread_move", "total_move", "h2h_move_home", "num_snapshots"]]
          .describe().to_string())
