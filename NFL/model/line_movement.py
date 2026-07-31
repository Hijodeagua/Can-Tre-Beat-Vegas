"""Aggregate the per-snapshot odds CSVs into per-game opening vs closing
spread/total/moneyline.

The snapshot files in `data/odds_api_data_*.csv` have evolved through several
schemas. Two generations matter:

- **legacy** (Oct 2025 - Jan 2026, 332 files, the whole 2025 season): no
  ``Game ID`` column and no spread *points* — only the juice on each side.
  It does carry ``Avg Home/Away H2H Odds``, so moneyline movement is
  recoverable.
- **current** (Feb 2026 onward): adds ``Game ID`` and ``Avg Home Spread
  Points`` / ``Avg Total Points``.

An earlier version keyed exclusively on ``Game ID`` and therefore dropped
every legacy file, which is where the "we have zero historical games with
line movement" claim came from. It was only ever true of *spread* movement.
Keying on ``home|away|date`` instead recovers the 2025 season's moneyline
movement.

Output (one row per game):
    game_key, game_id_odds, sport, game_date, home_team, away_team,
    open_spread_home, close_spread_home, spread_move,
    open_total, close_total, total_move,
    open_h2h_home, close_h2h_home, h2h_move_home,
    open_novig_home, close_novig_home, novig_move_home,
    num_snapshots, first_seen, last_seen
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_GLOB = str(REPO_ROOT / "data" / "odds_api_data_*.csv")

# Columns we read when present. Only Timestamp Pulled + the two team names are
# strictly required; everything else is filled with NaN if the schema predates it.
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
REQUIRED = {"Timestamp Pulled", "Home Team", "Away Team"}
DATE_CANDIDATES = ["Date of Game (ET)", "Date of Game"]
OPTIONAL_NUMERIC = ["spread_home", "total", "h2h_home", "h2h_away"]


def american_to_prob(odds: pd.Series) -> pd.Series:
    """American odds -> implied probability (vig still included)."""
    o = pd.to_numeric(odds, errors="coerce")
    return pd.Series(np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0)), index=odds.index)


def _read_snapshot(path: str) -> pd.DataFrame | None:
    try:
        head = pd.read_csv(path, nrows=0)
    except Exception:
        return None
    cols = set(head.columns)
    if not REQUIRED.issubset(cols):
        return None

    date_col = next((c for c in DATE_CANDIDATES if c in cols), None)
    usecols = [c for c in CORE_COLS if c in cols]
    usecols += [date_col] if date_col else []
    usecols += ["League"] if "League" in cols else []

    try:
        df = pd.read_csv(path, usecols=usecols)
    except Exception:
        return None

    df = df.rename(columns=CORE_COLS)
    if date_col:
        df = df.rename(columns={date_col: "game_date"})
    else:
        df["game_date"] = pd.NaT
    if "League" not in df.columns:
        df["League"] = "NFL"  # legacy files are NFL-only
    if "game_id_odds" not in df.columns:
        df["game_id_odds"] = pd.NA
    for c in OPTIONAL_NUMERIC:
        if c not in df.columns:
            df[c] = np.nan
    return df


def load_all_snapshots(pattern: str = SNAPSHOT_GLOB) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(pattern)):
        if path.endswith("latest.csv"):
            continue  # duplicate of the newest timestamped file
        d = _read_snapshot(path)
        if d is not None and len(d):
            d["source_file"] = Path(path).name
            frames.append(d)
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out.dropna(subset=["ts", "home_team", "away_team"])

    # Stable key that works across both schema generations.
    out["game_key"] = (
        out["home_team"].str.strip()
        + "|" + out["away_team"].str.strip()
        + "|" + out["game_date"].dt.date.astype(str)
    )
    for c in OPTIONAL_NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def drop_in_play(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Keep only snapshots taken strictly before kickoff.

    The fetch job sometimes catches a game after it has started, and the API
    hands back live in-play prices — a 3-point favourite showing -2490 because
    it is already up two scores. Treating that as the closing line makes
    "the line moved toward the winner" look wildly predictive when it is just
    reading the scoreboard. Both ``Timestamp Pulled`` and the game date are
    Eastern, so the comparison is direct.
    """
    if snapshots.empty:
        return snapshots
    kickoff_known = snapshots["game_date"].notna()
    return snapshots[~kickoff_known | (snapshots["ts"] < snapshots["game_date"])].copy()


def aggregate_movement(
    snapshots: pd.DataFrame, league: str = "NFL", pregame_only: bool = True
) -> pd.DataFrame:
    """Collapse snapshots to one opening/closing row per game."""
    df = snapshots[snapshots["League"] == league].copy()
    if pregame_only:
        df = drop_in_play(df)
    if df.empty:
        return pd.DataFrame()

    hp = american_to_prob(df["h2h_home"])
    ap = american_to_prob(df["h2h_away"])
    df["novig_home"] = hp / (hp + ap)

    df = df.sort_values(["game_key", "ts"])
    grouped = df.groupby("game_key", as_index=False)

    open_cols = {"spread_home": "open_spread_home", "total": "open_total",
                 "h2h_home": "open_h2h_home", "h2h_away": "open_h2h_away",
                 "novig_home": "open_novig_home", "ts": "first_seen"}
    close_cols = {"spread_home": "close_spread_home", "total": "close_total",
                  "h2h_home": "close_h2h_home", "h2h_away": "close_h2h_away",
                  "novig_home": "close_novig_home", "ts": "last_seen"}

    first = grouped.first().rename(columns=open_cols)
    last = grouped.last().rename(columns=close_cols)
    counts = grouped.size().rename(columns={"size": "num_snapshots"})

    merged = first[["game_key", "game_id_odds", "home_team", "away_team", "game_date",
                    *open_cols.values()]].merge(
        last[["game_key", *close_cols.values()]], on="game_key"
    ).merge(counts, on="game_key")

    merged["spread_move"] = merged["close_spread_home"] - merged["open_spread_home"]
    merged["total_move"] = merged["close_total"] - merged["open_total"]
    merged["h2h_move_home"] = merged["close_h2h_home"] - merged["open_h2h_home"]
    merged["novig_move_home"] = merged["close_novig_home"] - merged["open_novig_home"]
    merged["sport"] = league
    return merged


# Full team name -> abbreviation, in both dialects we have to speak.
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

PFR_TO_NFLVERSE = {"GNB": "GB", "KAN": "KC", "LAR": "LA", "LVR": "LV",
                   "NWE": "NE", "NOR": "NO", "SFO": "SF", "TAM": "TB"}

NAME_TO_NFLVERSE = {name: PFR_TO_NFLVERSE.get(pfr, pfr) for name, pfr in NAME_TO_PFR.items()}


def add_abbrevs(movement: pd.DataFrame) -> pd.DataFrame:
    """Attach both PFR and nflverse abbreviations for downstream joins."""
    movement = movement.copy()
    movement["home_pfr"] = movement["home_team"].map(NAME_TO_PFR)
    movement["away_pfr"] = movement["away_team"].map(NAME_TO_PFR)
    movement["home_nfl"] = movement["home_team"].map(NAME_TO_NFLVERSE)
    movement["away_nfl"] = movement["away_team"].map(NAME_TO_NFLVERSE)
    return movement


# Backwards-compatible alias — the old name only attached PFR abbrevs.
add_pfr_abbrevs = add_abbrevs


def movement_by_nflverse_game(games: pd.DataFrame, movement: pd.DataFrame) -> pd.DataFrame:
    """Join aggregated movement onto nflverse ``game_id`` via team + date.

    Kickoffs after 8pm ET roll past midnight UTC in some snapshot schemas, so
    the join also tries the day before the recorded game date.
    """
    m = add_abbrevs(movement)
    m["gameday"] = m["game_date"].dt.normalize()
    g = games[["game_id", "season", "week", "gameday", "home_team", "away_team"]].copy()
    g["gameday"] = pd.to_datetime(g["gameday"]).dt.normalize()

    keys = ["gameday", "home_nfl", "away_nfl"]
    exact = g.merge(m, left_on=["gameday", "home_team", "away_team"],
                    right_on=keys, how="inner")

    unmatched = g[~g["game_id"].isin(exact["game_id"])]
    shifted = m.assign(gameday=m["gameday"] - pd.Timedelta(days=1))
    lagged = unmatched.merge(shifted, left_on=["gameday", "home_team", "away_team"],
                             right_on=keys, how="inner")
    return pd.concat([exact, lagged], ignore_index=True)


if __name__ == "__main__":
    snaps = load_all_snapshots()
    print(f"Loaded {len(snaps)} snapshot rows from {snaps['ts'].min()} to {snaps['ts'].max()}")
    move = add_abbrevs(aggregate_movement(snaps))
    print(f"{len(move)} unique NFL games aggregated")
    print(f"  with spread points: {move['open_spread_home'].notna().sum()}")
    print(f"  with moneyline:     {move['open_h2h_home'].notna().sum()}")
    print()
    print(move[["game_key", "open_novig_home", "close_novig_home", "novig_move_home",
                "open_spread_home", "close_spread_home", "num_snapshots"]]
          .head(10).to_string())
    print()
    print("Movement summary:")
    print(move[["spread_move", "total_move", "novig_move_home", "num_snapshots"]]
          .describe().to_string())
