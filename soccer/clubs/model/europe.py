"""
Cross-league Elo glue: replay the five league histories *and* the UEFA club
competitions (CL / EL / Conference) in one chronological stream.

League matches update their league's pool exactly as in `elo.py`. A UEFA
match between two tracked clubs moves rating points *between* the two
league pools: the exchange is zero-sum, K is the mean of the two leagues'
tuned Ks scaled by UEFA_WEIGHT, and home advantage is the home club's
league value (zero for the neutral-venue finals). UEFA matches involving a
club outside the five leagues (Porto, Ajax, …) carry no rating for the
opponent and are skipped.

The glue's effect is deliberately modest — ~65 cross-league matches a
season against ~1,750 league matches — but it is the only competitive
signal linking the pools, so cross-league Elo comparisons mean something
with it and nothing without it.

Usage (module check):
    python -m soccer.clubs.model.europe
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from soccer.clubs.data.leagues import LEAGUES
from soccer.clubs.model.elo import (
    ClubEloEngine,
    expected_score,
    load_results,
    mov_multiplier,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UEFA_CSV = DATA_DIR / "uefa_results.csv"

# Scales K for cross-league matches. Validated on the 2024-25+ league
# holdout: log loss 0.99058 at 0.75 vs 0.99094 unglued, with an interior
# optimum across {0.25, 0.5, 0.75, 1.0, 1.5}.
UEFA_WEIGHT = 0.75


def load_uefa() -> pd.DataFrame:
    """Committed UEFA rows with both clubs tracked; empty frame if absent."""
    if not UEFA_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(UEFA_CSV)
    df = df[(df["home_league"].notna()) & (df["away_league"].notna())]
    df = df[(df["home_league"] != "") & (df["away_league"] != "")]
    return df.reset_index(drop=True)


def cross_update(
    engines: Dict[str, ClubEloEngine], row: pd.Series, weight: float = UEFA_WEIGHT
) -> dict:
    """One UEFA match between two tracked clubs; returns the feature record."""
    eh, ea = engines[row["home_league"]], engines[row["away_league"]]
    for e in (eh, ea):
        if row["season"] != e.current_season:
            e._roll_season(row["season"])

    home, away = row["home_team"], row["away_team"]
    r_home, r_away = eh.get(home), ea.get(away)
    adv = 0.0 if bool(row["neutral"]) else eh.home_advantage

    exp_home = expected_score(r_home + adv, r_away)
    goal_diff = int(row["home_score"]) - int(row["away_score"])
    actual = 1.0 if goal_diff > 0 else (0.0 if goal_diff < 0 else 0.5)

    k = weight * (eh.k + ea.k) / 2.0
    delta = k * mov_multiplier(goal_diff) * (actual - exp_home)
    eh.ratings[home] = r_home + delta
    ea.ratings[away] = r_away - delta
    for e, t in ((eh, home), (ea, away)):
        e.matches_played[t] = e.matches_played.get(t, 0) + 1

    return {
        "date": row["date"],
        "season": row["season"],
        "league": f"uefa:{row['competition']}",
        "home_team": home,
        "away_team": away,
        "elo_home_pre": r_home,
        "elo_away_pre": r_away,
        "elo_gap": (r_home + adv) - r_away,
        "exp_home": exp_home,
        "actual_home": actual,
        "outcome": "H" if goal_diff > 0 else ("A" if goal_diff < 0 else "D"),
        "home_score": int(row["home_score"]),
        "away_score": int(row["away_score"]),
    }


def run_all_european(
    df: Optional[pd.DataFrame] = None,
    uefa: Optional[pd.DataFrame] = None,
    end: Optional[str] = None,
    weight: float = UEFA_WEIGHT,
) -> tuple[Dict[str, ClubEloEngine], pd.DataFrame]:
    """League + UEFA replay. Returns {league: engine} and the stacked
    per-match records (league rows carry their league key, UEFA rows
    "uefa:<comp>" — filter on that when training the league outcome model)."""
    if df is None:
        df = load_results()
    if uefa is None:
        uefa = load_uefa()

    engines = {league: ClubEloEngine.for_league(league) for league in LEAGUES}

    frames = [df.assign(_kind="league")]
    if len(uefa):
        frames.append(uefa.assign(_kind="uefa"))
    stream = pd.concat(frames, ignore_index=True).sort_values(
        ["date", "_kind"], kind="stable"  # same-day: league first, then UEFA
    )
    if end:
        stream = stream[stream["date"] < end]

    records = []
    for _, row in stream.iterrows():
        if row["_kind"] == "league":
            records.append(engines[row["league"]].update(row))
        else:
            records.append(cross_update(engines, row, weight))
    return engines, pd.DataFrame(records)


if __name__ == "__main__":
    engines, history = run_all_european()
    n_uefa = int(history["league"].str.startswith("uefa:").sum())
    print(f"Processed {len(history)} matches ({n_uefa} UEFA cross-league)\n")
    print("Top 10 clubs across all leagues (glued ratings):")
    tables = []
    for league, engine in engines.items():
        latest = history[history["league"] == league]["season"].max()
        t = engine.table(season=latest)
        t["league"] = league
        tables.append(t)
    top = pd.concat(tables).sort_values("elo", ascending=False).head(10)
    print(top[["team", "league", "elo"]].round(1).to_string(index=False))
