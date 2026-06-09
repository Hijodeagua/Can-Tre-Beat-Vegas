"""
Squad-strength features from FIFA player ratings.

Two decoupled metrics per squad, vintage-matched to the match date
(see soccer/SPEC.md):

- depth: average overall rating of the squad's top 25 players
- star:  average overall rating of the squad's top 5 players

Both are z-scored across all squads within an edition so the model
coefficients (α for depth, β for star) read in standard-deviation units.

Ratings files are manual uploads: soccer/data/fifa_ratings/fifa_<year>.csv
with columns team,player,overall. The oldest practical edition is FIFA 07;
matches earlier than the oldest edition are imputed from it.
"""

import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

RATINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "fifa_ratings"

DEPTH_N = 25
STAR_N = 5


def load_editions() -> Dict[int, pd.DataFrame]:
    """edition year -> squad aggregate table with z-scored depth/star."""
    editions: Dict[int, pd.DataFrame] = {}
    if not RATINGS_DIR.exists():
        return editions
    for path in sorted(RATINGS_DIR.glob("fifa_*.csv")):
        m = re.match(r"fifa_(\d{4})\.csv", path.name)
        if not m:
            continue
        year = int(m.group(1))
        df = pd.read_csv(path)
        if not {"team", "player", "overall"}.issubset(df.columns):
            continue
        df = df.sort_values("overall", ascending=False)
        agg = df.groupby("team")["overall"].agg(
            depth=lambda s: s.head(DEPTH_N).mean(),
            star=lambda s: s.head(STAR_N).mean(),
        )
        # z-score across all squads in this edition
        for col in ("depth", "star"):
            agg[f"{col}_z"] = (agg[col] - agg[col].mean()) / agg[col].std(ddof=0)
        editions[year] = agg
    return editions


def edition_for_date(editions: Dict[int, pd.DataFrame], date: str) -> Optional[int]:
    """Edition published closest to but before the match date; matches that
    predate the oldest edition are imputed from it (FIFA 07 backfill)."""
    if not editions:
        return None
    year = int(str(date)[:4])
    eligible = [y for y in editions if y <= year]
    return max(eligible) if eligible else min(editions)


def attach_squad_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add depth_diff_z / star_diff_z columns (home minus away). Squads or
    editions without ratings get 0 — the model degrades to Elo-only."""
    editions = load_editions()
    depth, star = [], []
    for _, row in matches.iterrows():
        year = edition_for_date(editions, row["date"])
        d = s = 0.0
        if year is not None:
            table = editions[year]
            home, away = row["home_team"], row["away_team"]
            if home in table.index and away in table.index:
                d = table.at[home, "depth_z"] - table.at[away, "depth_z"]
                s = table.at[home, "star_z"] - table.at[away, "star_z"]
        depth.append(d)
        star.append(s)
    out = matches.copy()
    out["depth_diff_z"] = depth
    out["star_diff_z"] = star
    return out


def ratings_available() -> bool:
    return bool(load_editions())
