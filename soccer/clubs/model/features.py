"""
Squad-economics features for the club outcome model: transfer spend (from
the committed Transfermarkt aggregates) and squad market value / wage bill
(from optional per-season uploads).

Mirrors the international model's squad layer (`soccer/model/squad.py`):
every feature is a home-minus-away differential, z-scored within its
league-season so a EUR-inflation era or a rich league doesn't leak scale,
and 0-imputed when the underlying data isn't there — the outcome model
degrades gracefully to Elo-only.

Feature columns attached to a history table:

- `spend_diff_z`  — gross transfer spend, this season's windows
- `net_diff_z`    — net spend (spend − sales)
- `value_diff_z`  — squad market value (needs `data/market_values/` uploads)
- `wage_diff_z`   — wage bill (same uploads, optional column)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from soccer.clubs.data.leagues import canonical

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRANSFERS_CSV = DATA_DIR / "club_season_transfers.csv"
VALUES_DIR = DATA_DIR / "market_values"

TRANSFER_FEATURES = ["spend_diff_z", "net_diff_z"]
VALUE_FEATURES = ["value_diff_z", "wage_diff_z"]
ALL_FEATURES = TRANSFER_FEATURES + VALUE_FEATURES


def transfers_available() -> bool:
    return TRANSFERS_CSV.exists()


def values_available() -> bool:
    return VALUES_DIR.exists() and any(VALUES_DIR.glob("values_*.csv"))


def _z_within(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby(["league", "season"])[col]
    std = g.transform("std").replace(0.0, np.nan)
    return ((df[col] - g.transform("mean")) / std).fillna(0.0)


def _load_transfer_z() -> pd.DataFrame:
    t = pd.read_csv(TRANSFERS_CSV)
    t["spend_z"] = _z_within(t, "spend_eur_m")
    t["net_z"] = _z_within(t, "net_eur_m")
    return t[["league", "season", "club", "spend_z", "net_z"]]


def _load_value_z() -> pd.DataFrame:
    frames = []
    for path in sorted(VALUES_DIR.glob("values_*.csv")):
        season = path.stem.replace("values_", "")
        df = pd.read_csv(path)
        df["season"] = season
        # Uploads carry whatever spelling Transfermarkt shows for that
        # season — same posture as fetch_transfers.py: canonicalize on
        # load rather than requiring pre-canonicalized names in the file.
        df["club"] = df.apply(lambda r: canonical(r["league"], r["club"]), axis=1)
        frames.append(df)
    v = pd.concat(frames, ignore_index=True)
    v["value_z"] = _z_within(v, "squad_value_eur_m")
    if "wage_bill_eur_m" in v.columns and v["wage_bill_eur_m"].notna().any():
        v["wage_z"] = _z_within(v, "wage_bill_eur_m")
    else:
        v["wage_z"] = 0.0
    return v[["league", "season", "club", "value_z", "wage_z"]]


def _attach_diff(history: pd.DataFrame, table: pd.DataFrame,
                 mapping: dict[str, str]) -> pd.DataFrame:
    """Join a (league, season, club) -> z table on both sides of each match
    and write home-minus-away differentials. `mapping` is {out_col: z_col}."""
    for side in ("home", "away"):
        renames = {z: f"{side}_{z}" for z in mapping.values()}
        history = history.merge(
            table.rename(columns={"club": f"{side}_team", **renames}),
            on=["league", "season", f"{side}_team"],
            how="left",
        )
    for out_col, z in mapping.items():
        history[out_col] = (
            history[f"home_{z}"].fillna(0.0) - history[f"away_{z}"].fillna(0.0)
        )
        history = history.drop(columns=[f"home_{z}", f"away_{z}"])
    return history


def attach_features(history: pd.DataFrame) -> pd.DataFrame:
    """Add all squad-economics differentials to an Elo history table.
    UEFA rows (league "uefa:…") get zeros — the z-tables are league-keyed."""
    history = history.copy()
    if transfers_available():
        history = _attach_diff(
            history, _load_transfer_z(),
            {"spend_diff_z": "spend_z", "net_diff_z": "net_z"},
        )
    else:
        history[["spend_diff_z", "net_diff_z"]] = 0.0
    if values_available():
        history = _attach_diff(
            history, _load_value_z(),
            {"value_diff_z": "value_z", "wage_diff_z": "wage_z"},
        )
    else:
        history[["value_diff_z", "wage_diff_z"]] = 0.0
    return history
