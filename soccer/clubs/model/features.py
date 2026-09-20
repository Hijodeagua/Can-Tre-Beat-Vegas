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


def wages_available() -> bool:
    """Whether any club anywhere has a real wage bill.

    The market-value uploads carry a wage column that no source has
    filled yet, so `wage_z` is 0.0 on every row. For the model that is
    harmless — a constant column a learner ignores, and the differential
    it trains on is genuinely 0 — but published next to a club it would
    read as "average wage bill" when it means "nobody knows". The match
    card asks this and omits the metric while it is False; it lights up on
    its own when a wage upload lands.
    """
    if not values_available():
        return False
    return bool((_load_value_z()["wage_z"] != 0).any())


def _z_within(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby(["league", "season"])[col]
    std = g.transform("std").replace(0.0, np.nan)
    return ((df[col] - g.transform("mean")) / std).fillna(0.0)


def _load_transfer_z() -> pd.DataFrame:
    t = pd.read_csv(TRANSFERS_CSV)
    t["spend_z"] = _z_within(t, "spend_eur_m")
    t["net_z"] = _z_within(t, "net_eur_m")
    return t[["league", "season", "club", "spend_z", "net_z"]]


def _load_values_raw() -> pd.DataFrame:
    """Concatenate every `values_<season>.csv` upload, canonicalized, with
    every column the files carry — squad value/wage plus whatever of the
    squad-composition columns (squad_size, avg_age, foreigners,
    avg_value_eur_m) a given season/league has been backfilled with."""
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
    return pd.concat(frames, ignore_index=True)


def load_market_values_raw() -> pd.DataFrame:
    """Public entry point for exporters (not the model): the full raw
    squad-economics table, canonicalized, un-z-scored. Used to build
    display-only cross-league summaries (`daily/export_site.py`)."""
    return _load_values_raw()


def _load_value_z() -> pd.DataFrame:
    """Squad economics per (league, season, club): the within-league-season
    z-scores the differentials are built from, **and** the raw euro
    figures they were computed from.

    Both are kept because they answer different questions and neither
    substitutes for the other. The z says where a club sits among the
    clubs it actually plays — the only fair comparison, since a £900m
    Premier League squad and a €900m Bundesliga squad buy very different
    league positions. The raw euro figure says how big the club is in
    absolute terms, which the z destroys by construction: the richest
    club in every league scores about the same z whether it is Manchester
    City or Feyenoord. With the league one-hots and `season_idx` also in
    the feature set, a learner can use the raw number in context and use
    the z where context is all that matters.
    """
    v = _load_values_raw()
    v["value_z"] = _z_within(v, "squad_value_eur_m")
    if "wage_bill_eur_m" in v.columns and v["wage_bill_eur_m"].notna().any():
        v["wage_z"] = _z_within(v, "wage_bill_eur_m")
    else:
        v["wage_z"] = 0.0
    cols = ["league", "season", "club", "value_z", "wage_z", "squad_value_eur_m"]
    if "wage_bill_eur_m" in v.columns:
        cols.append("wage_bill_eur_m")
    return v[cols]


def _attach_side(history: pd.DataFrame, table: pd.DataFrame,
                 cols: list[str]) -> pd.DataFrame:
    """Join per-side columns with no differential taken — for figures
    where a missing side cannot honestly be filled with a zero (see
    SIDE_RAW_FEATURES)."""
    keep = [c for c in cols if c in table.columns]
    if not keep:
        return history
    for side in ("home", "away"):
        renames = {c: f"{side}_{c}" for c in keep}
        history = history.merge(
            table[["league", "season", "club"] + keep].rename(
                columns={"club": f"{side}_team", **renames}),
            on=["league", "season", f"{side}_team"],
            how="left",
        )
    return history


def _attach_diff(history: pd.DataFrame, table: pd.DataFrame,
                 mapping: dict[str, str], keep_sides: bool = False) -> pd.DataFrame:
    """Join a (league, season, club) -> z table on both sides of each match
    and write home-minus-away differentials. `mapping` is {out_col: z_col}.

    `keep_sides` leaves the joined `home_<z>` / `away_<z>` columns in place
    instead of dropping them once the differential is written — a published
    match card wants to say which side is the expensive one, not only that
    one of them is."""
    # Only the z columns this mapping names — the table also carries the
    # raw euro figures now, and merging those here would suffix them into
    # the frame as `*_x` / `*_y` on both sides of the join.
    needed = ["league", "season", "club"] + list(mapping.values())
    slim = table[[c for c in needed if c in table.columns]]
    for side in ("home", "away"):
        renames = {z: f"{side}_{z}" for z in mapping.values()}
        history = history.merge(
            slim.rename(columns={"club": f"{side}_team", **renames}),
            on=["league", "season", f"{side}_team"],
            how="left",
        )
    for out_col, z in mapping.items():
        history[out_col] = (
            history[f"home_{z}"].fillna(0.0) - history[f"away_{z}"].fillna(0.0)
        )
        if not keep_sides:
            history = history.drop(columns=[f"home_{z}", f"away_{z}"])
    return history


# The per-side economics columns `keep_sides=True` leaves behind.
SIDE_FEATURES = [f"{side}_{z}" for z in ("spend_z", "net_z", "value_z", "wage_z")
                 for side in ("home", "away")]

# Raw euro figures carried per side only — never differenced.
#
# The z differentials fill a missing side with 0.0, which for a z means
# "assume league-average", a defensible shrink. The same trick on a raw
# euro figure would mean "assume this club is worth nothing", and the
# resulting difference would be the other club's entire squad value
# masquerading as a gap. So these stay per-side and stay NaN when absent,
# and the learner's imputer handles them.
SIDE_RAW_FEATURES = [f"{side}_{c}" for c in ("squad_value_eur_m",)
                     for side in ("home", "away")]


def attach_features(history: pd.DataFrame,
                    keep_sides: bool = False) -> pd.DataFrame:
    """Add all squad-economics differentials to an Elo history table.
    UEFA rows (league "uefa:…") get zeros — the z-tables are league-keyed.

    `keep_sides` additionally leaves each side's own z-scores on the frame
    (see `_attach_diff`); it is off by default so the training frame keeps
    exactly the columns it always had."""
    history = history.copy()
    if transfers_available():
        history = _attach_diff(
            history, _load_transfer_z(),
            {"spend_diff_z": "spend_z", "net_diff_z": "net_z"},
            keep_sides=keep_sides,
        )
    else:
        history[["spend_diff_z", "net_diff_z"]] = 0.0
        if keep_sides:
            history[["home_spend_z", "away_spend_z",
                     "home_net_z", "away_net_z"]] = float("nan")
    if values_available():
        values = _load_value_z()
        history = _attach_diff(
            history, values,
            {"value_diff_z": "value_z", "wage_diff_z": "wage_z"},
            keep_sides=keep_sides,
        )
        if keep_sides:
            history = _attach_side(history, values, ["squad_value_eur_m"])
    else:
        history[["value_diff_z", "wage_diff_z"]] = 0.0
        if keep_sides:
            history[["home_value_z", "away_value_z",
                     "home_wage_z", "away_wage_z"]] = float("nan")
    if keep_sides:
        for col in SIDE_RAW_FEATURES:
            if col not in history.columns:
                history[col] = float("nan")
    return history
