"""
Convert raw sofifa/Kaggle player dumps into the pipeline's ratings schema:
soccer/data/fifa_ratings/fifa_<release_year>.csv with columns
team,player,overall (team = national side via player nationality).

Handles the two common dump layouts:
- stefanoleone992 style: nationality, short_name, overall
- "male2" style:         Nationality, Name, OVA

Edition naming: the file year is the RELEASE year (FIFA 15 -> fifa_2014.csv,
released Sept 2014), so vintage matching picks the edition actually on
shelves at match time.

Usage:
    python -m soccer.model.convert_ratings <raw.csv> <release_year>
"""

import sys
from pathlib import Path

import pandas as pd

RATINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "fifa_ratings"

# FIFA nationality label -> team name used in soccer/data/results.csv
NAME_MAP = {
    "Antigua & Barbuda": "Antigua and Barbuda",
    "Antigua &amp; Barbuda": "Antigua and Barbuda",
    "Bosnia Herzegovina": "Bosnia and Herzegovina",
    "Brunei Darussalam": "Brunei",
    "Central African Rep.": "Central African Republic",
    "China PR": "China",
    "Chinese Taipei": "Taiwan",
    "Curacao": "Curaçao",
    "FYR Macedonia": "North Macedonia",
    "Guinea Bissau": "Guinea-Bissau",
    "Korea DPR": "North Korea",
    "Korea Republic": "South Korea",
    "St Kitts Nevis": "Saint Kitts and Nevis",
    "St Lucia": "Saint Lucia",
    "São Tomé &amp; Príncipe": "São Tomé and Príncipe",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Trinidad &amp; Tobago": "Trinidad and Tobago",
}

COLUMN_LAYOUTS = [
    {"team": "nationality", "player": "short_name", "overall": "overall"},
    {"team": "nationality_name", "player": "short_name", "overall": "overall"},
    {"team": "Nationality", "player": "Name", "overall": "OVA"},
    {"team": "country_name", "player": "name", "overall": "overall_rating"},
]


# Fallback aliases for arbitrary author dumps (FC25/FC26 from various Kaggle
# uploaders). Checked case-insensitively, in order, when no exact layout matches.
FIELD_ALIASES = {
    "team": ["nationality", "nationality_name", "nation", "country", "country_name",
             "Nationality", "Country"],
    "player": ["short_name", "long_name", "name", "player_name", "full_name",
               "Name", "Player"],
    "overall": ["overall", "overall_rating", "OVA", "OVR", "ovr", "rating",
                "Overall", "Rating"],
}


def _detect_layout(columns) -> dict:
    layout = next(
        (m for m in COLUMN_LAYOUTS if set(m.values()).issubset(columns)), None
    )
    if layout:
        return layout

    # Fallback: alias-match each field (case-insensitive) against the header.
    lower = {c.lower(): c for c in columns}
    guess = {}
    for field, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower:
                guess[field] = lower[alias.lower()]
                break
    if len(guess) == 3:
        print(f"  (auto-detected columns: {guess})")
        return guess

    raise SystemExit(
        f"Unrecognized columns; could not find team/player/overall.\n"
        f"Got: {sorted(columns)}\n"
        f"Tell me which columns hold nationality, name, and overall rating."
    )


def convert(
    raw_path: str, release_year: int, game_version: int | None = None
) -> Path:
    # Read only the header first: the combined FIFA 23 dump is multi-GB, so we
    # never load the whole thing — just the columns we need, streamed in chunks.
    header = pd.read_csv(raw_path, nrows=0)
    layout = _detect_layout(header.columns)
    combined = "fifa_version" in header.columns

    usecols = list(layout.values())
    if combined:
        usecols.append("fifa_version")
        if "fifa_update" in header.columns:
            usecols.append("fifa_update")

    if combined and game_version is None:
        raise SystemExit(
            f"{raw_path} is a combined dump (has fifa_version); "
            f"pass --game-version to pick one edition"
        )

    parts = []
    for chunk in pd.read_csv(raw_path, usecols=usecols, chunksize=200_000, low_memory=False):
        if combined:
            chunk = chunk[chunk["fifa_version"] == game_version]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        raise SystemExit(f"No rows with fifa_version == {game_version}")
    df = pd.concat(parts, ignore_index=True)

    # Keep only the latest roster update within the chosen edition.
    if combined and "fifa_update" in df.columns:
        df = df[df["fifa_update"] == df["fifa_update"].max()]

    out = pd.DataFrame(
        {
            "team": df[layout["team"]].replace(NAME_MAP),
            "player": df[layout["player"]],
            "overall": pd.to_numeric(df[layout["overall"]], errors="coerce"),
        }
    ).dropna(subset=["overall"])
    out = out.drop_duplicates(subset=["team", "player"])

    RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RATINGS_DIR / f"fifa_{release_year}.csv"
    out.to_csv(out_path, index=False)
    print(f"{raw_path} -> {out_path}: {len(out)} players, {out['team'].nunique()} teams")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_csv")
    parser.add_argument("release_year", type=int)
    parser.add_argument(
        "--game-version",
        type=int,
        help="edition to extract from a combined fifa_version dump (e.g. 23)",
    )
    args = parser.parse_args()
    convert(args.raw_csv, args.release_year, args.game_version)
