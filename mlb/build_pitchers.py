"""Ingest Baseball-Reference "individual pitcher saber" CSV chunks into the
canonical pitcher-season table, data/mlb/pitcher_seasons.csv.

    python -m mlb.build_pitchers <chunk.csv | directory> [...]

The source export arrives in ranked chunks (top pitcher seasons 2009-present,
single season, IP >= 125, sorted by Adjusted Pitching Runs). More chunks may
arrive over time, so ingestion is append + dedupe:

- Dedupe key is (bbref_id, season) - the `Player-additional` column - NOT the
  display name. Names are unstable across sources (accents: "Liván Hernández";
  legal name changes: Roberto Hernández carries Fausto Carmona's carmofa01).
- Re-ingesting an already-loaded chunk is a no-op; a row with the same key but
  different values overwrites (assume the newer export is a correction).

Coverage caveat (by construction, do not paper over it): the IP >= 125 filter
means relievers, swingmen, injured/short seasons and partial rookie years are
absent. Consumers must treat a missing (pitcher, season) as "no qualifying
season", never as zero/average quality, and should surface unmatched pitchers
rather than silently dropping them (see mlb/daily and the backtest).
"""

from __future__ import annotations

import csv
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "mlb" / "pitcher_seasons.csv"

# source header -> canonical column. The export repeats IP and PtchR; we keep
# the first occurrence of each (values are identical).
KEEP = {
    "Rk": "rk",
    "Player": "player",
    "Player-additional": "bbref_id",
    "Season": "season",
    "Age": "age",
    "Team": "team",
    "Lg": "lg",
    "IP": "ip",
    "G": "g",
    "GS": "gs",
    "W": "w",
    "L": "l",
    "ERA": "era",
    "ERA+": "era_plus",
    "FIP": "fip",
    "WHIP": "whip",
    "BB9": "bb9",
    "SO9": "so9",
    "SO/BB": "so_bb",
    "PtchR": "ptchr",
}

FIELDNAMES = [
    "bbref_id", "season", "player", "name_norm", "age", "team", "lg",
    "ip", "g", "gs", "w", "l", "era", "era_plus", "fip", "whip",
    "bb9", "so9", "so_bb", "ptchr", "rk",
]


def normalize_name(name: str) -> str:
    """Accent-stripped, lowercased, punctuation-free form for cross-source
    joins (bbref <-> Retrosheet <-> MLB Stats API)."""
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in stripped)
    return " ".join(cleaned.lower().split())


def parse_chunk(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        header = next(csv.reader(fh))
        # First-occurrence index for each header we keep (IP/PtchR repeat).
        idx: dict[str, int] = {}
        for i, col in enumerate(header):
            if col in KEEP and col not in idx:
                idx[col] = i
        missing = set(KEEP) - set(idx)
        if missing:
            raise SystemExit(f"{path}: missing expected columns {sorted(missing)}")
        for rec in csv.reader(fh):
            if not rec or not rec[idx["Player-additional"]].strip():
                raise SystemExit(
                    f"{path}: row without a bbref id (Player-additional): {rec[:3]}"
                )
            row = {out: rec[idx[src]].strip() for src, out in KEEP.items()}
            row["name_norm"] = normalize_name(row["player"])
            rows.append(row)
    return rows


def load_existing() -> dict[tuple[str, str], dict]:
    if not OUT.exists():
        return {}
    with open(OUT, newline="", encoding="utf-8") as fh:
        return {(r["bbref_id"], r["season"]): r for r in csv.DictReader(fh)}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    paths: list[Path] = []
    for a in argv:
        p = Path(a)
        if p.is_dir():
            paths += sorted(p.glob("*individual_pitcher_saber*.csv"))
        else:
            paths.append(p)
    if not paths:
        raise SystemExit("no input CSVs found")

    table = load_existing()
    before = len(table)
    read = added = updated = unchanged = 0
    for p in paths:
        for row in parse_chunk(p):
            read += 1
            key = (row["bbref_id"], row["season"])
            old = table.get(key)
            if old is None:
                added += 1
            elif {k: old.get(k) for k in FIELDNAMES} != row:
                updated += 1
            else:
                unchanged += 1
            table[key] = row

    rows = sorted(table.values(), key=lambda r: (int(r["season"]), int(r["rk"])))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    seasons = sorted({int(r["season"]) for r in rows})
    print(f"read {read} rows from {len(paths)} files: "
          f"{added} added, {updated} updated, {unchanged} unchanged "
          f"(table {before} -> {len(table)})")
    print(f"seasons {seasons[0]}-{seasons[-1]}; wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
