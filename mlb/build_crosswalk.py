"""Build the pitcher ID crosswalk (bbref <-> Retrosheet <-> MLBAM) from a
local clone of the Chadwick Bureau register.

    git clone --depth 1 https://github.com/chadwickbureau/register.git
    python -m mlb.build_crosswalk register/data

Writes data/mlb/pitcher_id_crosswalk.csv restricted to the bbref IDs present
in data/mlb/pitcher_seasons.csv (the IP>=125 qualified-season table). The
register is the authoritative public mapping, so no name-based joining is
needed for pitchers who have a qualifying season; starters outside that table
are unmatched *by construction* (no qualifying season) and are surfaced as
coverage gaps by the backtest, not silently joined.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEASONS = REPO / "data" / "mlb" / "pitcher_seasons.csv"
OUT = REPO / "data" / "mlb" / "pitcher_id_crosswalk.csv"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    register_data = Path(argv[0])

    with open(SEASONS, newline="", encoding="utf-8") as fh:
        wanted = {r["bbref_id"] for r in csv.DictReader(fh)}

    rows = []
    for path in sorted(register_data.glob("people-*.csv")):
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("key_bbref") in wanted:
                    rows.append({
                        "bbref_id": r["key_bbref"],
                        "retro_id": r.get("key_retro", ""),
                        "mlbam_id": r.get("key_mlbam", ""),
                        "name_first": r.get("name_first", ""),
                        "name_last": r.get("name_last", ""),
                    })

    rows.sort(key=lambda r: r["bbref_id"])
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    missing = wanted - {r["bbref_id"] for r in rows}
    no_retro = [r["bbref_id"] for r in rows if not r["retro_id"]]
    no_mlbam = [r["bbref_id"] for r in rows if not r["mlbam_id"]]
    print(f"crosswalk: {len(rows)}/{len(wanted)} pitchers -> {OUT}")
    if missing:
        print(f"NOT IN REGISTER ({len(missing)}): {sorted(missing)}")
    if no_retro:
        print(f"missing retro id ({len(no_retro)}): {sorted(no_retro)}")
    if no_mlbam:
        print(f"missing mlbam id ({len(no_mlbam)}): {sorted(no_mlbam)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
