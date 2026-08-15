"""Build the FULL person ID crosswalk (Retrosheet <-> MLBAM <-> bbref) from a
local clone of the Chadwick Bureau register.

    git clone --depth 1 --filter=blob:none --no-checkout \
        https://github.com/chadwickbureau/register
    (cd register && git sparse-checkout set --no-cone 'data/people-*.csv' \
        && git checkout HEAD)
    python -m mlb.build_person_crosswalk register/data

Unlike mlb/build_crosswalk.py (which is restricted to the IP>=125
qualified-season pitcher table and exists for the season-quality backtest),
this writes every register row that carries a Retrosheet ID - the population
the per-start ingest (mlb/build_starts.py) needs, since *every* starter must
resolve to an MLBAM id for the live pipeline to match statsapi probables.

Writes data/mlb/person_id_crosswalk.csv:
    retro_id, mlbam_id, bbref_id, name_first, name_last
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "mlb" / "person_id_crosswalk.csv"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    register_data = Path(argv[0])

    rows = []
    for path in sorted(register_data.glob("people-*.csv")):
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("key_retro"):
                    rows.append({
                        "retro_id": r["key_retro"],
                        "mlbam_id": r.get("key_mlbam", ""),
                        "bbref_id": r.get("key_bbref", ""),
                        "name_first": r.get("name_first", ""),
                        "name_last": r.get("name_last", ""),
                    })

    dupes = len(rows) - len({r["retro_id"] for r in rows})
    if dupes:
        raise SystemExit(f"{dupes} duplicate retro_ids in register - refusing")

    rows.sort(key=lambda r: r["retro_id"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    no_mlbam = sum(1 for r in rows if not r["mlbam_id"])
    print(f"crosswalk: {len(rows)} people with retro ids -> {OUT}")
    print(f"  missing mlbam id: {no_mlbam} "
          "(mostly pre-MLBAM-era players; modern starters must all resolve)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
