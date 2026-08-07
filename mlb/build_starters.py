"""Extract per-game starting pitchers 2009-2025 from Retrosheet game logs.

    python -m mlb.build_starters

Downloads gl{year}.zip into data/mlb/raw_gamelogs/ (skipped when cached -
same cache directory build_games.py uses) and writes
data/mlb/starting_pitchers_2009_2025.csv with one row per game:

    date, season, game_num, away, home, away_fr, home_fr,
    away_sp_retro, away_sp_name, home_sp_retro, home_sp_name

Retrosheet's game log record puts the starting pitchers at 1-indexed fields
102-105 (visitor ID, visitor name, home ID, home name), after the umpire /
manager / winning-losing-save pitcher blocks. Because several nearby blocks
are also (id, name) pairs, the extraction validates every row against the
Retrosheet ID shape and fails loudly if the offsets ever look wrong, rather
than silently shipping umpires as starters.

NOTE: retrosheet.org is unreachable from the Claude sandbox (egress policy);
run this via the fetch-starters GitHub Actions workflow, which commits the
output CSV. Data courtesy of Retrosheet (https://www.retrosheet.org).
"""

from __future__ import annotations

import csv
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

from mlb.build_games import RS_TO_BREF, franchise

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "mlb" / "raw_gamelogs"
OUT = REPO / "data" / "mlb" / "starting_pitchers_2009_2025.csv"

YEARS = range(2009, 2026)
URL = "https://www.retrosheet.org/gamelogs/gl{year}.zip"

# 0-indexed positions in the game log record (spec fields 102-105).
V_SP_ID, V_SP_NAME, H_SP_ID, H_SP_NAME = 101, 102, 103, 104

# Retrosheet person IDs: 4 letters of surname (padded with -), 1 initial,
# 3 digits, e.g. kersc001, leec-001.
RETRO_ID = re.compile(r"^[a-z-]{5}\d{3}$")


def fetch_year(year: int) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    zp = RAW / f"gl{year}.zip"
    if not zp.exists():
        print(f"downloading {URL.format(year=year)}")
        with urllib.request.urlopen(URL.format(year=year), timeout=120) as r:
            zp.write_bytes(r.read())
    return zp


def extract_year(zp: Path, year: int) -> list[dict]:
    with zipfile.ZipFile(zp) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".txt")][0]
        text = z.read(name).decode("latin-1")
    rows, bad = [], 0
    for rec in csv.reader(io.StringIO(text)):
        date, gnum = rec[0], rec[1]
        away, home = RS_TO_BREF[rec[3]], RS_TO_BREF[rec[6]]
        v_id, v_name = rec[V_SP_ID].strip(), rec[V_SP_NAME].strip()
        h_id, h_name = rec[H_SP_ID].strip(), rec[H_SP_NAME].strip()
        if not (RETRO_ID.match(v_id) and RETRO_ID.match(h_id)):
            bad += 1
            continue
        rows.append({
            "date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
            "season": year,
            "game_num": int(gnum),
            "away": away, "home": home,
            "away_fr": franchise(away), "home_fr": franchise(home),
            "away_sp_retro": v_id, "away_sp_name": v_name,
            "home_sp_retro": h_id, "home_sp_name": h_name,
        })
    # Starters are recorded for essentially every regular-season game; more
    # than 1% failures means the field offsets are wrong - abort.
    if bad > 0.01 * max(len(rows) + bad, 1):
        raise SystemExit(
            f"{year}: {bad}/{len(rows) + bad} rows failed the Retrosheet ID "
            f"check at fields 102-105 - offsets look wrong, refusing to write"
        )
    if bad:
        print(f"{year}: skipped {bad} rows without valid starter IDs")
    return rows


def main() -> int:
    rows: list[dict] = []
    for year in YEARS:
        rows += extract_year(fetch_year(year), year)
    rows.sort(key=lambda r: (r["date"], r["game_num"], r["home"]))
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} games -> {OUT}")
    print("sample:", rows[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
