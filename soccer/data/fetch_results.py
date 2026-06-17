"""
Refresh the committed international results from the upstream martj42 dataset.

SPEC.md lists martj42/international_results (CC0) as the source of
`soccer/data/results.csv` (+ `shootouts.csv`). This pulls the latest revision
so newly-played matches — including 2026 World Cup results as they happen —
flow into the Elo engine on the next `export_ratings` run.

Network-gated and best-effort: if the fetch fails (offline CI, upstream down),
it leaves the committed CSVs untouched and exits non-zero so a caller can
decide whether to proceed with the existing data.

Usage:
    python -m soccer.data.fetch_results
"""

import sys
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent
RAW_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = {
    "results.csv": f"{RAW_BASE}/results.csv",
    "shootouts.csv": f"{RAW_BASE}/shootouts.csv",
}


def fetch(timeout: int = 30) -> int:
    updated = 0
    for name, url in FILES.items():
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  ! {name}: fetch failed ({exc}); keeping committed copy")
            continue
        dest = DATA_DIR / name
        before = dest.read_bytes() if dest.exists() else b""
        if resp.content == before:
            print(f"  = {name}: unchanged")
            continue
        dest.write_bytes(resp.content)
        rows = max(0, resp.content.count(b"\n") - 1)
        print(f"  + {name}: updated ({rows} rows)")
        updated += 1
    return updated


def main() -> None:
    print("Fetching latest international results from martj42…")
    updated = fetch()
    if updated == 0:
        print("No files updated.")
        # Distinguish "nothing changed" (fine) from "all fetches errored".
    print(f"Done ({updated} file(s) changed).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
