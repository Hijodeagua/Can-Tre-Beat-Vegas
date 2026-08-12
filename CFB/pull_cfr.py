"""One-off polite CFR pulls for Tier 1 (DATA_PULL_PLAN.md §3.1-3.3).

Fetches the school index plus per-season standings and ratings pages from
Sports Reference and saves each table as a CSV under
``data/college_football/raw/`` — the same artifact a manual *Share & Export
→ Get table as CSV* click would produce, which is why the CSVs are treated
as raw and committed while the fetched HTML is only cached locally
(``raw_html/``, gitignored) so a parser fix never re-hits the site.

Politeness: SR blocks clients above ~20 requests/min; this script sleeps
``SLEEP_S`` between requests (well under), sends a browser User-Agent, is
resumable (skips CSVs that already exist), and per the plan must NEVER run
in CI. Requires ``lxml`` (local dev dependency for ``pandas.read_html``).

Usage
    python3 -m CFB.pull_cfr --schools          # 1 page smoke test
    python3 -m CFB.pull_cfr                    # everything missing
    python3 -m CFB.pull_cfr --years 2024 2025  # specific seasons
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "college_football" / "raw"
HTML_CACHE = REPO_ROOT / "data" / "college_football" / "raw_html"

BASE = "https://www.sports-reference.com/cfb"
YEARS = range(2000, 2026)
SLEEP_S = 5.0

# Plain requests gets a Cloudflare 403 (the plan called it) — a real headless
# Chrome passes, with a persistent profile so a cleared challenge sticks.
CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path("/usr/bin/google-chrome"),
]
CHROME = next((p for p in CHROME_CANDIDATES if p.exists()), None)
PROFILE_DIR = Path(tempfile.gettempdir()) / "cfr_pull_profile"

_last_request = 0.0


def fetch(url: str, cache_name: str) -> str:
    """Fetch via headless Chrome, on-disk cache, hard inter-request sleep."""
    global _last_request
    HTML_CACHE.mkdir(parents=True, exist_ok=True)
    cached = HTML_CACHE / cache_name
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    if CHROME is None:
        raise RuntimeError("no Chrome/Edge found for headless fetch")

    wait = SLEEP_S - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    proc = subprocess.run(
        [
            str(CHROME),
            "--headless=new",
            "--disable-gpu",
            f"--user-data-dir={PROFILE_DIR}",
            "--virtual-time-budget=8000",  # let any JS challenge settle
            "--dump-dom",
            url,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
    )
    _last_request = time.monotonic()
    html = proc.stdout
    if "<table" not in html or "Just a moment" in html:
        raise RuntimeError(
            f"blocked or empty response for {url} ({len(html)} bytes)"
        )
    cached.write_text(html, encoding="utf-8")
    return html


def read_tables(html: str) -> list[pd.DataFrame]:
    """SR hides some tables inside HTML comments — uncomment, then parse."""
    html = html.replace("<!--", "").replace("-->", "")
    return pd.read_html(html)


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse SR's two-row headers ('Overall'/'W') into 'Overall_W'."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(p for p in map(str, tup) if not p.startswith("Unnamed"))
            for tup in df.columns
        ]
    # Drop mid-table repeated header rows.
    first = df.columns[0]
    df = df[df[first].astype(str) != str(first)]
    return df.reset_index(drop=True)


def _pick_table(tables: list[pd.DataFrame], must_have: set[str]) -> pd.DataFrame:
    for t in tables:
        flat = _flatten(t.copy())
        cols = {re.sub(r".*_", "", c) for c in map(str, flat.columns)}
        if must_have <= cols:
            return flat
    raise ValueError(f"no table with columns {must_have}")


def pull_schools() -> None:
    out = RAW_DIR / "cfb_schools.csv"
    if out.exists():
        print(f"skip {out.name} (exists)")
        return
    html = fetch(f"{BASE}/schools/", "schools.html")
    df = _pick_table(read_tables(html), {"School", "From", "To"})
    df.to_csv(out, index=False)
    print(f"wrote {out.name}: {len(df)} schools")


def pull_year(year: int, kind: str) -> None:
    out = RAW_DIR / kind / f"cfb_{kind}_{year}.csv"
    if out.exists():
        print(f"skip {out.name} (exists)")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    html = fetch(f"{BASE}/years/{year}-{kind}.html", f"{year}-{kind}.html")
    df = _pick_table(read_tables(html), {"School", "SRS"})
    df.to_csv(out, index=False)
    print(f"wrote {kind}/{out.name}: {len(df)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schools", action="store_true", help="school index only")
    ap.add_argument("--years", type=int, nargs="*", default=list(YEARS))
    args = ap.parse_args()

    pull_schools()
    if args.schools:
        return

    failures = []
    for year in args.years:
        for kind in ("standings", "ratings"):
            try:
                pull_year(year, kind)
            except Exception as e:  # keep pulling; report at the end
                failures.append((year, kind, str(e)))
                print(f"FAILED {year}-{kind}: {e}")

    if failures:
        print(f"\n{len(failures)} pulls failed:")
        for year, kind, err in failures:
            print(f"  {year}-{kind}: {err}")
        sys.exit(1)
    print("\nall pulls complete")


if __name__ == "__main__":
    main()
