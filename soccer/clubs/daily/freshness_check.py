"""Fail the soccer workflow loudly when a league's results have stalled.

Reads the `results_freshness` block the pipeline just exported to
`web/public/data/soccer/latest.json` (so this check and the site agree by
construction), prints one line per league, and exits nonzero when any
league is stale. It runs *after* the commit and the email steps so the
day's outputs - carrying their own stale warning - still land; the run is
then marked failed, which is what makes GitHub notify the repo owner.
The thresholds live in `config.py`.

    python -m soccer.clubs.daily.freshness_check [--latest PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from soccer.clubs.daily.config import SITE_LATEST


def check(payload: dict, log=print) -> list[str]:
    """Log every league's status; return the stale league keys."""
    block = payload.get("results_freshness") or {}
    leagues = block.get("leagues") or {}
    if not leagues:
        log("::error::latest.json carries no results_freshness block - the pipeline "
            "did not export it, so staleness cannot be judged")
        return ["<missing>"]
    stale = []
    for key, e in leagues.items():
        mark = "STALE" if e.get("stale") else "ok"
        log(f"   {e.get('name', key):<20} newest result {e.get('newest_played') or '-':<10} "
            f"overdue {e.get('overdue', 0):>3}  {mark}"
            + (f"  ({e['reason']})" if e.get("reason") and e.get("stale") else ""))
        if e.get("stale"):
            stale.append(key)
    for key in stale:
        e = leagues[key]
        log(f"::error::{e.get('name', key)} results are stale: {e.get('reason')}")
    return stale


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latest", default=str(SITE_LATEST))
    args = ap.parse_args(argv)
    path = Path(args.latest)
    if not path.exists():
        print(f"::error::{path} not found")
        return 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"Results freshness as of {payload.get('results_freshness', {}).get('as_of')} "
          f"(run {payload.get('run_date')}):")
    stale = check(payload)
    if stale:
        print(f"{len(stale)} league(s) stale: {', '.join(stale)}")
        return 1
    print("All leagues current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
