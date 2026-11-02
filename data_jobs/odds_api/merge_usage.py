"""Merge two copies of ``data/api_usage.json`` that diverged in a push race.

The unified odds job appends one history entry per sport per run and
commits the file. When two commits race (the daily sport jobs all push
around the same time), ``.github/scripts/git-push-retry.sh`` rebases the
odds commit onto whatever landed first; if that commit also touched this
file the rebase stops on a conflict. Dropping either side would lose a
usage record (and with it the quota reading the next run reports), so the
resolver calls this instead: the union of both histories keyed on
(timestamp, sport), newest-first order preserved, the same 100-entry cap
the client applies, and ``last_remaining`` taken from the newest entry.

    python -m data_jobs.odds_api.merge_usage --ours A.json --theirs B.json --out data/api_usage.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HISTORY_CAP = 100   # mirrors OddsAPIClient._log_usage


def _load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"history": [], "last_remaining": None}
    if not isinstance(data, dict):
        return {"history": [], "last_remaining": None}
    data.setdefault("history", [])
    data.setdefault("last_remaining", None)
    return data


def merge_usage(ours: dict, theirs: dict) -> dict:
    """Union of both histories, deduplicated on (timestamp, sport) and
    sorted by timestamp, capped like the client caps it."""
    seen: dict[tuple, dict] = {}
    for entry in list(ours.get("history", [])) + list(theirs.get("history", [])):
        if not isinstance(entry, dict):
            continue
        key = (str(entry.get("timestamp", "")), str(entry.get("sport", "")))
        seen.setdefault(key, entry)
    history = sorted(seen.values(), key=lambda e: str(e.get("timestamp", "")))
    history = history[-HISTORY_CAP:]
    merged = {k: v for side in (theirs, ours) for k, v in side.items()
              if k not in ("history", "last_remaining")}
    merged["history"] = history
    last = history[-1] if history else None
    merged["last_remaining"] = (last.get("requests_remaining") if last
                                else ours.get("last_remaining", theirs.get("last_remaining")))
    return merged


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--theirs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    merged = merge_usage(_load(Path(args.ours)), _load(Path(args.theirs)))
    Path(args.out).write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"merge_usage: {len(merged['history'])} history entries -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
