"""Generic email idempotency ledger — the sport-agnostic sibling of
mlb/daily/send_ledger.py, for pipelines that keep their manifest under
reports/<name>/.

A manifest is {"emails": {type: {path, subject, date, ...}}} with `path`
relative to the repo root and `date` the key date for that email. The
ledger (sent.json, next to the manifest) maps "{type}:{date}" to the
sha256 of the delivered HTML:

- plan(): annotate each entry with send/hash. Already delivered with the
  same hash -> send=false; delivered but content changed -> send=true with
  "(updated)" appended; never delivered -> send=true. Callers may then
  apply their own further gates (e.g. only-send-on-Mondays) on top.
- record (CLI): called by the workflow AFTER the SMTP step succeeded, then
  committed — a failed send is retried next run, never falsely remembered.

    python -m data_jobs.email_ledger record \
        --manifest reports/soccer/manifest_latest.json --types "update"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(ledger_path: Path) -> dict:
    if ledger_path.exists():
        return json.loads(ledger_path.read_text(encoding="utf-8"))
    return {}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(report_type: str, date: str) -> str:
    return f"{report_type}:{date}"


def plan(emails: dict, ledger_path: Path, repo_root: Path = REPO_ROOT) -> dict:
    """Annotate each manifest email entry (which must carry `path` and
    `date`) with send/hash, consulting the ledger. Mutates and returns
    `emails`."""
    ledger = _load(ledger_path)
    for report_type, entry in emails.items():
        h = _hash(repo_root / entry["path"])
        prior = ledger.get(_key(report_type, entry["date"]))
        if prior and prior.get("hash") == h:
            entry["send"] = False
        else:
            entry["send"] = True
            if prior:  # same key, new content: an update, and say so
                entry["subject"] += " (updated)"
        entry["hash"] = h
    return emails


def record(manifest_path: Path, types: list[str]) -> int:
    """Mark reports as delivered using the hashes in the manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger_path = manifest_path.parent / "sent.json"
    ledger = _load(ledger_path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for report_type in types:
        entry = manifest["emails"].get(report_type)
        if not entry:
            print(f"email_ledger: no manifest entry for {report_type} - skip")
            continue
        ledger[_key(report_type, entry["date"])] = {
            "hash": entry["hash"], "sent_at": now, "subject": entry["subject"],
        }
        print(f"email_ledger: recorded {report_type}:{entry['date']}")
    ledger_path.write_text(json.dumps(ledger, indent=1, sort_keys=True),
                           encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--manifest", required=True,
                     help="manifest path, relative to the repo root")
    rec.add_argument("--types", required=True,
                     help="space-separated report types actually delivered")
    args = ap.parse_args(argv)
    return record(REPO_ROOT / args.manifest, args.types.split())


if __name__ == "__main__":
    sys.exit(main())
