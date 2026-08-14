"""Email idempotency ledger, keyed on (report_type, date).

Fixes the duplicate-send bug (2026-08-10: a push-triggered rerun re-emailed
the same three reports the schedule run had already sent - see
docs/SP-AUDIT.md §3). The pipeline's data layer was already idempotent; the
sends were not.

reports/mlb_daily/sent.json maps "{report_type}:{date}" to the sha256 of the
delivered HTML plus a timestamp. Two halves:

- plan(): called by the pipeline when writing the manifest. For each email,
  compare its content hash against the ledger: already delivered with the
  same hash -> send=false (a rerun changes nothing, so it re-sends
  nothing); delivered but the content changed (late score correction, new
  probables) -> send=true with "(updated)" appended to the subject; never
  delivered -> send=true.
- record (CLI): called by the workflow AFTER the SMTP steps, only for
  emails whose send step actually succeeded, then committed. Recording
  after delivery means a failed send is retried by the next run rather than
  being falsely remembered as sent.

    python -m mlb.daily.send_ledger record --date 2026-08-14 \
        --types "futures slate" [--graded-date 2026-08-13]

The grade email is keyed by the graded date (its content date), not the run
date, so a makeup grade re-run keys correctly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

from mlb.daily.config import REPORTS_DIR

LEDGER = REPORTS_DIR / "sent.json"


def _load() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {}


def _hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(report_type: str, date: str) -> str:
    return f"{report_type}:{date}"


def email_date(report_type: str, run_date: str, graded_date: str) -> str:
    return graded_date if report_type == "grade" else run_date


def plan(emails: dict, run_date: str, graded_date: str) -> dict:
    """Annotate each manifest email entry with send/hash, consulting the
    ledger. Mutates and returns `emails`."""
    ledger = _load()
    for report_type, entry in emails.items():
        path = REPORTS_DIR.parent.parent / entry["path"]
        h = _hash(path)
        d = email_date(report_type, run_date, graded_date)
        prior = ledger.get(_key(report_type, d))
        if prior and prior.get("hash") == h:
            entry["send"] = False
        else:
            entry["send"] = True
            if prior:  # same key, new content: an update, and say so
                entry["subject"] += " (updated)"
        entry["hash"] = h
        entry["date"] = d
    return emails


def record(types: list[str], run_date: str, graded_date: str) -> int:
    """Mark reports as delivered using the hashes in the current manifest."""
    manifest = json.loads(
        (REPORTS_DIR / "manifest_latest.json").read_text(encoding="utf-8"))
    ledger = _load()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for report_type in types:
        entry = manifest["emails"].get(report_type)
        if not entry:
            print(f"send_ledger: no manifest entry for {report_type} - skip")
            continue
        d = entry.get("date") or email_date(report_type, run_date, graded_date)
        ledger[_key(report_type, d)] = {
            "hash": entry["hash"], "sent_at": now, "subject": entry["subject"],
        }
        print(f"send_ledger: recorded {report_type}:{d}")
    LEDGER.write_text(json.dumps(ledger, indent=1, sort_keys=True),
                      encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--date", required=True)
    rec.add_argument("--graded-date", default="")
    rec.add_argument("--types", required=True,
                     help="space-separated report types actually delivered")
    args = ap.parse_args(argv)
    return record(args.types.split(), args.date, args.graded_date)


if __name__ == "__main__":
    sys.exit(main())
