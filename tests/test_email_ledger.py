"""Tests for the generic email idempotency ledger (data_jobs/email_ledger.py)."""

import json

from data_jobs import email_ledger


def _emails(tmp_path, content="<p>update v1</p>"):
    day = tmp_path / "reports" / "soccer" / "2026-08-31"
    day.mkdir(parents=True, exist_ok=True)
    (day / "update.html").write_text(content, encoding="utf-8")
    return {
        "update": {
            "path": str((day / "update.html").relative_to(tmp_path)),
            "subject": "Soccer Update — 2026-08-31",
            "date": "2026-08-31",
        }
    }


def _ledger_path(tmp_path):
    return tmp_path / "reports" / "soccer" / "sent.json"


def _record(tmp_path, planned):
    manifest = tmp_path / "reports" / "soccer" / "manifest_latest.json"
    manifest.write_text(json.dumps({"emails": planned}), encoding="utf-8")
    email_ledger.record(manifest, ["update"])


def test_first_run_sends_and_rerun_does_not(tmp_path):
    planned = email_ledger.plan(_emails(tmp_path), _ledger_path(tmp_path), tmp_path)
    assert planned["update"]["send"] is True

    _record(tmp_path, planned)
    replanned = email_ledger.plan(_emails(tmp_path), _ledger_path(tmp_path), tmp_path)
    assert replanned["update"]["send"] is False


def test_changed_content_resends_as_updated(tmp_path):
    planned = email_ledger.plan(_emails(tmp_path), _ledger_path(tmp_path), tmp_path)
    _record(tmp_path, planned)

    changed = email_ledger.plan(
        _emails(tmp_path, content="<p>update v2</p>"),
        _ledger_path(tmp_path), tmp_path)
    assert changed["update"]["send"] is True
    assert changed["update"]["subject"].endswith("(updated)")


def test_unsent_email_is_retried(tmp_path):
    # plan() marks send=true but the workflow never records a delivery
    # (failed SMTP) — the next plan must still say send=true.
    email_ledger.plan(_emails(tmp_path), _ledger_path(tmp_path), tmp_path)
    replanned = email_ledger.plan(_emails(tmp_path), _ledger_path(tmp_path), tmp_path)
    assert replanned["update"]["send"] is True


def test_different_dates_key_separately(tmp_path):
    planned = email_ledger.plan(_emails(tmp_path), _ledger_path(tmp_path), tmp_path)
    _record(tmp_path, planned)

    nxt = _emails(tmp_path)
    nxt["update"]["date"] = "2026-09-03"
    replanned = email_ledger.plan(nxt, _ledger_path(tmp_path), tmp_path)
    assert replanned["update"]["send"] is True
