"""Tests for the email idempotency ledger (mlb/daily/send_ledger.py)."""

import json

from mlb.daily import send_ledger


def _setup(tmp_path, monkeypatch):
    reports = tmp_path / "reports" / "mlb_daily"
    reports.mkdir(parents=True)
    monkeypatch.setattr(send_ledger, "REPORTS_DIR", reports)
    monkeypatch.setattr(send_ledger, "LEDGER", reports / "sent.json")
    return reports


def _emails(reports, tmp_path, content="<p>slate v1</p>"):
    day = reports / "2026-08-14"
    day.mkdir(exist_ok=True)
    (day / "slate.html").write_text(content, encoding="utf-8")
    return {
        "slate": {
            "path": str((day / "slate.html").relative_to(tmp_path)),
            "subject": "MLB Slate — 2026-08-14",
        }
    }


def test_first_run_sends_and_rerun_does_not(tmp_path, monkeypatch):
    reports = _setup(tmp_path, monkeypatch)
    emails = _emails(reports, tmp_path)

    planned = send_ledger.plan(emails, "2026-08-14", "2026-08-13")
    assert planned["slate"]["send"] is True

    # Deliver, then re-plan the exact same content: no resend.
    (reports / "manifest_latest.json").write_text(
        json.dumps({"emails": planned}), encoding="utf-8")
    send_ledger.record(["slate"], "2026-08-14", "2026-08-13")
    replanned = send_ledger.plan(
        _emails(reports, tmp_path), "2026-08-14", "2026-08-13")
    assert replanned["slate"]["send"] is False


def test_changed_content_resends_as_updated(tmp_path, monkeypatch):
    reports = _setup(tmp_path, monkeypatch)
    planned = send_ledger.plan(
        _emails(reports, tmp_path), "2026-08-14", "2026-08-13")
    (reports / "manifest_latest.json").write_text(
        json.dumps({"emails": planned}), encoding="utf-8")
    send_ledger.record(["slate"], "2026-08-14", "2026-08-13")

    changed = send_ledger.plan(
        _emails(reports, tmp_path, content="<p>slate v2 — new probable</p>"),
        "2026-08-14", "2026-08-13")
    assert changed["slate"]["send"] is True
    assert changed["slate"]["subject"].endswith("(updated)")


def test_undelivered_send_is_retried(tmp_path, monkeypatch):
    # plan() marks send=true but record() is never called (SMTP failed):
    # the next run must still want to send.
    reports = _setup(tmp_path, monkeypatch)
    send_ledger.plan(_emails(reports, tmp_path), "2026-08-14", "2026-08-13")
    again = send_ledger.plan(
        _emails(reports, tmp_path), "2026-08-14", "2026-08-13")
    assert again["slate"]["send"] is True
    assert "(updated)" not in again["slate"]["subject"]


def test_grade_keys_by_graded_date(tmp_path, monkeypatch):
    reports = _setup(tmp_path, monkeypatch)
    day = reports / "2026-08-14"
    day.mkdir()
    (day / "grade.html").write_text("<p>grade</p>", encoding="utf-8")
    emails = {"grade": {
        "path": str((day / "grade.html").relative_to(tmp_path)),
        "subject": "MLB Grade — 2026-08-13",
    }}
    planned = send_ledger.plan(emails, "2026-08-14", "2026-08-13")
    assert planned["grade"]["date"] == "2026-08-13"
