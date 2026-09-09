"""Tests for the email archive: publishing stamps a permalink footer once,
copies the email to the public archive, keeps one index record per
(league, type, date), links each email to the previous one of its kind,
and the backfill picks up emails that predate the archive."""

import json

import pytest

from data_jobs import email_archive
from data_jobs.email_archive import FOOTER_MARK, backfill, load_index, publish


def _write(root, rel, content="<p>hello</p>"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return rel


@pytest.fixture
def repo(tmp_path):
    return tmp_path


def _publish(repo, league, email_type, date, subject="Subj", content="<p>hello</p>"):
    rel = _write(repo, f"reports/{league}/{date}/{email_type}.html", content)
    entries = {email_type: {"path": rel, "subject": subject, "date": date}}
    return publish(league, entries, repo_root=repo, archive_dir=repo / "web" / "public" / "emails")


def test_publish_stamps_footer_copies_and_indexes(repo):
    entries = _publish(repo, "nfl", "update", "2026-09-07")
    sent = (repo / "reports/nfl/2026-09-07/update.html").read_text()
    archived = (repo / "web/public/emails/nfl/2026-09-07_update.html").read_text()
    assert sent == archived                       # delivered == archived bytes
    assert FOOTER_MARK in sent and "All NFL emails" in sent
    assert entries["update"]["url"].endswith("/vegas/emails/nfl/2026-09-07_update.html")
    idx = load_index(repo / "web/public/emails/index.json")
    assert len(idx["emails"]) == 1
    rec = idx["emails"][0]
    assert rec["league"] == "nfl" and rec["type"] == "update" and rec["date"] == "2026-09-07"
    assert rec["public_path"] == "emails/nfl/2026-09-07_update.html"
    assert idx["leagues"]["nfl"] == "NFL"


def test_republish_is_idempotent_and_updates_in_place(repo):
    _publish(repo, "cfb", "update", "2026-09-05", subject="v1")
    # A rerun of the same day (new content) replaces the record, never duplicates.
    _publish(repo, "cfb", "update", "2026-09-05", subject="v2", content="<p>v2</p>")
    idx = load_index(repo / "web/public/emails/index.json")
    assert [e["subject"] for e in idx["emails"]] == ["v2"]
    sent = (repo / "reports/cfb/2026-09-05/update.html").read_text()
    assert sent.count(FOOTER_MARK) == 1


def test_previous_email_link_and_league_sort(repo):
    _publish(repo, "mlb", "slate", "2026-09-01")
    _publish(repo, "mlb", "grade", "2026-09-01")
    _publish(repo, "mlb", "slate", "2026-09-02")
    later = (repo / "reports/mlb/2026-09-02/slate.html").read_text()
    assert "Previous: <a" in later and "2026-09-01_slate.html" in later
    first = (repo / "reports/mlb/2026-09-01/slate.html").read_text()
    assert "Previous:" not in first
    # The grade email is a different type: it is not the slate's "previous".
    assert "2026-09-01_grade.html" not in later
    _publish(repo, "nfl", "update", "2026-09-07")
    idx = load_index(repo / "web/public/emails/index.json")
    keys = [(e["league"], e["date"], e["type"]) for e in idx["emails"]]
    assert keys == sorted(keys, reverse=True)


def test_unknown_league_rejected(repo):
    with pytest.raises(ValueError):
        _publish(repo, "curling", "update", "2026-09-07")


def test_backfill_walks_reports_and_uses_ledger_dates(repo):
    _write(repo, "reports/mlb_daily/2026-09-04/futures.html", "<p>f</p>")
    _write(repo, "reports/mlb_daily/2026-09-04/grade.html", "<p>g</p>")
    _write(repo, "reports/mlb_daily/2026-09-04/notes.txt", "x")
    (repo / "reports/mlb_daily/sent.json").write_text(json.dumps({
        "futures:2026-09-04": {"hash": "h", "subject": "MLB Futures — 2026-09-04"},
        "grade:2026-09-03": {"hash": "h", "subject": "MLB Grade — 2026-09-03"},
    }))
    _write(repo, "reports/models_check/2026-08-31/check.html", "<p>c</p>")
    _write(repo, "reports/nfl/manifest_latest.json", "{}")
    n = backfill(repo_root=repo, archive_dir=repo / "web/public/emails")
    assert n == 3
    idx = load_index(repo / "web/public/emails/index.json")
    by = {(e["league"], e["type"]): e for e in idx["emails"]}
    assert by[("mlb", "grade")]["date"] == "2026-09-03"       # graded date, from the ledger
    assert by[("mlb", "grade")]["subject"] == "MLB Grade — 2026-09-03"
    assert by[("mlb", "futures")]["date"] == "2026-09-04"
    assert by[("models", "models")]["date"] == "2026-08-31"
    assert (repo / "web/public/emails/mlb/2026-09-03_grade.html").read_text() == "<p>g</p>"
    # Backfilled copies are as sent: no footer added retroactively; a second
    # backfill adds nothing.
    assert FOOTER_MARK not in (repo / "web/public/emails/models/2026-08-31_models.html").read_text()
    assert backfill(repo_root=repo, archive_dir=repo / "web/public/emails") == 0
