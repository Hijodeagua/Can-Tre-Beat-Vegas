"""Weekly model-review drafts: every model builds from the committed
files, the page carries the five sections with no script, the numbers
match their sources, the schedule maps weekdays to models, and the
review session's freshness check refuses an old or missing draft."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from data_jobs.reports import model_review as mr
from data_jobs.reports.model_review_html import SECTION_ORDER, render

RUN_DATE = "2026-11-02"
REPO = mr.REPO


@pytest.fixture(scope="module")
def drafts(tmp_path_factory):
    out = tmp_path_factory.mktemp("reviews")
    docs = {}
    for m in mr.MODELS:
        doc = mr.build(m, RUN_DATE)
        mr.write(doc, out)
        docs[m] = doc
    return out, docs


class TestDrafts:
    def test_every_model_writes_dated_and_latest_files(self, drafts):
        out, docs = drafts
        for m in mr.MODELS:
            for name in (f"{RUN_DATE}.json", f"{RUN_DATE}.html", "latest.json", "latest.html"):
                assert (out / m / name).exists(), f"{m}/{name}"
            latest = json.loads((out / m / "latest.json").read_text())
            assert latest["date"] == RUN_DATE and latest["model"] == m and latest["kind"] == "draft"

    def test_five_sections_each_with_a_source(self, drafts):
        _, docs = drafts
        for m, doc in docs.items():
            for key, _label in SECTION_ORDER:
                sec = doc[key]
                assert sec.get("blocks") or sec.get("summary"), f"{m}.{key} empty"
                has_source = bool(sec.get("source")) or any(b.get("source") for b in sec.get("blocks", []))
                assert has_source or key == "improvements", f"{m}.{key} has no data source"

    def test_html_is_static_phone_friendly_and_complete(self, drafts):
        out, _ = drafts
        for m in mr.MODELS:
            html = (out / m / "latest.html").read_text()
            assert "<script" not in html.lower()
            assert 'name=\'viewport\'' in html and "width=device-width" in html
            assert not re.search(r"<link[^>]+href=|src=['\"]https?://", html), "external resource"
            for _key, label in SECTION_ORDER:
                assert f">{label}</h2>" in html, f"{m}: section {label} missing"

    def test_nfl_numbers_match_their_sources(self, drafts):
        _, docs = drafts
        doc = docs["nfl"]
        g = pd.read_csv(REPO / "data" / "nfl" / "predictions" / "grades.csv")
        g = g[g["date"] <= RUN_DATE]
        ledger = json.loads((REPO / "web" / "public" / "data" / "nfl" / "latest.json").read_text())["ledger"]
        windows = next(b for b in doc["report_card"]["blocks"] if b["heading"].startswith("Windows"))["rows"]
        season = next(r for r in windows if r[0] == "Season to date")
        assert season[1] == len(g)
        assert season[2] == f"{100 * g['pick_correct'].astype(bool).mean():.1f}%"
        assert season[3] == round(float(g["log_loss"].mean()), 4) == round(ledger["log_loss"], 4)
        kv = dict(next(b for b in doc["status"]["blocks"] if b.get("heading") == "Pipeline")["rows"])
        assert kv["Last graded game"] == g["date"].max()

    def test_soccer_reports_stale_leagues_from_the_site_block(self, drafts):
        _, docs = drafts
        doc = docs["soccer"]
        site = json.loads((REPO / "web" / "public" / "data" / "soccer" / "latest.json").read_text())
        fresh = site.get("results_freshness") or {}
        if fresh.get("stale_leagues"):
            assert doc["status"]["state"] == "stale"
            caveats = next(b for b in doc["status"]["blocks"] if b.get("heading", "").startswith("Known"))["items"]
            assert any("stale" in c for c in caveats)
        rows = next(b for b in doc["status"]["blocks"] if b["heading"].startswith("Results freshness"))["rows"]
        for key, e in (fresh.get("leagues") or {}).items():
            row = next(r for r in rows if r[0] == e["name"])
            assert row[1] == e["newest_played"] and row[4] == e["overdue"]

    def test_basketball_invents_no_model_metrics(self, drafts):
        _, docs = drafts
        doc = docs["basketball"]
        assert doc["status"]["state"] == "none"
        html = render(doc).lower()
        assert "no nba model" in html
        card = dict(doc["report_card"]["blocks"][0]["rows"])
        assert card["Graded games"] == 0 and card["Accuracy"] is None and card["Log loss"] is None
        assert "what would be needed" in html


class TestSchedule:
    def test_weekday_mapping(self):
        assert mr.models_for("2026-11-02", None) == ["cfb", "soccer"]      # Monday
        assert mr.models_for("2026-11-03", None) == ["basketball"]         # Tuesday
        assert mr.models_for("2026-11-04", None) == ["nfl"]                # Wednesday
        assert mr.models_for("2026-11-05", None) == []                     # Thursday: nothing
        assert mr.models_for("2026-11-05", "all") == list(mr.MODELS)
        assert mr.models_for("2026-11-05", "nfl") == ["nfl"]


class TestFreshnessCheck:
    def _write(self, out: Path, model: str, date: str, generated_at: str) -> None:
        d = out / model
        d.mkdir(parents=True, exist_ok=True)
        (d / "latest.json").write_text(json.dumps({"model": model, "date": date, "generated_at": generated_at,
                                                   "status": {"state": "live"}}))

    def test_todays_draft_passes(self, tmp_path):
        now = datetime.now(timezone.utc)
        self._write(tmp_path, "nfl", now.date().isoformat(), now.isoformat(timespec="seconds"))
        logs = []
        assert mr.check_latest("nfl", now.date().isoformat(), tmp_path, log=logs.append) == 0
        assert logs[0].startswith("DRAFT OK")

    def test_old_draft_is_refused(self, tmp_path):
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=7)
        self._write(tmp_path, "nfl", old.date().isoformat(), old.isoformat(timespec="seconds"))
        logs = []
        assert mr.check_latest("nfl", now.date().isoformat(), tmp_path, log=logs.append) == 1
        assert logs[0].startswith("DRAFT OLD") and old.date().isoformat() in logs[0]

    def test_missing_draft_is_refused(self, tmp_path):
        logs = []
        assert mr.check_latest("soccer", "2026-11-02", tmp_path, log=logs.append) == 1
        assert logs[0].startswith("DRAFT MISSING")

    def test_cli_check(self, tmp_path):
        now = datetime.now(timezone.utc)
        self._write(tmp_path, "cfb", now.date().isoformat(), now.isoformat(timespec="seconds"))
        assert mr.main(["--check", "--model", "cfb", "--date", now.date().isoformat(), "--out", str(tmp_path)]) == 0
        assert mr.main(["--check", "--model", "nfl", "--date", now.date().isoformat(), "--out", str(tmp_path)]) == 1


class TestPromptDoc:
    def test_prompt_covers_the_required_steps(self):
        text = (REPO / "docs" / "weekly_model_review_prompt.md").read_text()
        for needle in ("git clone", "latest.json", "DRAFT MISSING", "DRAFT OLD", "stop",
                       "Check key numbers", "letter grade", "no grade (no model)", "em dashes",
                       "Do not push", ".html"):
            assert needle in text, needle
        assert "—" not in text


class TestTemplate:
    def test_page_never_contains_an_em_dash(self, drafts):
        out, docs = drafts
        for m in mr.MODELS:
            html = (out / m / "latest.html").read_text()
            assert "\u2014" not in html and "&mdash;" not in html, m
        html = render({"model": "x", "title": "A \u2014 B", "date": "2026-11-02",
                       "status": {"state": "live", "blocks": [{"kind": "kv", "rows": [["k", None]]}]}})
        assert "\u2014" not in html and "&mdash;" not in html


    def test_review_kind_renders_grade(self):
        html = render({"model": "nfl", "title": "NFL", "date": "2026-11-04", "kind": "review", "grade": "B",
                       "grade_reason": "n=108, edge -0.05 ± 0.03.", "status": {"state": "live", "blocks": []}})
        assert "<span class='grade'>B</span>" in html and "Grade B:" in html
        assert "Review for" in html
