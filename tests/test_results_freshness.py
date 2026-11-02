"""Results freshness: the per-league overdue-fixture check that fails the
soccer workflow when finals stop arriving (openfootball stalled on
2026-09-20 and six weeks of runs stayed green), the season-aware feed
tolerance, and the warnings the email carries."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from soccer.clubs.daily import emails, freshness_check, state
from soccer.clubs.daily.config import RESULTS_OVERDUE_MAX


def results(rows):
    return pd.DataFrame(rows, columns=["date", "season", "league", "home_team", "away_team",
                                       "home_score", "away_score"])


def fixture(date, league, home, away, hs=None, as_=None, season="2026-27"):
    return (date, season, league, home, away, hs, as_)


def _round(league, date, played, n=10):
    """One matchday: n fixtures on one date, scored or not."""
    return [fixture(date, league, f"H{i}", f"A{i}", 1 if played else None, 0 if played else None)
            for i in range(n)]


class TestResultsStatus:
    def test_stalled_feed_reads_stale(self):
        rows = _round("epl", "2026-09-19", True) + _round("epl", "2026-09-26", False) \
             + _round("epl", "2026-10-03", False)
        st = state.results_status(results(rows), "2026-10-06")
        e = st["leagues"]["epl"]
        assert e["stale"] and "epl" in st["stale_leagues"]
        assert e["overdue"] == 20 and e["newest_played"] == "2026-09-19"
        assert e["oldest_overdue"] == "2026-09-26"
        assert len(e["overdue_fixtures"]) == 10
        assert st["as_of"] == "2026-09-19"

    def test_isolated_postponement_is_not_stale(self):
        rows = _round("la_liga", "2026-10-03", True)
        rows[0] = fixture("2026-10-03", "la_liga", "H0", "A0")      # one postponed match
        rows += _round("la_liga", "2026-10-17", True)
        st = state.results_status(results(rows), "2026-10-20")
        e = st["leagues"]["la_liga"]
        assert not e["stale"] and e["overdue"] == 1
        assert e["overdue_fixtures"] == ["2026-10-03 H0 v A0"]

    def test_threshold_is_the_configured_count(self):
        rows = _round("serie_a", "2026-10-03", True)
        for i in range(RESULTS_OVERDUE_MAX):
            rows[i] = fixture("2026-10-03", "serie_a", f"H{i}", f"A{i}")
        assert not state.results_status(results(rows), "2026-10-10")["leagues"]["serie_a"]["stale"]
        rows[RESULTS_OVERDUE_MAX] = fixture("2026-10-03", "serie_a", "Hx", "Ax")
        assert state.results_status(results(rows), "2026-10-10")["leagues"]["serie_a"]["stale"]

    def test_grace_period_covers_last_nights_matches(self):
        rows = _round("bundesliga", "2026-10-31", False)      # yesterday, not yet scored
        st = state.results_status(results(rows), "2026-11-01")
        assert st["leagues"]["bundesliga"]["overdue"] == 0
        assert not st["leagues"]["bundesliga"]["stale"]

    def test_breaks_schedule_nothing_so_overdue_nothing(self):
        # Winter break: last round on Dec 20, next on Jan 10, run on Jan 5.
        rows = _round("bundesliga", "2026-12-20", True) + _round("bundesliga", "2027-01-10", False)
        st = state.results_status(results(rows), "2027-01-05")
        assert not st["leagues"]["bundesliga"]["stale"]
        assert st["leagues"]["bundesliga"]["newest_played_age_days"] == 16

    def test_second_division_without_fixtures_is_reported_not_stale(self):
        rows = _round("epl", "2026-10-03", True)
        st = state.results_status(results(rows), "2026-10-06")
        assert st["leagues"]["bundesliga_2"]["reason"] == "no_fixtures"
        assert not st["leagues"]["bundesliga_2"]["stale"]
        assert st["leagues"]["mls"]["reason"] == "no_fixture_list"

    def test_live_data_is_internally_consistent(self):
        """The committed results.csv, on whatever day it was last written:
        every league's verdict follows the rule, and the as-of date is the
        newest final across the fixture-list leagues. (The first version of
        this test pinned the Championship as stale; it stopped being true
        on 2026-11-02 when football-data.co.uk filled it, which is the
        point of the overlay.)"""
        from soccer.clubs.model.elo import DATA_DIR
        r = pd.read_csv(DATA_DIR / "results.csv")
        run_date = str(r.dropna(subset=["home_score"])["date"].max())
        st = state.results_status(r, run_date)
        for key, e in st["leagues"].items():
            has_fixtures = e["reason"] not in ("no_fixtures", "no_fixture_list")
            assert e["stale"] == (has_fixtures and e["overdue"] > RESULTS_OVERDUE_MAX), key
            assert (key in st["stale_leagues"]) == e["stale"]
        newest = [e["newest_played"] for k, e in st["leagues"].items()
                  if e["reason"] != "no_fixture_list" and e["newest_played"]]
        assert st["as_of"] == max(newest)


class TestSeasonAwareFeedTolerance:
    def test_in_season_tightens_the_report_tolerance(self):
        in_season = results(_round("epl", "2026-10-30", True))
        feeds = state.feed_status("2026-11-02", in_season)
        assert all(r.tolerance_days == 10 for r in feeds.values())
        off_season = results(_round("epl", "2026-05-24", True))
        feeds = state.feed_status("2026-07-01", off_season)
        assert all(r.tolerance_days == 130 for r in feeds.values())

    def test_six_week_old_feed_cannot_read_fresh_in_season(self):
        from common import freshness
        in_season = results(_round("epl", "2026-10-30", True))
        assert state.season_active(in_season, "2026-11-02")
        report = freshness.check("shots", pd.DataFrame({"date": ["2026-09-20"]}), "date",
                                 "2026-11-02", 10)
        assert not report.fresh and report.age_days == 43


class TestWorkflowGate:
    def _payload(self, stale: bool) -> dict:
        return {"run_date": "2026-11-02", "results_freshness": {
            "as_of": "2026-11-01", "run_date": "2026-11-02",
            "stale_leagues": ["championship"] if stale else [],
            "leagues": {
                "epl": {"name": "Premier League", "newest_played": "2026-10-31", "overdue": 0,
                        "stale": False, "reason": None},
                "championship": {"name": "Championship", "newest_played": "2026-09-20",
                                 "overdue": 61, "stale": stale,
                                 "reason": "61 fixtures dated on or before 2026-10-31 have no result (limit 3); newest result 2026-09-20"},
            }}}

    def test_stale_fixture_fails_the_check(self, tmp_path):
        p = tmp_path / "latest.json"
        p.write_text(json.dumps(self._payload(True)))
        logs = []
        stale = freshness_check.check(json.loads(p.read_text()), log=logs.append)
        assert stale == ["championship"]
        assert any(l.startswith("::error::Championship results are stale") for l in logs)
        assert freshness_check.main(["--latest", str(p)]) == 1

    def test_current_data_passes(self, tmp_path):
        p = tmp_path / "latest.json"
        p.write_text(json.dumps(self._payload(False)))
        assert freshness_check.main(["--latest", str(p)]) == 0

    def test_missing_block_fails(self, tmp_path):
        p = tmp_path / "latest.json"
        p.write_text(json.dumps({"run_date": "2026-11-02"}))
        assert freshness_check.main(["--latest", str(p)]) == 1
        assert freshness_check.main(["--latest", str(tmp_path / "nope.json")]) == 1


class TestEmailWarnings:
    def test_banner_and_stale_tags(self):
        fresh = {"as_of": "2026-11-01", "stale_leagues": ["championship"], "leagues": {
            "epl": {"name": "Premier League", "newest_played": "2026-10-31", "stale": False, "reason": None},
            "championship": {"name": "Championship", "newest_played": "2026-09-20", "stale": True,
                             "reason": "61 fixtures have no result"},
        }}
        fixtures = pd.DataFrame([
            {"date": "2026-11-07", "league": "championship", "home_team": "A", "away_team": "B",
             "p_H": 0.4, "p_D": 0.3, "p_A": 0.3, "pick": "H", "score_home": 1, "score_away": 0},
        ])
        futures = {"championship": {"season": "2026-27", "clubs": [
            {"team": "A", "points": 10, "exp_points": 60.0, "exp_position": 1.0, "p_title": 0.5,
             "p_top4": 0.9, "p_uel": 0.05, "p_relegation": 0.0}], "sims": 10}}
        html = emails.update_html("2026-11-02", fixtures, pd.DataFrame(), {"graded": 0}, futures,
                                  freshness=fresh)
        assert "Results as of 2026-11-01" in html
        assert "Championship results are stale" in html
        assert html.count("STALE") >= 3          # banner + fixtures heading + forecast heading
        assert "Premier League 2026-10-31" in html
        # No freshness block: the email renders exactly as before.
        plain = emails.update_html("2026-11-02", fixtures, pd.DataFrame(), {"graded": 0}, futures)
        assert "Results as of" not in plain and "STALE" not in plain
