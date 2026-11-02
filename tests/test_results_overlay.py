"""The secondary-source overlay that keeps results.csv current when
openfootball stalls (it did on 2026-09-20), and the football-data.co.uk
results parser that feeds the Championship half of it."""

from __future__ import annotations

import csv
import json

import pytest

from soccer.clubs.data import fd_results, results_overlay as ro


def row(date, league, home, away, hs="", as_="", season="2026-27"):
    return {"date": date, "season": season, "league": league, "home_team": home,
            "away_team": away, "home_score": hs, "away_score": as_}


def src(date, league, home, away, hs, as_, season="2026-27"):
    return {"date": date, "season": season, "league": league, "home_team": home,
            "away_team": away, "home_score": str(hs), "away_score": str(as_)}


def quiet(*_a, **_k):
    pass


class TestApply:
    def test_fills_unplayed_fixture_on_exact_key(self):
        rows = [row("2026-10-03", "epl", "A", "B")]
        out, rep = ro.apply(rows, [("understat", [src("2026-10-03", "epl", "A", "B", 2, 1)])], log=quiet)
        assert (out[0]["home_score"], out[0]["away_score"]) == ("2", "1")
        assert rep["sources"]["understat"]["filled"] == 1
        assert rep["by_league"]["epl"]["newest_filled"] == "2026-10-03"

    def test_primary_score_is_never_overwritten_and_conflict_is_logged(self):
        rows = [row("2026-10-03", "epl", "A", "B", "3", "0")]
        logs = []
        out, rep = ro.apply(rows, [("understat", [src("2026-10-03", "epl", "A", "B", 0, 0)])],
                            log=logs.append)
        assert (out[0]["home_score"], out[0]["away_score"]) == ("3", "0")
        assert rep["conflicts"][0]["kind"] == "score_mismatch"
        assert rep["conflicts"][0]["primary_score"] == [3, 0]
        assert rep["conflicts"][0]["source_score"] == [0, 0]
        assert any("keeping openfootball" in l for l in logs)

    def test_provisional_date_is_moved_inside_the_window(self):
        rows = [row("2026-10-25", "la_liga", "A", "B")]
        out, rep = ro.apply(rows, [("understat", [src("2026-10-23", "la_liga", "A", "B", 1, 1)])], log=quiet)
        assert out[0]["date"] == "2026-10-23"
        assert out[0]["home_score"] == "1"
        assert rep["sources"]["understat"]["redated"] == 1
        assert len(out) == 1          # moved, not duplicated

    def test_pairing_outside_the_window_is_left_alone(self):
        # The reverse-fixture trap: the same ordered pairing exists months
        # away. It must not be filled from a source row dated now.
        rows = [row("2027-03-06", "ligue_1", "PSG", "Rennes")]
        out, rep = ro.apply(rows, [("understat", [src("2026-08-23", "ligue_1", "PSG", "Rennes", 2, 2)])], log=quiet)
        assert out[0]["home_score"] == "" and out[0]["date"] == "2027-03-06"
        assert rep["sources"]["understat"]["filled"] == 0
        assert rep["unmatched"][0]["kind"] == "no_fixture"

    def test_swapped_home_away_is_reported_not_merged(self):
        rows = [row("2026-08-23", "ligue_1", "Rennes", "PSG", "2", "2")]
        out, rep = ro.apply(rows, [("understat", [src("2026-08-23", "ligue_1", "PSG", "Rennes", 2, 2)])], log=quiet)
        assert len(out) == 1
        assert rep["conflicts"][0]["kind"] == "home_away_swapped"
        assert rep["conflicts"][0]["primary_date"] == "2026-08-23"

    def test_source_rows_are_never_appended(self):
        rows = [row("2026-10-03", "epl", "A", "B")]
        out, rep = ro.apply(rows, [("understat", [src("2026-10-03", "epl", "X", "Y", 1, 0)])], log=quiet)
        assert len(out) == 1
        assert rep["sources"]["understat"]["unmatched"] == 1

    def test_precedence_first_source_wins_second_cross_checks(self):
        rows = [row("2026-10-03", "epl", "A", "B")]
        sources = [("understat", [src("2026-10-03", "epl", "A", "B", 2, 1)]),
                   ("football-data", [src("2026-10-03", "epl", "A", "B", 1, 1)])]
        out, rep = ro.apply(rows, sources, log=quiet)
        assert (out[0]["home_score"], out[0]["away_score"]) == ("2", "1")
        assert rep["sources"]["football-data"]["conflicts"] == 1
        assert rep["conflicts"][0]["source"] == "football-data"

    def test_source_league_scope(self):
        # Understat does not cover the Championship; a row claiming to
        # must not be applied even if it matches a fixture.
        rows = [row("2026-10-03", "championship", "A", "B")]
        out, rep = ro.apply(rows, [("understat", [src("2026-10-03", "championship", "A", "B", 2, 1)])], log=quiet)
        assert out[0]["home_score"] == ""
        assert rep["sources"]["understat"]["skipped_league"] == 1
        out, rep = ro.apply(rows, [("football-data", [src("2026-10-03", "championship", "A", "B", 2, 1)])], log=quiet)
        assert out[0]["home_score"] == "2"

    def test_idempotent(self):
        rows = [row("2026-10-25", "la_liga", "A", "B"), row("2026-10-03", "epl", "C", "D")]
        sources = [("understat", [src("2026-10-23", "la_liga", "A", "B", 1, 1),
                                  src("2026-10-03", "epl", "C", "D", 0, 2)])]
        once, rep1 = ro.apply([dict(r) for r in rows], sources, log=quiet)
        twice, rep2 = ro.apply([dict(r) for r in once], sources, log=quiet)
        assert once == twice
        assert rep2["sources"]["understat"]["filled"] == 0
        assert rep2["sources"]["understat"]["agree"] == 2

    def test_other_leagues_pass_through_untouched(self):
        rows = [row("2026-10-03", "mls", "A", "B", "1", "0", season="2026")]
        out, _ = ro.apply(rows, [("understat", [])], log=quiet)
        assert out == [row("2026-10-03", "mls", "A", "B", "1", "0", season="2026")]


class TestIO:
    def test_readers_skip_unplayed_and_write_roundtrip(self, tmp_path):
        u = tmp_path / "u.csv"
        with u.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["league", "season", "date", "home_team", "away_team",
                                              "goals_home", "goals_away"])
            w.writeheader()
            w.writerow({"league": "epl", "season": "2026-27", "date": "2026-10-03",
                        "home_team": "A", "away_team": "B", "goals_home": "2", "goals_away": "1"})
            w.writerow({"league": "epl", "season": "2026-27", "date": "2026-10-04",
                        "home_team": "C", "away_team": "D", "goals_home": "", "goals_away": ""})
        got = ro.understat_rows(u)
        assert got == [src("2026-10-03", "epl", "A", "B", 2, 1)]
        assert ro.understat_rows(tmp_path / "missing.csv") == []
        out = tmp_path / "results.csv"
        ro.write_results([row("2026-10-04", "epl", "C", "D"), row("2026-10-03", "epl", "A", "B", "2", "1")], out)
        back = ro.read_results(out)
        assert [r["date"] for r in back] == ["2026-10-03", "2026-10-04"]
        rep_path = tmp_path / "r.json"
        ro.write_report({"a": 1}, rep_path)
        assert json.loads(rep_path.read_text()) == {"a": 1}


FD_SAMPLE = (
    "﻿Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR\n"
    "E1,08/08/26,20:00,Wrexham,Sheffield Weds,2,1,H,1,0,H\n"
    "E1,09/08/26,15:00,West Brom,QPR,0,0,D,0,0,D\n"
    "E1,09/08/26,15:00,Mystery FC,Wolves,1,3,A,0,1,A\n"
    "E1,10/08/26,15:00,Preston,Charlton,,,,,,\n"
    "\n"
)


class TestFootballDataParser:
    def test_parses_finals_maps_names_and_reports_unknowns(self):
        rows, unmatched = fd_results.parse_csv(FD_SAMPLE, "championship", "2026-27")
        assert unmatched == {"Mystery FC"}
        assert [(r["date"], r["home_team"], r["away_team"], r["home_score"], r["away_score"]) for r in rows] == [
            ("2026-08-08", "Wrexham AFC", "Sheffield Wednesday FC", 2, 1),
            ("2026-08-09", "West Bromwich Albion FC", "Queens Park Rangers FC", 0, 0),
        ]
        assert all(r["league"] == "championship" and r["season"] == "2026-27" for r in rows)

    def test_championship_aliases_cover_the_current_clubs(self):
        import pandas as pd
        r = pd.read_csv(ro.RESULTS_CSV, usecols=["season", "league", "home_team", "away_team"])
        s = r[(r["season"] == "2026-27") & (r["league"] == "championship")]
        clubs = set(s["home_team"]) | set(s["away_team"])
        assert clubs <= set(fd_results.CHAMPIONSHIP_ALIASES.values())

    def test_merge_replaces_a_refetched_league_season_only(self, tmp_path):
        out = tmp_path / "fd.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fd_results.COLUMNS)
            w.writeheader()
            w.writerow({"league": "championship", "season": "2025-26", "date": "2025-08-09",
                        "home_team": "A", "away_team": "B", "home_score": "1", "away_score": "0"})
            w.writerow({"league": "championship", "season": "2026-27", "date": "2026-08-09",
                        "home_team": "C", "away_team": "D", "home_score": "0", "away_score": "0"})
        new = [{"league": "championship", "season": "2026-27", "date": "2026-08-09",
                "home_team": "C", "away_team": "D", "home_score": 2, "away_score": 2}]
        merged = fd_results.merge(new, out)
        assert [(r["season"], r["home_score"]) for r in merged] == [("2025-26", "1"), ("2026-27", "2")]
