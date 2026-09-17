"""Tests for the soccer advanced pregame feature layer
(soccer/clubs/model/advanced.py): strict pre-match shifting, home/away
splits, staleness, rest and congestion from the whole calendar, and the
legacy-xG fill."""

import numpy as np
import pandas as pd
import pytest

from soccer.clubs.model import advanced as adv


def _matches(n=8, league="epl", teams=("A", "B", "C", "D")):
    """A round of matches per week: A-B, C-D alternating venues, with
    metrics that make the expected form easy to compute by hand."""
    rows = []
    for w in range(n):
        d = (pd.Timestamp("2024-08-10") + pd.Timedelta(days=7 * w)).date().isoformat()
        home, away = (teams[0], teams[1]) if w % 2 == 0 else (teams[1], teams[0])
        rows.append({"league": league, "season": "2024-25", "date": d, "match_id": str(w),
                     "home_team": home, "away_team": away, "goals_home": 1, "goals_away": 0,
                     "xg_home": 2.0, "xg_away": 1.0, "npxg_home": 1.8, "npxg_away": 0.9,
                     "xpts_home": 2.0, "xpts_away": 0.7,
                     "ppda_att_home": 100.0, "ppda_def_home": 10.0,
                     "ppda_att_away": 100.0, "ppda_def_away": 10.0,
                     "deep_home": 10.0, "deep_away": 4.0})
        c_home, c_away = (teams[2], teams[3]) if w % 2 == 0 else (teams[3], teams[2])
        rows.append({"league": league, "season": "2024-25", "date": d, "match_id": str(100 + w),
                     "home_team": c_home, "away_team": c_away, "goals_home": 0, "goals_away": 0,
                     "xg_home": 1.0, "xg_away": 1.0, "npxg_home": 1.0, "npxg_away": 1.0,
                     "xpts_home": 1.3, "xpts_away": 1.3,
                     "ppda_att_home": 150.0, "ppda_def_home": 10.0,
                     "ppda_att_away": 150.0, "ppda_def_away": 10.0,
                     "deep_home": 5.0, "deep_away": 5.0})
    return pd.DataFrame(rows)


def _shots(matches):
    s = matches[["league", "date", "home_team", "away_team"]].copy()
    s["shots_home"], s["shots_away"], s["sot_home"], s["sot_away"] = 10, 10, 4, 3
    return s


class TestTeamRows:
    def test_two_rows_per_match_with_for_and_against(self):
        m = _matches(1)
        long = adv.team_rows(adv.match_metrics(m, legacy_xg=m.iloc[0:0], shots=_shots(m)))
        assert len(long) == 4
        a = long[long.team == "A"].iloc[0]
        assert a.is_home == 1 and a.xg_for == 2.0 and a.xg_against == 1.0
        assert a.npxg_against == 0.9 and a.deep_against == 4.0
        assert a.ppda_for == pytest.approx(10.0)        # 100 / 10
        assert a.xg_per_shot == pytest.approx(0.2)
        assert a.deep_share == pytest.approx(10 / 14)


class TestForm:
    def test_form_is_strictly_pre_match_and_warms_up(self):
        m = _matches(8)
        long = adv.team_rows(adv.match_metrics(m, legacy_xg=m.iloc[0:0], shots=_shots(m)))
        form = adv.team_form(long)
        a = form[form.team == "A"].sort_values("date")
        # Fewer than MIN_MATCHES earlier rows -> NaN; then the mean of the
        # earlier rows only. A alternates 2.0 (home) / 1.0 (away) xG for.
        assert a["xg_for_r10"].iloc[:adv.MIN_MATCHES].isna().all()
        expected = np.mean([2.0, 1.0, 2.0, 1.0, 2.0])
        assert a["xg_for_r10"].iloc[adv.MIN_MATCHES] == pytest.approx(expected)
        # The row's own value is not in its feature: the 6th row (home,
        # xg 2.0) sees exactly the first five.
        assert a["xg_for_ewm"].iloc[adv.MIN_MATCHES] < 2.0

    def test_home_split_only_sees_home_matches(self):
        m = _matches(8)
        long = adv.team_rows(adv.match_metrics(m, legacy_xg=m.iloc[0:0], shots=_shots(m)))
        form = adv.team_form(long)
        a_home = form[(form.team == "A") & (form.is_home == 1)].sort_values("date")
        # A's home matches all have xg_for 2.0; the split never blends the 1.0s.
        assert a_home["xg_for_split"].dropna().eq(2.0).all()


class TestAttach:
    def test_features_are_differentials_and_a_new_fixture_gets_them(self):
        m = _matches(8)
        metrics = adv.match_metrics(m, legacy_xg=m.iloc[0:0], shots=_shots(m))
        cal = pd.DataFrame({"team": [], "league": [], "date": [], "played": [], "uefa": []})
        # A fixture after the data: A (home) v C.
        hist = pd.DataFrame([{"league": "epl", "season": "2024-25", "date": "2024-10-12",
                              "home_team": "A", "away_team": "C"}])
        out = adv.attach_advanced(hist, metrics, calendar=cal)
        # A's ewm xg_for is between 1 and 2; C's is 1.0 flat -> positive diff.
        assert out["xg_for_ewm_diff"].iloc[0] > 0
        assert out["npxg_for_ewm_diff"].iloc[0] > 0
        # A at home attacks 2.0; C away concedes 1.0 -> +1.0 matchup.
        assert out["home_att_vs_away_def"].iloc[0] == pytest.approx(1.0)
        assert out["ppda_diff"].iloc[0] == pytest.approx(-5.0)   # A allows 10 passes per action, C 15
        assert set(adv.ALL_ADVANCED) <= set(out.columns)

    def test_played_match_row_does_not_see_itself(self):
        m = _matches(8)
        metrics = adv.match_metrics(m, legacy_xg=m.iloc[0:0], shots=_shots(m))
        cal = pd.DataFrame({"team": [], "league": [], "date": [], "played": [], "uefa": []})
        hist = m[["league", "season", "date", "home_team", "away_team"]]
        out = adv.attach_advanced(hist, metrics, calendar=cal)
        # First MIN_MATCHES weeks: NaN for the A-B fixture (no history yet).
        ab = out[out.home_team.isin(["A", "B"])].sort_values("date")
        assert ab["xg_for_ewm_diff"].iloc[:adv.MIN_MATCHES].isna().all()
        assert ab["xg_for_ewm_diff"].iloc[adv.MIN_MATCHES:].notna().all()

    def test_stale_coverage_is_nan_not_a_number(self):
        m = _matches(8)
        metrics = adv.match_metrics(m, legacy_xg=m.iloc[0:0], shots=_shots(m))
        cal = pd.DataFrame({"team": [], "league": [], "date": [], "played": [], "uefa": []})
        far = (pd.Timestamp(m["date"].max()) + pd.Timedelta(days=adv.MAX_AGE_DAYS + 1)).date().isoformat()
        hist = pd.DataFrame([{"league": "epl", "season": "2025-26", "date": far,
                              "home_team": "A", "away_team": "C"}])
        out = adv.attach_advanced(hist, metrics, calendar=cal)
        assert np.isnan(out["xg_for_ewm_diff"].iloc[0])

    def test_legacy_xg_fills_uncovered_matches(self):
        m = _matches(2)
        legacy = pd.DataFrame([{"league": "epl", "date": "2024-08-24", "home_team": "C",
                                "away_team": "D", "xg_home": 0.5, "xg_away": 0.4}])
        proc = m[m.home_team == "A"]
        metrics = adv.match_metrics(proc, legacy_xg=legacy, shots=_shots(m))
        assert set(metrics["source"]) == {"understat", "legacy_xg"}
        row = metrics[metrics.source == "legacy_xg"].iloc[0]
        assert row.xg_home == 0.5 and np.isnan(row.npxg_home)


class TestRestAndCongestion:
    def test_from_the_whole_calendar(self):
        cal = pd.DataFrame([
            {"team": "A", "league": "epl", "date": "2024-09-01", "played": True, "uefa": False},
            {"team": "A", "league": "epl", "date": "2024-09-04", "played": True, "uefa": True},
            {"team": "A", "league": "epl", "date": "2024-09-14", "played": False, "uefa": False},
            {"team": "B", "league": "epl", "date": "2024-08-20", "played": True, "uefa": False},
        ])
        rows = pd.DataFrame({"team": ["A", "B", "Z"], "date": ["2024-09-07"] * 3})
        r = adv.rest_and_congestion(rows, cal)
        assert r["rest"].tolist()[0] == 3            # since the European tie on the 4th
        assert r["congestion14"].tolist()[0] == 2    # 09-01 and 09-04; 09-14 is the future
        assert r["uefa7"].tolist()[0] == 1
        assert r["rest"].tolist()[1] == 18 and r["uefa7"].tolist()[1] == 0
        assert np.isnan(r["rest"].tolist()[2]) and r["congestion14"].tolist()[2] == 0


class TestCoverage:
    def test_coverage_table(self):
        m = _matches(2)
        metrics = adv.match_metrics(m, legacy_xg=m.iloc[0:0], shots=_shots(m))
        cov = adv.coverage_by_season(metrics)
        assert cov.iloc[0]["matches"] == 4 and cov.iloc[0]["with_npxg"] == 4
