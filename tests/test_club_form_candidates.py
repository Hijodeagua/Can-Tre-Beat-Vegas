"""Tests for the two results-derived form candidates
(soccer/clubs/model/goals.py and momentum.py).

Neither ships in the live outcome model yet — `eval_goals_form.py` is
where that is decided — but both are wired to the same shared machinery
as the xG and shots layers, so the contract they have to honour is the
same: strictly pre-match, warm-up minimum, staleness guard, and a zero
where only one side has form.
"""

import pandas as pd

from soccer.clubs.model import goals, momentum


def _history(rows):
    return pd.DataFrame(rows)


class TestGoalForm:
    def test_net_is_rolling_goal_difference(self):
        f = goals._Form()
        for i in range(goals.MIN_MATCHES):
            d = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * i)).date().isoformat()
            f.push("epl", "A", "B", d, 3.0, 1.0)
        assert abs(f.net("epl", "A", d) - 2.0) < 1e-9
        assert abs(f.net("epl", "B", d) + 2.0) < 1e-9

    def test_unplayed_fixtures_push_nothing(self, tmp_path, monkeypatch):
        csv = tmp_path / "results.csv"
        csv.write_text(
            "date,season,league,home_team,away_team,home_score,away_score\n"
            "2026-08-01,2026-27,epl,A,B,2,0\n"
            "2026-08-08,2026-27,epl,B,A,,\n"
        )
        monkeypatch.setattr(goals, "RESULTS_CSV", csv)
        values = goals.match_values()
        assert list(values) == [("epl", "2026-08-01", "A", "B")]

    def test_attach_is_strictly_pre_match(self, tmp_path, monkeypatch):
        # One club, enough matches to warm up, then the row under test:
        # its own result must not be inside its own feature.
        rows, csv_rows = [], []
        for i in range(goals.MIN_MATCHES + 1):
            d = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=7 * i)).date().isoformat()
            rows.append({"date": d, "league": "epl", "home_team": "A", "away_team": "B"})
            csv_rows.append(f"{d},2026-27,epl,A,B,5,0")
        csv = tmp_path / "results.csv"
        csv.write_text("date,season,league,home_team,away_team,home_score,away_score\n"
                       + "\n".join(csv_rows) + "\n")
        monkeypatch.setattr(goals, "RESULTS_CSV", csv)
        out = goals.attach_goals(_history(rows))
        # The first MIN_MATCHES rows are warm-up (0). The last sees only
        # the matches before it: A is +5 a game and B is −5, and the
        # feature is the differential, so +10 rather than +15 — its own
        # 5-0 is not in it.
        assert list(out["goals_net_diff"])[:goals.MIN_MATCHES] == [0.0] * goals.MIN_MATCHES
        assert abs(out["goals_net_diff"].iloc[-1] - 10.0) < 1e-9


class TestSurpriseForm:
    def test_surprise_is_actual_minus_expected(self):
        # Home wins a match it was expected to win 40% of the time: the
        # surprise is +0.6, split symmetrically between the two sides.
        history = _history([{
            "date": "2026-01-01", "league": "epl", "home_team": "A",
            "away_team": "B", "exp_home": 0.4, "actual_home": 1.0,
        }])
        (home, away), = momentum.match_values(history).values()
        assert abs(home - 0.3) < 1e-9 and abs(away + 0.3) < 1e-9

    def test_beating_the_rating_reads_positive(self):
        rows = []
        for i in range(momentum.MIN_MATCHES + 1):
            d = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=7 * i)).date().isoformat()
            rows.append({"date": d, "league": "epl", "home_team": "A",
                         "away_team": "B", "exp_home": 0.5, "actual_home": 1.0})
        out = momentum.attach_momentum(_history(rows))
        # A has been winning coin-flips: its form is positive, and the
        # differential against the side it kept beating is twice that.
        assert out["surprise_net_diff"].iloc[-1] > 0
        assert abs(out["surprise_net_diff"].iloc[-1] - 1.0) < 1e-9
        assert list(out["surprise_net_diff"])[:momentum.MIN_MATCHES] == \
            [0.0] * momentum.MIN_MATCHES

    def test_a_rating_that_predicted_everything_has_no_form(self):
        rows = []
        for i in range(momentum.MIN_MATCHES + 1):
            d = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=7 * i)).date().isoformat()
            rows.append({"date": d, "league": "epl", "home_team": "A",
                         "away_team": "B", "exp_home": 1.0, "actual_home": 1.0})
        out = momentum.attach_momentum(_history(rows))
        assert abs(out["surprise_net_diff"].iloc[-1]) < 1e-9

    def test_missing_columns_degrade_to_zero(self):
        history = _history([{"date": "2026-01-01", "league": "epl",
                             "home_team": "A", "away_team": "B"}])
        assert momentum.match_values(history) == {}
        assert list(momentum.attach_momentum(history)["surprise_net_diff"]) == [0.0]
