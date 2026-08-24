"""Tests for the rolling xG-form feature layer (soccer/clubs/model/xg.py)."""

import pandas as pd

from soccer.clubs.model.xg import (
    MAX_AGE_DAYS,
    MIN_MATCHES,
    WINDOW,
    _Form,
    attach_xg,
    slate_diff,
)


def push_n(form, n, league="epl", home="A", away="B", start="2024-01-01",
           xg_home=2.0, xg_away=1.0):
    for i in range(n):
        d = (pd.Timestamp(start) + pd.Timedelta(days=7 * i)).date().isoformat()
        form.push(league, home, away, d, xg_home, xg_away)
    return d


class TestForm:
    def test_no_form_below_min_matches(self):
        f = _Form()
        push_n(f, MIN_MATCHES - 1)
        assert f.net("epl", "A", "2024-06-01") is None

    def test_net_is_rolling_mean_of_for_minus_against(self):
        f = _Form()
        last = push_n(f, MIN_MATCHES)  # A: 2.0 for, 1.0 against each match
        assert abs(f.net("epl", "A", last) - 1.0) < 1e-9
        assert abs(f.net("epl", "B", last) - (-1.0)) < 1e-9

    def test_window_caps_history(self):
        f = _Form()
        push_n(f, WINDOW, xg_home=0.0, xg_away=0.0)
        last = push_n(f, WINDOW, start="2025-01-01", xg_home=3.0, xg_away=0.0)
        # the zeros have rolled out entirely
        assert abs(f.net("epl", "A", last) - 3.0) < 1e-9

    def test_staleness_guard_voids_old_form(self):
        f = _Form()
        last = push_n(f, MIN_MATCHES)
        stale = (pd.Timestamp(last) + pd.Timedelta(days=MAX_AGE_DAYS + 1)).date().isoformat()
        assert f.net("epl", "A", stale) is None

    def test_form_is_per_league_pool(self):
        f = _Form()
        last = push_n(f, MIN_MATCHES, league="epl")
        assert f.net("serie_a", "A", last) is None


class TestSlateDiff:
    def test_one_sided_form_is_zero(self):
        f = _Form()
        last = push_n(f, MIN_MATCHES, home="A", away="B")
        # C has no history: the differential must not lean on A's form alone
        assert slate_diff(f, "epl", "A", "C", last) == 0.0

    def test_two_sided_form_diff(self):
        f = _Form()
        push_n(f, MIN_MATCHES, home="A", away="B", xg_home=2.0, xg_away=0.5)
        last = push_n(f, MIN_MATCHES, home="C", away="D", xg_home=1.0, xg_away=1.0)
        d = slate_diff(f, "epl", "A", "C", last)
        assert abs(d - 1.5) < 1e-9  # A net +1.5, C net 0.0


class TestAttachXg:
    def test_uncovered_league_rows_get_zero(self):
        hist = pd.DataFrame([
            {"date": "2024-01-01", "league": "serie_b",
             "home_team": "X", "away_team": "Y"},
        ])
        out = attach_xg(hist)
        assert (out["xg_net_diff"] == 0.0).all()

    def test_feature_is_strictly_pre_match(self):
        # A club's very first covered match must feature 0 — no lookahead.
        hist = pd.DataFrame([
            {"date": "2014-08-16", "league": "epl",
             "home_team": "Manchester United FC", "away_team": "Swansea City AFC"},
        ])
        out = attach_xg(hist)
        assert out["xg_net_diff"].iloc[0] == 0.0
