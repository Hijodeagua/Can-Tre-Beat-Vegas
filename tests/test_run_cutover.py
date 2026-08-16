"""Regression test for the model-version cutover (mlb/daily/run.py).

Before this fix, the daily pipeline only ever applied the pitcher/rest/
travel adjustment to whichever slate was built for the *shadow* role -
`adjustments=` was hardcoded onto that one call site and never onto the
"active" one. Flipping `ACTIVE_MODEL = MODEL_V2` in config.py alone would
therefore have silently relabeled the plain team-Elo slate as "v2-sp"
without ever computing the adjustment. `build_slate_for_version` fixes this
by keying the adjustment strictly off which model_version is requested, not
off which role (active/shadow) is asking for it - this test asserts that
directly, and would fail against the old shadow-only wiring.
"""

import pytest

from mlb.daily import simulate
from mlb.daily.config import MODEL_V1, MODEL_V2
from mlb.daily.run import build_slate_for_version


@pytest.fixture
def params():
    return simulate.ScoreParams(
        margin_intercept=0.1, margin_slope=8.0, total_mean=9.0, dispersion=4.5
    )


RATINGS = {"HOU": 1550.0, "STL": 1490.0}
TODAYS = [{
    "date": "2026-08-08", "away": "HOU", "home": "STL",
    "away_fr": "HOU", "home_fr": "STL", "game_num": 1,
    "away_sp": "Framber Valdez", "home_sp": "", "away_sp_id": "664285",
}]
ADJUSTMENTS = {("2026-08-08", 1, "STL"): {
    "home_adj": 0.0, "away_adj": 30.0,
    "home_sp_adj": 0.0, "away_sp_adj": 30.0,
    "home_sp_mode": "staff", "away_sp_mode": "pitcher",
    "home_rt_adj": 0.0, "away_rt_adj": 0.0,
}}


def test_v2_gets_adjustments_regardless_of_which_call_site_asks(params):
    """The defining regression: call the builder twice with the same
    adjustments dict, once as if it were building the "active" slate and
    once as if "shadow" - model_version is the only thing that should ever
    decide whether the adjustment is applied."""
    for role_label in ("active", "shadow"):  # role never enters the call
        v2 = build_slate_for_version(
            MODEL_V2, RATINGS, TODAYS, params, None, ADJUSTMENTS, n=500)
        assert v2.iloc[0].away_sp_adj == 30.0
        assert v2.iloc[0].model_version == MODEL_V2


def test_v1_never_gets_adjustments_even_if_supplied(params):
    """v1 must stay adjustment-free even when a non-empty adjustments dict
    is in scope (e.g. because v2 is running alongside it as a shadow)."""
    v1 = build_slate_for_version(
        MODEL_V1, RATINGS, TODAYS, params, None, ADJUSTMENTS, n=500)
    assert "away_sp_adj" not in v1.columns
    assert v1.iloc[0].model_version == MODEL_V1


def test_v2_probability_actually_moves_vs_v1(params):
    """Not just a label swap: v2's win probability must differ from v1's on
    the same matchup, since that's the entire point of the adjustment."""
    v1 = build_slate_for_version(
        MODEL_V1, RATINGS, TODAYS, params, None, ADJUSTMENTS, n=500)
    v2 = build_slate_for_version(
        MODEL_V2, RATINGS, TODAYS, params, None, ADJUSTMENTS, n=500)
    # +30 Elo to the away starter lowers the home win probability.
    assert v2.iloc[0].p_home < v1.iloc[0].p_home


def test_no_adjustments_available_is_a_safe_no_op(params):
    """When adjustments is None (e.g. no games today, or neither active nor
    shadow model uses them), v2 must not error - it just runs adjustment-
    free for that call, same as v1."""
    v2 = build_slate_for_version(
        MODEL_V2, RATINGS, TODAYS, params, None, None, n=500)
    assert "away_sp_adj" not in v2.columns
