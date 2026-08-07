"""Tests for the daily MLB pipeline: statsapi parsing, score sim,
season Monte Carlo seeding, and grading math."""

import numpy as np
import pandas as pd
import pytest

from mlb.daily import grade, simulate
from mlb.daily.config import ALL_TEAMS, DIVISIONS
from mlb.daily.update_games import parse_games


def _api_game(away, home, state="F", away_score=None, home_score=None, num=1,
              away_sp=None, home_sp=None):
    g = {
        "gameNumber": num,
        "status": {"codedGameState": state},
        "teams": {
            "away": {"team": {"name": away}},
            "home": {"team": {"name": home}},
        },
    }
    if away_score is not None:
        g["teams"]["away"]["score"] = away_score
        g["teams"]["home"]["score"] = home_score
    if away_sp is not None:
        g["teams"]["away"]["probablePitcher"] = {"id": 1, "fullName": away_sp}
    if home_sp is not None:
        g["teams"]["home"]["probablePitcher"] = {"id": 2, "fullName": home_sp}
    return g


def test_parse_games_splits_finals_and_upcoming():
    payload = {"dates": [{
        "date": "2026-08-06",
        "games": [
            _api_game("Houston Astros", "St. Louis Cardinals", "F", 9, 3),
            _api_game("New York Yankees", "Boston Red Sox", "S",
                      away_sp="Gerrit Cole"),  # home SP not yet announced
            _api_game("Athletics", "Seattle Mariners", "F", 2, 4, num=2),
            _api_game("Toronto Blue Jays", "Detroit Tigers", "D"),  # postponed
            _api_game("Fake Exhibition Team", "Boston Red Sox", "F", 1, 2),
        ],
    }]}
    finals, upcoming = parse_games(payload)
    assert len(finals) == 2 and len(upcoming) == 1
    hou = finals[0]
    assert (hou["away_fr"], hou["home_fr"]) == ("HOU", "STL")
    assert (hou["away_score"], hou["home_score"]) == (9, 3)
    assert finals[1]["game_num"] == 2
    assert upcoming[0]["home_fr"] == "BOS"
    # Probable starters ride along on upcoming games; missing -> empty string.
    assert upcoming[0]["away_sp"] == "Gerrit Cole"
    assert upcoming[0]["home_sp"] == ""


@pytest.fixture
def params():
    return simulate.ScoreParams(
        margin_intercept=0.1, margin_slope=8.0, total_mean=9.0, dispersion=4.5
    )


def test_simulate_game_score_consistent_with_pick(params):
    for p_home in (0.35, 0.5, 0.62, 0.8):
        r = simulate.simulate_game(p_home, params, n=4000, seed=1)
        assert r["pick_home"] == (p_home >= 0.5)
        if r["pick_home"]:
            assert r["home_score"] > r["away_score"]
        else:
            assert r["away_score"] > r["home_score"]
        # Sim win rate should be in the neighborhood of the Elo probability.
        assert abs(r["sim_home_win_rate"] - p_home) < 0.12


def test_stronger_team_gets_bigger_margin(params):
    weak = simulate.simulate_game(0.55, params, n=8000, seed=2)
    strong = simulate.simulate_game(0.75, params, n=8000, seed=2)
    assert (strong["home_score"] - strong["away_score"]) >= (
        weak["home_score"] - weak["away_score"]
    )


def _uniform_standings():
    return pd.DataFrame(
        {"team": ALL_TEAMS, "wins": 60, "losses": 50,
         "run_diff": [0] * len(ALL_TEAMS)}
    )


def test_season_sim_probabilities_sum():
    ratings = {t: 1500.0 for t in ALL_TEAMS}
    ratings["LAD"] = 1650.0  # clear favorite
    remaining = [
        {"date": "2026-09-01", "home_fr": h, "away_fr": a, "game_num": 1}
        for h in ["LAD", "SFG", "SDP"] for a in ["ARI", "COL"]
    ]
    futures = simulate.simulate_season(
        ratings, remaining, _uniform_standings(), n_sims=400, seed=3
    )
    # Each division crowns exactly one champ per sim.
    for teams in DIVISIONS.values():
        assert futures[futures.team.isin(teams)].division_pct.sum() == pytest.approx(1.0)
    # Six playoff teams per league, one top seed each.
    assert futures.playoff_pct.sum() == pytest.approx(12.0)
    assert futures.top_seed_pct.sum() == pytest.approx(2.0)
    # With near-tied standings and few games left, the boosted team should
    # still be the clear division favorite (random tiebreaks put everyone
    # else near the 1/5 baseline).
    nl_west = futures[futures.team.isin(DIVISIONS["NL West"])].set_index("team")
    assert nl_west.division_pct.idxmax() == "LAD"
    assert nl_west.loc["LAD", "division_pct"] > 0.25


def test_grading_math(tmp_path, monkeypatch):
    monkeypatch.setattr(grade, "PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr(grade, "GRADES_CSV", tmp_path / "grades.csv")

    slate = pd.DataFrame([
        # correct pick, exact score hit
        {"date": "2026-08-06", "away": "HOU", "home": "STL", "game_num": 1,
         "p_home": 0.38, "pick": "HOU", "pick_prob": 0.62,
         "pred_home_score": 3, "pred_away_score": 9,
         "elo_home": 1500, "elo_away": 1550},
        # wrong pick
        {"date": "2026-08-06", "away": "NYY", "home": "BOS", "game_num": 1,
         "p_home": 0.45, "pick": "NYY", "pick_prob": 0.55,
         "pred_home_score": 4, "pred_away_score": 5,
         "elo_home": 1510, "elo_away": 1520},
        # postponed - never shows up in actuals
        {"date": "2026-08-06", "away": "TOR", "home": "DET", "game_num": 1,
         "p_home": 0.50, "pick": "DET", "pick_prob": 0.50,
         "pred_home_score": 5, "pred_away_score": 4,
         "elo_home": 1500, "elo_away": 1500},
    ])
    slate.to_csv(grade.slate_path("2026-08-06"), index=False)

    games = pd.DataFrame([
        {"date": "2026-08-06", "season": 2026, "game_num": 1,
         "away_fr": "HOU", "home_fr": "STL", "away_score": 9, "home_score": 3},
        {"date": "2026-08-06", "season": 2026, "game_num": 1,
         "away_fr": "NYY", "home_fr": "BOS", "away_score": 2, "home_score": 6},
    ])

    graded = grade.grade_day("2026-08-06", games)
    row = grade.update_ledger("2026-08-06", graded)

    assert int(row["games"]) == 2
    assert int(row["correct"]) == 1
    assert row["accuracy"] == pytest.approx(0.5)
    assert int(row["skipped"]) == 1
    # log-loss by hand: -[ln(1-0.38) + ln(0.45)]/2
    expected_ll = -(np.log(1 - 0.38) + np.log(0.45)) / 2
    assert row["log_loss"] == pytest.approx(expected_ll, abs=1e-3)
    # margin err: |(3-9)-(3-9)|=0 and |(4-5)-(6-2)|=5 -> mean 2.5
    assert row["avg_margin_err"] == pytest.approx(2.5)
    # cumulative equals daily on the first ledger row
    assert row["cum_accuracy"] == pytest.approx(0.5)

    # Re-grading the same day must not double-count.
    graded2 = grade.grade_day("2026-08-06", games)
    row2 = grade.update_ledger("2026-08-06", graded2)
    assert int(row2["cum_games"]) == 2


def test_slate_predictions_carry_probable_starters(params):
    ratings = {"HOU": 1550.0, "STL": 1490.0}
    slate = [{
        "date": "2026-08-08", "away": "HOU", "home": "STL",
        "away_fr": "HOU", "home_fr": "STL", "game_num": 1,
        "away_sp": "Framber Valdez", "home_sp": "",
    }]
    df = simulate.slate_predictions(ratings, slate, params, n=500)
    assert df.iloc[0].away_sp == "Framber Valdez"
    assert df.iloc[0].home_sp == ""


def test_pitcher_name_normalization():
    from mlb.build_pitchers import normalize_name
    assert normalize_name("Félix Hernández") == "felix hernandez"
    assert normalize_name("Liván  Hernández") == "livan hernandez"
    assert normalize_name("J.A. Happ") == "j a happ"


def test_missing_slate_returns_none():
    assert grade.grade_day("1999-01-01", pd.DataFrame(
        columns=["date", "away_fr", "home_fr", "game_num",
                 "away_score", "home_score"])) is None
