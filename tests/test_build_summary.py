"""Tests for the cross-sport summary builder: game weighting, the
report-nothing-until-graded rule, and season status detection."""

from datetime import date
from pathlib import Path

from data_jobs.build_summary import (
    build_models, build_overall, mlb_record, next_season_start, odds_sport_games,
)

TODAY = date(2026, 8, 15)

# A ledger that does not exist: the state every sport is in before its first
# graded day. Passed explicitly so these tests never read the repo's real one.
UNGRADED = Path(__file__).with_name("no-such-grades.csv")

LEDGER_HEADER = (
    "date,games,correct,accuracy,log_loss,brier,avg_margin_err,avg_total_err,"
    "skipped,cum_games,cum_correct,cum_accuracy,cum_log_loss,cum_brier\n"
)


def _ledger(tmp_path, *rows):
    path = tmp_path / "grades.csv"
    path.write_text(LEDGER_HEADER + "".join(rows), encoding="utf-8")
    return path


def test_mlb_record_reads_cumulative_row(tmp_path):
    # Two days: 15 games at 6 correct, then 5 games at 5 correct. The ledger's
    # cumulative columns carry the game-weighted truth (11/20), which is what
    # should be reported - not the mean of the two daily accuracies (0.70).
    path = _ledger(
        tmp_path,
        "2026-08-07,15,6,0.4,0.7109,0.259,4.47,2.6,0,15,6,0.4,0.7109,0.259\n",
        "2026-08-08,5,5,1.0,0.5,0.2,1.0,1.0,0,20,11,0.55,0.65,0.244\n",
    )
    record = mlb_record(path)
    assert record["games"] == 20
    assert record["correct"] == 11
    assert record["accuracy"] == 0.55
    assert record["log_loss"] == 0.65
    assert record["last_graded"] == "2026-08-08"


def test_mlb_record_missing_ledger_reports_nothing(tmp_path):
    record = mlb_record(tmp_path / "absent.csv")
    assert record == {"games": 0, "correct": 0, "accuracy": None,
                      "log_loss": None, "brier": None, "last_graded": None}


def test_mlb_record_ignores_row_order(tmp_path):
    path = _ledger(
        tmp_path,
        "2026-08-08,5,5,1.0,0.5,0.2,1.0,1.0,0,20,11,0.55,0.65,0.244\n",
        "2026-08-07,15,6,0.4,0.7109,0.259,4.47,2.6,0,15,6,0.4,0.7109,0.259\n",
    )
    assert mlb_record(path)["last_graded"] == "2026-08-08"


def test_ungraded_model_reports_null_not_zero():
    # The site renders every null as an em dash. A zero here would publish a
    # 0% accuracy for a model that has simply never been graded.
    models = build_models(TODAY, mlb_latest=None, slate=None, grades_csv=UNGRADED)
    nfl = next(m for m in models if m["sport"] == "NFL")
    assert nfl["record"] is None
    assert nfl["accuracy"] is None
    assert nfl["log_loss"] is None
    assert nfl["brier"] is None
    assert nfl["last_graded"] is None
    assert nfl["games"] == 0


def test_roi_is_always_null():
    models = build_models(TODAY, mlb_latest=None, slate=None, grades_csv=UNGRADED)
    assert all(m["roi"] is None for m in models)
    assert build_overall(models)["roi"] is None


def test_overall_is_game_weighted_not_a_mean_of_models():
    models = [
        {"sport": "A", "games": 100, "record": "60-40", "log_loss": 0.60, "brier": 0.20},
        {"sport": "B", "games": 10, "record": "3-7", "log_loss": 0.90, "brier": 0.30},
    ]
    overall = build_overall(models)
    assert overall["record"] == "63-47"
    assert overall["games"] == 110
    assert overall["accuracy"] == round(63 / 110, 4)
    # Game-weighted: (0.60*100 + 0.90*10)/110, not the (0.60+0.90)/2 = 0.75
    # a per-model mean would give.
    assert overall["log_loss"] == round((0.60 * 100 + 0.90 * 10) / 110, 4)
    assert overall["sports_reporting"] == 2


def test_overall_with_nothing_graded_reports_nothing():
    overall = build_overall(build_models(TODAY, mlb_latest=None, slate=None, grades_csv=UNGRADED))
    assert overall["record"] is None
    assert overall["accuracy"] is None
    assert overall["sports_reporting"] == 0


def test_overall_skips_models_missing_a_metric():
    # A model can have graded games but no Brier; it should still count toward
    # the record while sitting out that one weighted average.
    models = [
        {"sport": "A", "games": 100, "record": "60-40", "log_loss": 0.60, "brier": 0.20},
        {"sport": "B", "games": 10, "record": "3-7", "log_loss": None, "brier": None},
    ]
    overall = build_overall(models)
    assert overall["games"] == 110
    assert overall["log_loss"] == 0.60


def test_in_season_from_a_populated_slate():
    slate = {"sports": [{"key": "nfl", "games": [{"game_id": "x"}, {"game_id": "y"}]}]}
    models = build_models(TODAY, mlb_latest=None, slate=slate)
    nfl = next(m for m in models if m["sport"] == "NFL")
    assert nfl["status"] == "in_season"
    assert nfl["slate_games"] == 2
    # An in-season sport has no "up next season" copy to render.
    assert "season_starts" not in nfl


def test_off_season_model_carries_a_season_start():
    models = build_models(TODAY, mlb_latest=None, slate=None, grades_csv=UNGRADED)
    nfl = next(m for m in models if m["sport"] == "NFL")
    nba = next(m for m in models if m["sport"] == "NBA")
    assert nfl["season_starts"] == "2026-09"
    assert nba["season_starts"] == "2026-10"


def test_season_start_rolls_to_next_year_once_the_month_has_passed():
    assert next_season_start(9, date(2026, 8, 15)) == "2026-09"
    assert next_season_start(9, date(2026, 10, 1)) == "2027-09"
    assert next_season_start(3, date(2026, 3, 20)) == "2026-03"


def test_odds_sport_games_tolerates_a_missing_sport():
    assert odds_sport_games({"sports": []}, "nfl") == 0
    assert odds_sport_games(None, "nfl") == 0
    assert odds_sport_games({"sports": [{"key": "nfl", "games": None}]}, "nfl") == 0
