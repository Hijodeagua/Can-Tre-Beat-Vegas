"""Tests for data_jobs.odds_api.check_staleness"""

import json
import os
from datetime import datetime, timedelta

import pandas as pd
import pytest
import pytz

from data_jobs.odds_api import check_staleness
from data_jobs.odds_api.check_staleness import (
    check_sport_staleness,
    load_fetch_status,
    newest_timestamp_pulled,
)

EASTERN = pytz.timezone("US/Eastern")


def write_latest_csv(base_dir, sport, timestamps):
    """Write a minimal _latest.csv for a sport with given 'Timestamp Pulled' values"""
    from data_jobs.odds_api.config import SUPPORTED_SPORTS

    cfg = SUPPORTED_SPORTS[sport]
    save_dir = os.path.join(base_dir, cfg["data_dir"])
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{cfg['file_prefix']}_latest.csv")

    df = pd.DataFrame(
        {
            "League": [cfg["name"]] * len(timestamps),
            "Timestamp Pulled": timestamps,
            "Home Team": ["Team A"] * len(timestamps),
        }
    )
    df.to_csv(path, index=False)
    return path


def eastern_stamp(days_ago=0, hours_ago=0):
    dt = datetime.now(pytz.UTC).astimezone(EASTERN) - timedelta(
        days=days_ago, hours=hours_ago
    )
    return dt.strftime("%Y-%m-%d %H:%M:%S")


class TestCheckSportStaleness:
    def test_fresh_file_passes(self, tmp_path):
        write_latest_csv(str(tmp_path), "nfl", [eastern_stamp(hours_ago=2)])
        status = {"nfl": {"success": True, "games_count": 12, "error": None}}

        ok, message = check_sport_staleness("nfl", status, str(tmp_path), 3)
        assert ok is True

    def test_old_file_fails_when_games_upcoming(self, tmp_path):
        write_latest_csv(str(tmp_path), "nfl", [eastern_stamp(days_ago=10)])
        status = {"nfl": {"success": True, "games_count": 12, "error": None}}

        ok, message = check_sport_staleness("nfl", status, str(tmp_path), 3)
        assert ok is False
        assert "STALE" in message

    def test_old_file_passes_in_offseason(self, tmp_path):
        """NBA in July: file frozen for weeks but API reports 0 games -> OK"""
        write_latest_csv(str(tmp_path), "nba", [eastern_stamp(days_ago=30)])
        status = {"nba": {"success": True, "games_count": 0, "error": None}}

        ok, message = check_sport_staleness("nba", status, str(tmp_path), 3)
        assert ok is True
        assert "offseason" in message

    def test_missing_file_fails_when_games_upcoming(self, tmp_path):
        status = {"nfl": {"success": True, "games_count": 5, "error": None}}
        ok, message = check_sport_staleness("nfl", status, str(tmp_path), 3)
        assert ok is False

    def test_no_status_for_sport_skips(self, tmp_path):
        write_latest_csv(str(tmp_path), "nfl", [eastern_stamp(days_ago=30)])
        ok, message = check_sport_staleness("nfl", {}, str(tmp_path), 3)
        assert ok is True

    def test_uses_newest_timestamp_in_file(self, tmp_path):
        """A file with one old and one fresh row is not stale"""
        write_latest_csv(
            str(tmp_path),
            "nfl",
            [eastern_stamp(days_ago=10), eastern_stamp(hours_ago=1)],
        )
        status = {"nfl": {"success": True, "games_count": 3, "error": None}}
        ok, _ = check_sport_staleness("nfl", status, str(tmp_path), 3)
        assert ok is True

    def test_boundary_just_under_threshold_passes(self, tmp_path):
        write_latest_csv(str(tmp_path), "nfl", [eastern_stamp(days_ago=2, hours_ago=20)])
        status = {"nfl": {"success": True, "games_count": 3, "error": None}}
        ok, _ = check_sport_staleness("nfl", status, str(tmp_path), 3)
        assert ok is True


class TestHelpers:
    def test_newest_timestamp_pulled_missing_file(self, tmp_path):
        assert newest_timestamp_pulled(str(tmp_path / "nope.csv")) is None

    def test_newest_timestamp_pulled_no_column(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"Other": [1]}).to_csv(path, index=False)
        assert newest_timestamp_pulled(str(path)) is None

    def test_load_fetch_status_missing(self, tmp_path):
        assert load_fetch_status(str(tmp_path / "nope.json")) == {}

    def test_load_fetch_status_valid(self, tmp_path):
        path = tmp_path / "status.json"
        path.write_text(json.dumps({"sports": {"nfl": {"games_count": 2}}}))
        assert load_fetch_status(str(path))["nfl"]["games_count"] == 2


class TestMain:
    def _write_status(self, base_dir, sports):
        status_path = os.path.join(base_dir, "data", "fetch_status.json")
        os.makedirs(os.path.dirname(status_path), exist_ok=True)
        with open(status_path, "w") as f:
            json.dump({"sports": sports}, f)

    def test_main_exits_nonzero_on_stale(self, monkeypatch, tmp_path):
        write_latest_csv(str(tmp_path), "nfl", [eastern_stamp(days_ago=10)])
        self._write_status(
            str(tmp_path), {"nfl": {"success": True, "games_count": 5, "error": None}}
        )
        monkeypatch.setattr(
            "sys.argv",
            ["check_staleness", "--sport", "nfl", "--base-dir", str(tmp_path)],
        )
        with pytest.raises(SystemExit) as excinfo:
            check_staleness.main()
        assert excinfo.value.code == 1

    def test_main_passes_on_fresh(self, monkeypatch, tmp_path):
        write_latest_csv(str(tmp_path), "nfl", [eastern_stamp(hours_ago=1)])
        self._write_status(
            str(tmp_path), {"nfl": {"success": True, "games_count": 5, "error": None}}
        )
        monkeypatch.setattr(
            "sys.argv",
            ["check_staleness", "--sport", "nfl", "--base-dir", str(tmp_path)],
        )
        assert check_staleness.main() is None

    def test_main_passes_offseason(self, monkeypatch, tmp_path):
        write_latest_csv(str(tmp_path), "nba", [eastern_stamp(days_ago=30)])
        self._write_status(
            str(tmp_path), {"nba": {"success": True, "games_count": 0, "error": None}}
        )
        monkeypatch.setattr(
            "sys.argv",
            ["check_staleness", "--sport", "nba", "--base-dir", str(tmp_path)],
        )
        assert check_staleness.main() is None

    def test_main_no_status_file_is_noop(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "sys.argv", ["check_staleness", "--base-dir", str(tmp_path)]
        )
        assert check_staleness.main() is None
