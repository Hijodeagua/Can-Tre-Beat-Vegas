"""Tests for data_jobs.odds_api.fetch_odds - no real network calls"""

import json
import os
from unittest import mock

import pytest

from data_jobs.odds_api import fetch_odds
from tests.fixtures import make_game


def make_mock_client(fetch_side_effect=None, fetch_return=None):
    """Build a mock OddsAPIClient"""
    client = mock.Mock()
    if fetch_side_effect is not None:
        client.fetch_odds.side_effect = fetch_side_effect
    else:
        client.fetch_odds.return_value = fetch_return if fetch_return is not None else []
    client.get_quota_status.return_value = {
        "requests_remaining": 400,
        "monthly_limit": 500,
        "recent_calls": 3,
    }
    client.check_quota_warning.return_value = None
    return client


class TestFetchSingleSport:
    def test_success_with_games(self, tmp_path):
        client = make_mock_client(fetch_return=[make_game()])
        result = fetch_odds.fetch_single_sport(client, "nfl", str(tmp_path))

        assert result["success"] is True
        assert result["games_count"] == 1
        assert result["error"] is None
        assert len(result["files"]) == 2

    def test_success_with_empty_response_is_not_failure(self, tmp_path):
        """Offseason: API succeeds but returns zero games"""
        client = make_mock_client(fetch_return=[])
        result = fetch_odds.fetch_single_sport(client, "nba", str(tmp_path))

        assert result["success"] is True
        assert result["games_count"] == 0
        assert result["error"] is None

    def test_client_exception_returns_failure(self, tmp_path):
        client = make_mock_client(fetch_side_effect=RuntimeError("API is down"))
        result = fetch_odds.fetch_single_sport(client, "nfl", str(tmp_path))

        assert result["success"] is False
        assert "API is down" in result["error"]
        assert result["files"] == []


class TestMainExitBehavior:
    def _run_main(self, monkeypatch, tmp_path, client, sport="nba"):
        monkeypatch.setattr(
            fetch_odds, "OddsAPIClient", mock.Mock(return_value=client)
        )
        monkeypatch.setattr(
            "sys.argv",
            ["fetch_odds", "--sport", sport, "--base-dir", str(tmp_path)],
        )
        return fetch_odds.main()

    def test_exits_nonzero_on_fetch_failure(self, monkeypatch, tmp_path):
        client = make_mock_client(fetch_side_effect=RuntimeError("boom"))
        with pytest.raises(SystemExit) as excinfo:
            self._run_main(monkeypatch, tmp_path, client)
        assert excinfo.value.code == 1

    def test_exits_zero_on_success_with_empty_response(self, monkeypatch, tmp_path):
        """Offseason (zero games, successful call) must NOT fail the run"""
        client = make_mock_client(fetch_return=[])
        # main() returning normally means exit code 0
        assert self._run_main(monkeypatch, tmp_path, client) is None

    def test_exits_zero_on_success_with_games(self, monkeypatch, tmp_path):
        client = make_mock_client(fetch_return=[make_game()])
        assert self._run_main(monkeypatch, tmp_path, client, sport="nfl") is None

    def test_exits_nonzero_when_any_sport_fails(self, monkeypatch, tmp_path):
        """--sport all: one sport fails, one succeeds -> still non-zero"""

        def side_effect(sport, *args, **kwargs):
            if sport == "nfl":
                return [make_game()]
            raise RuntimeError("nba endpoint 500")

        client = make_mock_client(fetch_side_effect=side_effect)
        with pytest.raises(SystemExit) as excinfo:
            self._run_main(monkeypatch, tmp_path, client, sport="all")
        assert excinfo.value.code == 1

    def test_writes_fetch_status_file(self, monkeypatch, tmp_path):
        client = make_mock_client(fetch_return=[])
        self._run_main(monkeypatch, tmp_path, client, sport="nba")

        status_path = os.path.join(str(tmp_path), "data", "fetch_status.json")
        assert os.path.exists(status_path)
        with open(status_path) as f:
            status = json.load(f)
        assert status["sports"]["nba"]["success"] is True
        assert status["sports"]["nba"]["games_count"] == 0

    def test_status_file_records_failure(self, monkeypatch, tmp_path):
        client = make_mock_client(fetch_side_effect=RuntimeError("boom"))
        with pytest.raises(SystemExit):
            self._run_main(monkeypatch, tmp_path, client, sport="nba")

        status_path = os.path.join(str(tmp_path), "data", "fetch_status.json")
        with open(status_path) as f:
            status = json.load(f)
        assert status["sports"]["nba"]["success"] is False
        assert "boom" in status["sports"]["nba"]["error"]
