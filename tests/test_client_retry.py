"""Tests for OddsAPIClient retry logic - no real network calls"""

from unittest import mock

import pytest
import requests

from data_jobs.odds_api.client import OddsAPIClient
from tests.fixtures import make_game


class FakeResponse:
    """Minimal stand-in for requests.Response"""

    def __init__(self, status_code=200, json_data=None, headers=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else []
        self.headers = headers or {"x-requests-remaining": "400", "x-requests-used": "100"}
        self.text = "fake body"

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Error", response=self
            )


@pytest.fixture
def client(tmp_path):
    return OddsAPIClient(api_key="test-key", usage_file=str(tmp_path / "usage.json"))


@pytest.fixture
def no_sleep(monkeypatch):
    """Don't actually sleep during backoff"""
    sleeps = []
    monkeypatch.setattr(
        "data_jobs.odds_api.client.time.sleep", lambda s: sleeps.append(s)
    )
    return sleeps


class TestRetryLogic:
    def test_retries_on_500_then_succeeds(self, client, no_sleep, monkeypatch):
        responses = [
            FakeResponse(status_code=500),
            FakeResponse(status_code=200, json_data=[make_game()]),
        ]
        get_mock = mock.Mock(side_effect=responses)
        monkeypatch.setattr("data_jobs.odds_api.client.requests.get", get_mock)

        games = client.fetch_odds("nfl")

        assert len(games) == 1
        assert get_mock.call_count == 2
        assert no_sleep == [2.0]  # one backoff of 2s

    def test_retries_on_429(self, client, no_sleep, monkeypatch):
        responses = [
            FakeResponse(status_code=429),
            FakeResponse(status_code=200, json_data=[]),
        ]
        get_mock = mock.Mock(side_effect=responses)
        monkeypatch.setattr("data_jobs.odds_api.client.requests.get", get_mock)

        games = client.fetch_odds("nba")
        assert games == []
        assert get_mock.call_count == 2

    def test_does_not_retry_on_401(self, client, no_sleep, monkeypatch):
        get_mock = mock.Mock(return_value=FakeResponse(status_code=401))
        monkeypatch.setattr("data_jobs.odds_api.client.requests.get", get_mock)

        with pytest.raises(requests.HTTPError):
            client.fetch_odds("nfl")

        assert get_mock.call_count == 1
        assert no_sleep == []

    def test_does_not_retry_on_404(self, client, no_sleep, monkeypatch):
        get_mock = mock.Mock(return_value=FakeResponse(status_code=404))
        monkeypatch.setattr("data_jobs.odds_api.client.requests.get", get_mock)

        with pytest.raises(requests.HTTPError):
            client.fetch_odds("nfl")

        assert get_mock.call_count == 1

    def test_retries_on_connection_error_then_succeeds(self, client, no_sleep, monkeypatch):
        get_mock = mock.Mock(
            side_effect=[
                requests.ConnectionError("connection reset"),
                requests.Timeout("timed out"),
                FakeResponse(status_code=200, json_data=[]),
            ]
        )
        monkeypatch.setattr("data_jobs.odds_api.client.requests.get", get_mock)

        games = client.fetch_odds("nfl")
        assert games == []
        assert get_mock.call_count == 3
        assert no_sleep == [2.0, 4.0]  # exponential backoff

    def test_gives_up_after_max_attempts(self, client, no_sleep, monkeypatch):
        get_mock = mock.Mock(return_value=FakeResponse(status_code=503))
        monkeypatch.setattr("data_jobs.odds_api.client.requests.get", get_mock)

        with pytest.raises(requests.HTTPError):
            client.fetch_odds("nfl")

        assert get_mock.call_count == 3

    def test_unsupported_sport_raises_without_request(self, client, monkeypatch):
        get_mock = mock.Mock()
        monkeypatch.setattr("data_jobs.odds_api.client.requests.get", get_mock)

        with pytest.raises(ValueError):
            client.fetch_odds("cricket")

        assert get_mock.call_count == 0

    def test_missing_api_key_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ODDS_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            OddsAPIClient(api_key=None, usage_file=str(tmp_path / "usage.json"))
