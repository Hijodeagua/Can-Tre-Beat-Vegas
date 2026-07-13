"""Tests for data_jobs.odds_api.processors"""

import os

import pandas as pd
import pytest

from data_jobs.odds_api.processors import (
    process_game_data,
    save_odds_data,
    _extract_odds,
    _safe_avg,
)
from tests.fixtures import make_game, make_bookmaker


class TestExtractOdds:
    def test_well_formed_game(self):
        game = make_game()
        result = _extract_odds(game, "Kansas City Chiefs", "Buffalo Bills")

        averages = result["averages"]
        assert averages["Avg Home H2H Odds"] == pytest.approx(-152.5)
        assert averages["Avg Away H2H Odds"] == pytest.approx(132.5)
        assert averages["Avg Home Spread Odds"] == pytest.approx(-109.0)
        assert averages["Avg Away Spread Odds"] == pytest.approx(-111.0)
        assert averages["Avg Home Spread Points"] == pytest.approx(-3.5)
        assert averages["Avg Away Spread Points"] == pytest.approx(3.5)
        assert averages["Avg Over Odds"] == pytest.approx(-111.0)
        assert averages["Avg Under Odds"] == pytest.approx(-109.0)
        assert averages["Avg Total Points"] == pytest.approx(48.0)

        per_book = result["per_book"]
        assert per_book["Home DraftKings H2H Odds"] == -150
        assert per_book["Away DraftKings H2H Odds"] == 130
        assert per_book["Home FanDuel Spread Odds"] == -108
        assert per_book["Home DraftKings O/U Odds"] == -110

    def test_missing_markets(self):
        """A bookmaker with no markets yields None per-book values and None averages"""
        game = make_game(
            bookmakers=[{"key": "emptybook", "title": "EmptyBook", "markets": []}]
        )
        result = _extract_odds(game, "Kansas City Chiefs", "Buffalo Bills")

        assert result["averages"]["Avg Home H2H Odds"] is None
        assert result["averages"]["Avg Total Points"] is None
        assert result["per_book"]["Home EmptyBook H2H Odds"] is None
        assert result["per_book"]["Home EmptyBook Spread Odds"] is None

    def test_missing_outcomes(self):
        """Markets with empty/None outcomes are handled without error"""
        game = make_game(
            bookmakers=[
                {
                    "key": "weird",
                    "title": "WeirdBook",
                    "markets": [
                        {"key": "h2h", "outcomes": []},
                        {"key": "spreads", "outcomes": None},
                        {"key": "totals", "outcomes": []},
                    ],
                }
            ]
        )
        result = _extract_odds(game, "Kansas City Chiefs", "Buffalo Bills")
        assert result["averages"]["Avg Home H2H Odds"] is None
        assert result["per_book"]["Home WeirdBook H2H Odds"] is None

    def test_no_bookmakers(self):
        game = make_game(bookmakers=[])
        result = _extract_odds(game, "Kansas City Chiefs", "Buffalo Bills")
        assert result["per_book"] == {}
        assert all(v is None for v in result["averages"].values())

    def test_outcome_names_not_matching_teams_ignored(self):
        game = make_game(
            bookmakers=[
                make_bookmaker("SomeBook", "Not A Team", "Also Not A Team")
            ]
        )
        result = _extract_odds(game, "Kansas City Chiefs", "Buffalo Bills")
        assert result["averages"]["Avg Home H2H Odds"] is None
        # Totals still parse - Over/Under names don't depend on teams
        assert result["averages"]["Avg Total Points"] == pytest.approx(47.5)


class TestSafeAvg:
    def test_filters_none(self):
        assert _safe_avg([10, None, 20]) == 15.0

    def test_all_none(self):
        assert _safe_avg([None, None]) is None

    def test_empty(self):
        assert _safe_avg([]) is None


class TestProcessGameData:
    def test_well_formed_payload(self):
        games = [
            make_game(),
            make_game(
                home_team="Philadelphia Eagles",
                away_team="Dallas Cowboys",
                game_id="def789",
            ),
        ]
        df = process_game_data(games, "nfl")

        assert len(df) == 2
        assert df.iloc[0]["Home Team"] == "Kansas City Chiefs"
        assert df.iloc[0]["Away Team"] == "Buffalo Bills"
        assert df.iloc[0]["League"] == "NFL"
        assert df.iloc[0]["Divisional Matchup"] == "No"
        # Eagles vs Cowboys are both NFC East
        assert df.iloc[1]["Divisional Matchup"] == "Yes"
        assert "Timestamp Pulled" in df.columns
        assert "Avg Home H2H Odds" in df.columns
        assert df.iloc[0]["Travel Distance (mi)"] > 0

    def test_unknown_teams_filtered(self):
        games = [
            make_game(home_team="Unknown United", away_team="Buffalo Bills"),
            make_game(),
        ]
        df = process_game_data(games, "nfl")
        assert len(df) == 1
        assert df.iloc[0]["Home Team"] == "Kansas City Chiefs"

    def test_all_unknown_teams_yields_empty_df(self):
        games = [make_game(home_team="Unknown A", away_team="Unknown B")]
        df = process_game_data(games, "nfl")
        assert df.empty

    def test_empty_games_list(self):
        df = process_game_data([], "nfl")
        assert df.empty

    def test_nba_payload(self):
        games = [
            make_game(
                home_team="Boston Celtics",
                away_team="Los Angeles Lakers",
                commence_time="2026-12-25T22:00:00Z",
            )
        ]
        df = process_game_data(games, "nba")
        assert len(df) == 1
        assert df.iloc[0]["League"] == "NBA"


class TestSaveOddsData:
    def test_writes_latest_and_timestamped(self, tmp_path):
        games = [make_game()]
        df = process_game_data(games, "nfl")

        timestamped, latest = save_odds_data(df, "nfl", str(tmp_path))

        assert os.path.exists(timestamped)
        assert os.path.exists(latest)
        assert latest.endswith("odds_api_data_latest.csv")

        reloaded = pd.read_csv(latest)
        assert len(reloaded) == 1
        assert reloaded.iloc[0]["Home Team"] == "Kansas City Chiefs"

    def test_empty_df_does_not_overwrite_existing_latest(self, tmp_path):
        # Seed a non-empty latest file
        df = process_game_data([make_game()], "nfl")
        timestamped1, latest = save_odds_data(df, "nfl", str(tmp_path))
        assert len(pd.read_csv(latest)) == 1

        # Attempt to save an empty DataFrame over it
        empty_df = process_game_data([], "nfl")
        timestamped2, latest2 = save_odds_data(empty_df, "nfl", str(tmp_path))

        assert latest2 == latest
        # Existing data preserved
        preserved = pd.read_csv(latest)
        assert len(preserved) == 1
        # No empty timestamped file written either (a same-minute path
        # collision with the first write still holds the original data)
        if timestamped2 != timestamped1:
            assert not os.path.exists(timestamped2)
        else:
            assert len(pd.read_csv(timestamped2)) == 1

    def test_empty_df_writes_when_no_existing_latest(self, tmp_path):
        """With no prior data, an empty write is allowed (nothing to protect)"""
        empty_df = pd.DataFrame()
        timestamped, latest = save_odds_data(empty_df, "nfl", str(tmp_path))
        assert os.path.exists(latest)

    def test_non_empty_df_overwrites_latest(self, tmp_path):
        df1 = process_game_data([make_game()], "nfl")
        save_odds_data(df1, "nfl", str(tmp_path))

        df2 = process_game_data(
            [
                make_game(),
                make_game(
                    home_team="Chicago Bears",
                    away_team="Green Bay Packers",
                    game_id="xyz",
                ),
            ],
            "nfl",
        )
        _, latest = save_odds_data(df2, "nfl", str(tmp_path))
        assert len(pd.read_csv(latest)) == 2
