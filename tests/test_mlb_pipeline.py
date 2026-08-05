"""Tests for the MLB data pipeline (MLB/data_jobs)."""

import os

import pandas as pd
import pytest

from MLB.data_jobs import gamelogs
from MLB.data_jobs.config import (
    GAMES_PATH,
    RETRO_TO_FRANCHISE,
    TEAMS,
)
from MLB.data_jobs.gamelog_fields import GAMELOG_COLUMNS
from data_jobs.odds_api.config import MLB_TEAMS, get_team_info


# ---------------------------------------------------------------------------
# Schema / parsing units
# ---------------------------------------------------------------------------

def test_gamelog_schema_has_161_fields():
    assert len(GAMELOG_COLUMNS) == 161
    assert len(set(GAMELOG_COLUMNS)) == 161  # no duplicate names


@pytest.mark.parametrize(
    "linescore,expected",
    [
        ("000000000", 9),
        ("010000(10)0x", 9),          # big inning + home ninth not played
        ("0000000", 7),               # 2020 seven-inning doubleheader game
        ("00000000000012", 14),       # extras
        ("21010000", 8),              # home team of a rain-shortened game
    ],
)
def test_innings_from_linescore(linescore, expected):
    assert gamelogs.innings_from_linescore(linescore) == expected


# ---------------------------------------------------------------------------
# Crosswalk consistency
# ---------------------------------------------------------------------------

def test_crosswalk_covers_30_franchises():
    assert len(TEAMS) == 30
    mlb_ids = [info["mlb_id"] for info in TEAMS.values()]
    assert len(set(mlb_ids)) == 30


def test_every_odds_api_name_maps_to_a_franchise():
    """data_jobs/odds_api MLB_TEAMS and the MLB crosswalk must agree."""
    crosswalk_names = {
        name for info in TEAMS.values() for name in info["odds_names"]
    }
    assert crosswalk_names == set(MLB_TEAMS)


def test_odds_api_supports_mlb():
    info = get_team_info("mlb")
    assert "New York Yankees" in info
    # Both A's identities resolve, at their respective parks.
    assert info["Oakland Athletics"]["loc"] != info["Athletics"]["loc"]


# ---------------------------------------------------------------------------
# Games table invariants (run only when the cache exists)
# ---------------------------------------------------------------------------

needs_cache = pytest.mark.skipif(
    not os.path.exists(GAMES_PATH), reason="games cache not built"
)


@pytest.fixture(scope="module")
def games():
    return pd.read_csv(GAMES_PATH, low_memory=False)


@needs_cache
def test_no_null_join_keys(games):
    for col in ("game_id", "date", "home_team", "away_team",
                "home_score", "away_score", "park_id"):
        assert games[col].notna().all(), f"nulls in {col}"


@needs_cache
def test_game_ids_unique(games):
    assert not games["game_id"].duplicated().any()


@needs_cache
def test_every_retro_code_mapped(games):
    codes = set(games["home_team_retro"]) | set(games["away_team_retro"])
    unmapped = codes - set(RETRO_TO_FRANCHISE)
    assert not unmapped, f"retro codes missing from crosswalk: {sorted(unmapped)}"


@needs_cache
def test_starting_pitchers_present(games):
    assert games["home_sp_id"].notna().all()
    assert games["away_sp_id"].notna().all()


@needs_cache
def test_season_range_and_counts(games):
    reg = games[games["game_type"] == "regular"]
    by_season = reg.groupby("season").size()
    assert by_season.index.min() == 2005
    full_seasons = by_season.drop(index=2020, errors="ignore")
    assert (full_seasons.between(2424, 2436)).all(), full_seasons.to_dict()
    if 2020 in by_season.index:
        assert 890 <= by_season.loc[2020] <= 900


@needs_cache
def test_home_advantage_sane(games):
    """Baseball home-win rate is ~54%; far outside that means a parsing bug
    (e.g. home/away columns swapped)."""
    rate = games.loc[games["game_type"] == "regular", "home_win"].mean()
    assert 0.52 < rate < 0.56, rate


# ---------------------------------------------------------------------------
# Pitching lines (retrosplits)
# ---------------------------------------------------------------------------

from MLB.data_jobs import pitching  # noqa: E402

needs_pitching = pytest.mark.skipif(
    not pitching.cached_seasons(), reason="pitching cache not built"
)


@needs_pitching
@needs_cache
def test_pitching_join_coverage(games):
    problems, _notes = pitching.assert_join_coverage(
        games, pitching.cached_seasons()
    )
    assert not problems, problems


@needs_pitching
def test_pitching_start_counts_sane():
    """A full season has 2 starts per game: ~4,860 (2020: ~1,800)."""
    for year in (2005, 2019, 2024):
        if year not in pitching.cached_seasons():
            continue
        p = pitching.load_season(year)
        starts = int((p["P_GS"] == 1).sum())
        assert 4700 <= starts <= 5100, (year, starts)


@needs_pitching
def test_pitching_outs_plausible():
    p = pitching.load_season(2024)
    # No pitcher records more than 27 outs; team totals per game do.
    assert p["P_OUT"].max() <= 27
    starters = p[p["P_GS"] == 1]
    # Mean start length in the modern era is ~5.1-5.5 innings.
    mean_ip = (starters["P_OUT"].mean()) / 3
    assert 4.5 < mean_ip < 6.5, mean_ip


@needs_cache
def test_innings_parse(games):
    reg = games[games["game_type"] == "regular"]
    # Innings must be present and plausible everywhere.
    assert reg["innings"].between(5, 26).all()
    # Extra-inning share of full seasons runs ~7-10%.
    share = (reg["innings"] > 9).mean()
    assert 0.05 < share < 0.12, share
