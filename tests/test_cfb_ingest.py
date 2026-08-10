"""The Winner/Loser -> home/away transform is the one CFB parse that, done
backwards, silently inverts home-field advantage (DATA_PULL_PLAN.md §2.1).
These tests pin it with hand-built rows in CFR's exact export shape."""

import pandas as pd
import pytest

from CFB.ingest import parse_schedule, _split_rank, _season_of


def _schedule_csv(tmp_path, rows):
    header = "Rk,Wk,Date,Day,Winner,Pts,,Loser,Pts,Notes\n"
    p = tmp_path / "1999games.csv"
    p.write_text(header + "\n".join(rows) + "\n")
    return p


def test_blank_location_means_winner_was_home(tmp_path):
    p = _schedule_csv(tmp_path, ["1,1,Sep 4 1999,Sat,Home U,31,,Road U,10,"])
    g = parse_schedule(p)
    assert g.loc[0, "home_team"] == "Home U"
    assert g.loc[0, "home_score"] == 31
    assert g.loc[0, "away_score"] == 10
    assert g.loc[0, "location"] == "Home"


def test_at_sign_means_winner_was_road_team(tmp_path):
    p = _schedule_csv(tmp_path, ["1,1,Sep 4 1999,Sat,Road U,31,@,Home U,10,"])
    g = parse_schedule(p)
    # The winner travelled: the LOSER is the home team, and the home team's
    # score is the loser's score. Getting this wrong flips HFA.
    assert g.loc[0, "home_team"] == "Home U"
    assert g.loc[0, "away_team"] == "Road U"
    assert g.loc[0, "home_score"] == 10
    assert g.loc[0, "away_score"] == 31


def test_neutral_site_flagged_winner_listed_as_home(tmp_path):
    p = _schedule_csv(
        tmp_path, ["1,16,Jan 3 2000,Mon,Champ U,13,N,Runner Up,2,Orange Bowl (Miami FL)"]
    )
    g = parse_schedule(p)
    assert g.loc[0, "location"] == "Neutral"
    assert g.loc[0, "game_type"] == "BOWL"
    # January bowl belongs to the prior calendar year's season.
    assert g.loc[0, "season"] == 1999


def test_rank_prefix_stripped_and_kept():
    rank, name = _split_rank("(2) Florida State")
    assert rank == 2 and name == "Florida State"
    rank, name = _split_rank("Louisiana-Monroe")
    assert pd.isna(rank) and name == "Louisiana-Monroe"


def test_championship_note_is_ccg_not_bowl(tmp_path):
    p = _schedule_csv(
        tmp_path, ["1,14,Dec 2 2000,Sat,Big U,27,N,Other U,24,SEC Championship (Atlanta GA)"]
    )
    g = parse_schedule(p)
    assert g.loc[0, "game_type"] == "CCG"


def test_repeated_header_rows_dropped(tmp_path):
    p = _schedule_csv(
        tmp_path,
        [
            "1,1,Sep 4 1999,Sat,Home U,31,,Road U,10,",
            "Rk,Wk,Date,Day,Winner,Pts,,Loser,Pts,Notes",
            "2,1,Sep 4 1999,Sat,Other U,21,,Visitor U,14,",
        ],
    )
    g = parse_schedule(p)
    assert len(g) == 2


@pytest.mark.parametrize(
    "date,season",
    [("2000-08-26", 2000), ("2000-11-25", 2000), ("2001-01-03", 2000), ("2001-06-01", 2001)],
)
def test_season_rollover(date, season):
    assert _season_of(pd.Timestamp(date)) == season
