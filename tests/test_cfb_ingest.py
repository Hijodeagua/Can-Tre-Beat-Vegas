"""The Winner/Loser -> home/away transform is the one CFB parse that, done
backwards, silently inverts home-field advantage (DATA_PULL_PLAN.md §2.1).
These tests pin it with hand-built rows in CFR's exact export shape."""

import pandas as pd
import pytest

from CFB.ingest import dedupe_games, parse_schedule, _split_rank, _season_of


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


# --- game_type: the 2016-2019 exports put a venue string on EVERY game ---


def test_venue_note_is_not_postseason(tmp_path):
    p = _schedule_csv(
        tmp_path, ["1,10,Nov 4 2017,Sat,Home U,31,,Road U,10,Lane Stadium - Blacksburg Virginia"]
    )
    assert parse_schedule(p).loc[0, "game_type"] == "REG"


def test_december_venue_note_is_still_reg(tmp_path):
    # Army-Navy et al.: December regular-season game with a venue-only note.
    p = _schedule_csv(
        tmp_path, ["1,15,Dec 9 2017,Sat,Home U,14,N,Road U,13,Lincoln Financial Field - Philadelphia Pennsylvania"]
    )
    assert parse_schedule(p).loc[0, "game_type"] == "REG"


def test_red_river_at_cotton_bowl_stadium_is_reg(tmp_path):
    # The October rivalry game is *at* the Cotton Bowl stadium — not a bowl.
    p = _schedule_csv(
        tmp_path, ["1,6,Oct 14 2017,Sat,Winner U,29,N,Loser U,24,Cotton Bowl - Dallas Texas"]
    )
    assert parse_schedule(p).loc[0, "game_type"] == "REG"


def test_bowl_name_with_venue_parens_is_bowl(tmp_path):
    p = _schedule_csv(
        tmp_path,
        ['1,16,Dec 30 2017,Sat,Winner U,21,N,Loser U,20,"Liberty Bowl (Liberty Bowl Memorial Stadium - Memphis Tennessee)"'],
    )
    assert parse_schedule(p).loc[0, "game_type"] == "BOWL"


def test_august_kickoff_classic_is_reg(tmp_path):
    p = _schedule_csv(
        tmp_path, ["1,1,Aug 27 2000,Sun,Winner U,29,N,Loser U,5,Kickoff Classic (East Rutherford NJ)"]
    )
    assert parse_schedule(p).loc[0, "game_type"] == "REG"


def test_double_listed_bowl_deduped_keeping_bowl_row(tmp_path):
    # CFR lists some bowls twice: once as a venue string, once by name.
    p = _schedule_csv(
        tmp_path,
        [
            "1,16,Dec 29 2006,Fri,Winner U,44,N,Loser U,36,Liberty Bowl Memorial Stadium - Memphis Tennessee",
            "2,16,Dec 29 2006,Fri,Winner U,44,N,Loser U,36,Liberty Bowl (Memphis TN)",
        ],
    )
    g = dedupe_games(parse_schedule(p))
    assert len(g) == 1
    assert g.iloc[0]["game_type"] == "BOWL"


# --- the dedupe tie-break must be decided, not left to the sort ---


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_double_listed_regular_season_keeps_neutral_site(tmp_path, order):
    # A neutral-site REG game listed twice: the venue row carries the 'N'
    # marker, the bare row does not. Both are REG, so the postseason key
    # cannot separate them. Keeping the wrong twin invents home-field
    # advantage — and the answer must not depend on the input row order.
    rows = [
        "1,1,Aug 31 2008,Sun,Colorado,31,N,Colorado State,28,Empower Field at Mile High - Denver Colorado",
        "2,1,Aug 31 2008,Sun,Colorado,31,,Colorado State,28,",
    ]
    p = _schedule_csv(tmp_path, [rows[order[0]], rows[order[1]]])
    g = dedupe_games(parse_schedule(p))
    assert len(g) == 1
    assert g.iloc[0]["location"] == "Neutral"


def test_dedupe_is_order_independent(tmp_path):
    # Same games, reversed input: identical output, not a coin flip.
    rows = [
        "1,16,Dec 29 2006,Fri,Winner U,44,N,Loser U,36,Liberty Bowl Memorial Stadium - Memphis Tennessee",
        "2,16,Dec 29 2006,Fri,Winner U,44,N,Loser U,36,Liberty Bowl (Memphis TN)",
        "3,1,Aug 31 2008,Sun,Colorado,31,N,Colorado State,28,Empower Field at Mile High - Denver Colorado",
        "4,1,Aug 31 2008,Sun,Colorado,31,,Colorado State,28,",
    ]
    cols = ["season", "gameday", "home_team", "away_team", "game_type", "location"]
    forward = dedupe_games(parse_schedule(_schedule_csv(tmp_path, rows)))
    backward = dedupe_games(parse_schedule(_schedule_csv(tmp_path, rows[::-1])))
    pd.testing.assert_frame_equal(
        forward[cols].sort_values(cols).reset_index(drop=True),
        backward[cols].sort_values(cols).reset_index(drop=True),
    )
