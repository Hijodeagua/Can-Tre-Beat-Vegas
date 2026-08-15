"""Unit tests for the starting-pitcher rating layer (mlb/pitcher_rating.py)."""

import pytest

from mlb.pitcher_rating import (
    LEAGUE_SEED, LeakageError, PitcherBook, game_score,
)


# Five hand-verified starts. The first three are famous games whose game
# scores are public record; all five lines were checked against
# data/mlb/pitcher_starts.csv (outs, h, r, er, bb, so order).
HAND_VERIFIED = [
    # Kershaw no-hitter, LAD 2014-06-18: 9 IP, 0 H, 0 R, 0 BB, 15 K
    ((27, 0, 0, 0, 0, 15), 102),
    # Scherzer 20-K game, WSN 2016-05-11: 9 IP, 6 H, 2 ER, 0 BB, 20 K
    ((27, 6, 2, 2, 0, 20), 87),
    # Cain perfect game, SFG 2012-06-13: 9 IP, 0 H, 0 R, 0 BB, 14 K
    ((27, 0, 0, 0, 0, 14), 101),
    # Marco Gonzales, SEA 2019-03-20: 6 IP, 7 H, 4 R (3 ER), 1 BB, 4 K
    # 50 + 18 + 2*2 + 4 - 14 - 12 - 2*1 - 1 = 47
    ((18, 7, 4, 3, 1, 4), 47),
    # Brad Keller, KCR 2021-04-01: 1.1 IP, 9 H, 6 R (all earned), 2 BB, 0 K
    # 50 + 4 + 0 + 0 - 18 - 24 - 0 - 2 = 10
    ((4, 9, 6, 6, 2, 0), 10),
]


@pytest.mark.parametrize("line,expected", HAND_VERIFIED)
def test_game_score_hand_verified(line, expected):
    assert game_score(*line) == expected


def test_game_score_no_bonus_before_fifth_inning():
    # 4 complete innings: no completed-innings bonus yet; 5th adds 2.
    assert game_score(12, 0, 0, 0, 0, 0) == 62
    assert game_score(15, 0, 0, 0, 0, 0) == 67


def test_ew_mean_is_debiased_average_of_first_starts():
    book = PitcherBook(half_life=10)
    book.record_start("2024-04-01", "LAD", "p1", 60)
    book.record_start("2024-04-06", "LAD", "p1", 40)
    book.advance_to("2024-04-11")
    ew = book.pitchers["p1"]
    d = 0.5 ** (1 / 10)
    expected = (40 + d * 60) / (1 + d)
    assert ew.value(0) == pytest.approx(expected)
    # Debiased: with only 2 starts the value sits between them, closer to
    # the newer one, not dragged toward any seed.
    assert 40 < ew.value(0) < 60


def test_fallback_ladder():
    book = PitcherBook(half_life=10, min_starts=5, rookie_shrink=0.5)
    # Give the LAD staff history from a veteran (6 starts, all GS 70).
    for i in range(6):
        book.record_start(f"2024-04-0{i + 1}", "LAD", "vet", 70)
    book.advance_to("2024-05-01")

    vet = book.pregame_adj("LAD", "vet", "2024-05-01")
    assert vet["mode"] == "pitcher"
    assert vet["adj"] == pytest.approx(0.0, abs=1e-9)  # vet IS the staff

    # TBD: staff rGS, adjustment exactly zero.
    tbd = book.pregame_adj("LAD", None, "2024-05-01")
    assert tbd["mode"] == "staff"
    assert tbd["adj"] == 0.0

    # Rookie with 1 career start: shrunk toward league mean, and the one
    # start does not enter his effective rating.
    book.record_start("2024-04-20", "LAD", "rook", 99)
    book.advance_to("2024-05-01")
    rook = book.pregame_adj("LAD", "rook", "2024-05-01")
    assert rook["mode"] == "thin"
    league, staff = rook["league_rgs"], rook["staff_rgs"]
    assert rook["effective_rgs"] == pytest.approx(
        league + 0.5 * (staff - league))
    # Staff (~70) sits above league mean, so the thin-history discount
    # must pull the adjustment negative.
    assert rook["adj"] < 0

    # A genuinely above-staff veteran gets a positive adjustment.
    for i in range(5):
        book.record_start(f"2024-04-1{i}", "LAD", "ace", 90)
    book.advance_to("2024-05-01")
    ace = book.pregame_adj("LAD", "ace", "2024-05-01")
    assert ace["mode"] == "pitcher"
    assert ace["adj"] > 0


def test_c_scales_adjustment():
    lo, hi = PitcherBook(c=1.0), PitcherBook(c=4.7)
    for book in (lo, hi):
        for i in range(5):
            book.record_start(f"2024-04-0{i + 1}", "LAD", "ace", 90)
        for i in range(5):
            book.record_start(f"2024-04-0{i + 1}", "LAD", "avg", 50)
        book.advance_to("2024-05-01")
    a_lo = lo.pregame_adj("LAD", "ace", "2024-05-01")["adj"]
    a_hi = hi.pregame_adj("LAD", "ace", "2024-05-01")["adj"]
    assert a_hi == pytest.approx(4.7 * a_lo)


def test_league_seed_before_any_data():
    book = PitcherBook()
    res = book.pregame_adj("SEA", None, "2010-04-05")
    assert res["adj"] == 0.0
    assert res["staff_rgs"] == LEAGUE_SEED


class TestLeakage:
    """Every rGS input must predate the game being predicted."""

    def test_same_day_start_is_not_visible(self):
        book = PitcherBook()
        for i in range(5):
            book.record_start(f"2024-04-0{i + 1}", "LAD", "ace", 90)
        # A monster start on game day, buffered but not yet committed.
        book.record_start("2024-05-01", "LAD", "ace", 105)
        book.advance_to("2024-05-01")  # commits only strictly-earlier starts
        res = book.pregame_adj("LAD", "ace", "2024-05-01")
        assert book.pitchers["ace"].n == 5  # the 105 never entered
        assert res["career_starts"] == 5

    def test_committed_same_day_start_raises(self):
        book = PitcherBook()
        book._commit("2024-05-01", "LAD", "ace", 90)
        with pytest.raises(LeakageError):
            book.pregame_adj("LAD", "ace", "2024-05-01")

    def test_committed_future_start_raises_even_for_tbd(self):
        # Leakage through the staff/league books must also be caught.
        book = PitcherBook()
        book._commit("2024-05-02", "LAD", "someone", 55)
        with pytest.raises(LeakageError):
            book.pregame_adj("LAD", None, "2024-05-01")

    def test_doubleheader_game1_not_used_for_game2(self):
        book = PitcherBook()
        for i in range(5):
            book.record_start(f"2024-04-0{i + 1}", "LAD", "g1", 50)
        book.advance_to("2024-05-01")
        before = book.pregame_adj("LAD", None, "2024-05-01")["staff_rgs"]
        book.record_start("2024-05-01", "LAD", "g1", 100)  # DH game 1 result
        book.advance_to("2024-05-01")
        after = book.pregame_adj("LAD", None, "2024-05-01")["staff_rgs"]
        assert after == before
