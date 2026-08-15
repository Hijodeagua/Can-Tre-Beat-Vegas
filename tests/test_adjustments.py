"""Tests for the rest/travel adjustments (mlb/adjustments.py)."""

import pytest

from mlb.adjustments import (
    RestTravelBook, haversine_miles, rest_adjustment, travel_adjustment,
    venue_coords,
)


def test_travel_formula_and_cap():
    assert travel_adjustment(0) == 0.0
    assert travel_adjustment(1000) == pytest.approx(-3.1)
    # Transcontinental trips hit the cap (-0.31 * m^(1/3) <= -4 above ~2150mi).
    assert travel_adjustment(2500) == -4.0


def test_rest_formula_and_cap():
    assert rest_adjustment(1) == 0.0            # played yesterday: no rest
    assert rest_adjustment(2) == pytest.approx(2.3)
    assert rest_adjustment(4) == pytest.approx(6.9)
    assert rest_adjustment(30) == pytest.approx(6.9)  # capped at 3 days


def test_haversine_lax_to_jfk_ballpark():
    # Dodger Stadium to Yankee Stadium is about 2,460 miles great-circle.
    miles = haversine_miles(34.074, -118.240, 40.829, -73.926)
    assert 2350 < miles < 2550


def test_venue_eras():
    assert venue_coords("ATH", 2024)[0] == pytest.approx(37.751)   # Oakland
    assert venue_coords("ATH", 2025)[0] == pytest.approx(38.580)   # Sacramento
    assert venue_coords("TBR", 2024) != venue_coords("TBR", 2025)


def test_book_walk_forward_and_flags():
    book = RestTravelBook()
    # No prior state: both zero.
    first = book.pregame("LAD", "2024-04-01", 2024, "LAD")
    assert first["adj"] == 0.0
    book.update("LAD", "2024-04-01", 2024, "LAD")

    # Next day, cross-country to NYY: no rest, capped travel.
    away = book.pregame("LAD", "2024-04-02", 2024, "NYY")
    assert away["rest_adj"] == 0.0
    assert away["travel_adj"] == -4.0
    book.update("LAD", "2024-04-02", 2024, "NYY")

    # Four days after the last game (three full off days): max rest,
    # no travel.
    rested = book.pregame("LAD", "2024-04-06", 2024, "NYY")
    assert rested["rest_adj"] == pytest.approx(6.9)
    assert rested["travel_adj"] == 0.0

    # Ablation flags zero out their component independently.
    no_travel = RestTravelBook(use_travel=False)
    no_travel.update("LAD", "2024-04-01", 2024, "LAD")
    res = no_travel.pregame("LAD", "2024-04-02", 2024, "NYY")
    assert res["travel_adj"] == 0.0 and res["rest_adj"] == 0.0

    no_rest = RestTravelBook(use_rest=False)
    no_rest.update("LAD", "2024-04-01", 2024, "LAD")
    res = no_rest.pregame("LAD", "2024-04-06", 2024, "NYY")
    assert res["rest_adj"] == 0.0 and res["travel_adj"] == -4.0
