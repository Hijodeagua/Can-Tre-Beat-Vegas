"""Rest and travel Elo adjustments (FiveThirtyEight's two cheap factors),
each gated behind its own flag so it can be ablated independently.

    travel_adj = -0.31 * miles_traveled ** (1/3), capped at -4 points
    rest_adj   = +2.3 * rest_days, rest_days = min(days since last game - 1, 3)

Both are computed per team from that team's own previous game (venue and
date), walk-forward like everything else: RestTravelBook.update() is called
after each game, pregame() reads only prior state.

Venue coordinates come from a static per-(team, era) table of home-park
locations (data below). At the cube-root-of-miles scale a metro-level
coordinate is plenty: the cap binds at ~2,150 miles and a within-metro park
move shifts the adjustment by well under 0.1 Elo points. The table covers
every 2010-present relocation that moves a club to a different metro
(Athletics to Sacramento 2025, Rays to Tampa 2025, plus within-metro moves
for completeness). Known approximation, documented not modeled: Toronto's
2020-21 Buffalo/Dunedin stints and one-off international/neutral-site games
use the listed home park. A statsapi venue-endpoint refresh can replace this
table where egress allows; the numbers below were seeded from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import asin, cos, radians, sin, sqrt

TRAVEL_COEF = -0.31
TRAVEL_CAP = 4.0          # cap on the magnitude, in Elo points
REST_COEF = 2.3
REST_MAX_DAYS = 3

_EARTH_RADIUS_MILES = 3958.8

# team -> list of (first_season, last_season, lat, lon)
_VENUES: dict[str, list[tuple[int, int, float, float]]] = {
    "ARI": [(2009, 9999, 33.445, -112.067)],
    "ATL": [(2009, 2016, 33.735, -84.389), (2017, 9999, 33.890, -84.468)],
    "BAL": [(2009, 9999, 39.284, -76.622)],
    "BOS": [(2009, 9999, 42.346, -71.097)],
    "CHC": [(2009, 9999, 41.948, -87.655)],
    "CHW": [(2009, 9999, 41.830, -87.634)],
    "CIN": [(2009, 9999, 39.097, -84.507)],
    "CLE": [(2009, 9999, 41.496, -81.685)],
    "COL": [(2009, 9999, 39.756, -104.994)],
    "DET": [(2009, 9999, 42.339, -83.049)],
    "HOU": [(2009, 9999, 29.757, -95.356)],
    "KCR": [(2009, 9999, 39.051, -94.480)],
    "LAA": [(2009, 9999, 33.800, -117.883)],
    "LAD": [(2009, 9999, 34.074, -118.240)],
    "MIA": [(2009, 2011, 25.958, -80.239), (2012, 9999, 25.778, -80.220)],
    "MIL": [(2009, 9999, 43.028, -87.971)],
    "MIN": [(2009, 9999, 44.982, -93.278)],
    "NYM": [(2009, 9999, 40.757, -73.846)],
    "NYY": [(2009, 9999, 40.829, -73.926)],
    "ATH": [(2009, 2024, 37.751, -122.200), (2025, 9999, 38.580, -121.513)],
    "PHI": [(2009, 9999, 39.906, -75.166)],
    "PIT": [(2009, 9999, 40.447, -80.006)],
    "SDP": [(2009, 9999, 32.707, -117.157)],
    "SEA": [(2009, 9999, 47.591, -122.332)],
    "SFG": [(2009, 9999, 37.778, -122.389)],
    "STL": [(2009, 9999, 38.622, -90.193)],
    "TBR": [(2009, 2024, 27.768, -82.653), (2025, 9999, 27.980, -82.507)],
    "TEX": [(2009, 9999, 32.751, -97.083)],
    "TOR": [(2009, 9999, 43.641, -79.389)],
    "WSN": [(2009, 9999, 38.873, -77.008)],
}


def venue_coords(home_team: str, season: int) -> tuple[float, float]:
    for first, last, lat, lon in _VENUES[home_team]:
        if first <= season <= last:
            return lat, lon
    raise KeyError(f"no venue for {home_team} in {season}")


def haversine_miles(lat1: float, lon1: float,
                    lat2: float, lon2: float) -> float:
    la1, lo1, la2, lo2 = map(radians, (lat1, lon1, lat2, lon2))
    a = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 2 * _EARTH_RADIUS_MILES * asin(sqrt(a))


def travel_adjustment(miles: float) -> float:
    return max(TRAVEL_COEF * miles ** (1.0 / 3.0), -TRAVEL_CAP)


def rest_adjustment(days_since_last_game: int) -> float:
    rest_days = min(max(days_since_last_game - 1, 0), REST_MAX_DAYS)
    return REST_COEF * rest_days


@dataclass
class RestTravelBook:
    """Per-team walk-forward state: where and when each team last played."""
    use_rest: bool = True
    use_travel: bool = True
    last: dict = field(default_factory=dict)  # team -> (date, lat, lon)

    def pregame(self, team: str, date: str, season: int,
                home_team: str) -> dict:
        """Adjustments for `team` playing at `home_team`'s park on `date`."""
        lat, lon = venue_coords(home_team, season)
        rest = travel = 0.0
        prev = self.last.get(team)
        if prev is not None:
            prev_date, prev_lat, prev_lon = prev
            gap = (_ord(date) - _ord(prev_date))
            if self.use_rest:
                rest = rest_adjustment(gap)
            if self.use_travel:
                travel = travel_adjustment(
                    haversine_miles(prev_lat, prev_lon, lat, lon))
        return {"rest_adj": rest, "travel_adj": travel,
                "adj": rest + travel}

    def update(self, team: str, date: str, season: int,
               home_team: str) -> None:
        lat, lon = venue_coords(home_team, season)
        self.last[team] = (date, lat, lon)


def _ord(date: str) -> int:
    from datetime import date as _d
    return _d.fromisoformat(date).toordinal()
