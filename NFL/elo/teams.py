"""Team metadata for the NFL Elo: franchise continuity across relocations,
the current conference/division alignment, and display names.

nflverse labels each game with the abbreviation the franchise used *at the
time* (STL for the 2015 Rams, SD for the 2016 Chargers, OAK for the 2019
Raiders). A move does not reset a roster, so the Elo carries the rating
across it: FRANCHISE maps every historical abbreviation to the current one
and `NFL.elo.engine.load_games` applies it, so everything downstream sees
32 identities from 1999 to today. Same posture as `CFB/data/teams.ALIASES`.

The division map is the 2002-present alignment (the eight-division
realignment that came with Houston's arrival); the spine before 2002 is
burn-in only, so no per-season membership is needed.
"""

FRANCHISE: dict[str, str] = {
    "STL": "LA",    # Rams, St. Louis -> Los Angeles (2016)
    "SD": "LAC",    # Chargers, San Diego -> Los Angeles (2017)
    "OAK": "LV",    # Raiders, Oakland -> Las Vegas (2020)
}

DIVISIONS: dict[str, tuple[str, ...]] = {
    "AFC East": ("BUF", "MIA", "NE", "NYJ"),
    "AFC North": ("BAL", "CIN", "CLE", "PIT"),
    "AFC South": ("HOU", "IND", "JAX", "TEN"),
    "AFC West": ("DEN", "KC", "LAC", "LV"),
    "NFC East": ("DAL", "NYG", "PHI", "WAS"),
    "NFC North": ("CHI", "DET", "GB", "MIN"),
    "NFC South": ("ATL", "CAR", "NO", "TB"),
    "NFC West": ("ARI", "LA", "SEA", "SF"),
}

DIVISION_OF: dict[str, str] = {t: d for d, ts in DIVISIONS.items() for t in ts}
CONFERENCE_OF: dict[str, str] = {t: d.split()[0] for t, d in DIVISION_OF.items()}
TEAMS: tuple[str, ...] = tuple(sorted(DIVISION_OF))
CONFERENCES: tuple[str, ...] = ("AFC", "NFC")

NAMES: dict[str, str] = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

NICKNAMES: dict[str, str] = {t: n.split()[-1] for t, n in NAMES.items()}

# nflverse game_type -> label for the slate and the ledger.
GAME_TYPE_LABEL: dict[str, str] = {
    "REG": "Regular season", "WC": "Wild Card", "DIV": "Divisional",
    "CON": "Conference Championship", "SB": "Super Bowl",
}
PLAYOFF_TYPES: frozenset[str] = frozenset({"WC", "DIV", "CON", "SB"})


def canonical(abbr: str) -> str:
    return FRANCHISE.get(abbr, abbr)


def week_label(week: int, game_type: str) -> str:
    """"Week 7" in the regular season, the round's name in the playoffs."""
    if game_type in PLAYOFF_TYPES:
        return GAME_TYPE_LABEL[game_type]
    return f"Week {int(week)}"


def name(abbr: str) -> str:
    return NAMES.get(abbr, abbr)


def nickname(abbr: str) -> str:
    return NICKNAMES.get(abbr, abbr)
