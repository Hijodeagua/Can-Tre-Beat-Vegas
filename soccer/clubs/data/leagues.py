"""
League registry + team-name canonicalization for the top-5 European leagues.

The openfootball dataset renamed most clubs to their long legal names partway
through its history (2020-21 for England/Germany/Spain/Italy, 2023-24 for
France), which would split one club's Elo history across two identities.
ALIASES maps every historical spelling to the current canonical name; the
fetch step applies it, so `results.csv` — and everything downstream — only
ever sees one name per club.

Season coverage differs by league because that's where the upstream data
starts; each league's Elo is a fresh start at its own FIRST_SEASON.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class League:
    key: str            # our identifier ("epl", …) used in results.csv + artifacts
    name: str           # display name
    code: str           # openfootball file stem ("en.1", …)
    first_season: str   # earliest season available upstream ("2010-11", …)
    txt_repo: str       # openfootball country repo with Football.TXT files
    txt_file: str       # file stem inside <season>/ in that repo
    country: str        # UEFA 3-letter code, used to map cl/el/conf entries


LEAGUES: dict[str, League] = {
    lg.key: lg
    for lg in [
        League("epl", "Premier League", "en.1", "2010-11", "england", "1-premierleague", "ENG"),
        League("bundesliga", "Bundesliga", "de.1", "2010-11", "deutschland", "1-bundesliga", "GER"),
        League("la_liga", "La Liga", "es.1", "2012-13", "espana", "1-liga", "ESP"),
        League("serie_a", "Serie A", "it.1", "2013-14", "italy", "1-seriea", "ITA"),
        League("ligue_1", "Ligue 1", "fr.1", "2014-15", "france", "1-ligue1", "FRA"),
    ]
}

COUNTRY_TO_LEAGUE = {lg.country: lg.key for lg in LEAGUES.values()}

# Historical name -> current canonical name, per league. Only clubs the
# upstream actually renamed appear here; everything else passes through.
ALIASES: dict[str, dict[str, str]] = {
    "epl": {
        "Aston Villa": "Aston Villa FC",
        "Brighton & Hove Albion": "Brighton & Hove Albion FC",
        "Crystal Palace": "Crystal Palace FC",
        "Leicester City": "Leicester City FC",
        "Manchester City": "Manchester City FC",
        "Manchester United": "Manchester United FC",
        "Newcastle United": "Newcastle United FC",
        "Norwich City": "Norwich City FC",
        "Sheffield United": "Sheffield United FC",
        "Tottenham Hotspur": "Tottenham Hotspur FC",
        "West Bromwich Albion": "West Bromwich Albion FC",
        "West Ham United": "West Ham United FC",
        "Wolverhampton Wanderers": "Wolverhampton Wanderers FC",
    },
    "bundesliga": {
        "1899 Hoffenheim": "TSG 1899 Hoffenheim",
        "Bayer Leverkusen": "Bayer 04 Leverkusen",
        "Bayern München": "FC Bayern München",
        "Bor. Mönchengladbach": "Borussia Mönchengladbach",
        "FC St. Pauli": "FC St. Pauli 1910",
        "SpVgg Greuther Fürth": "SpVgg Greuther Fürth 1903",
        "Werder Bremen": "SV Werder Bremen",
    },
    "la_liga": {
        "Atlético Madrid": "Club Atlético de Madrid",
        "CD Alavés": "Deportivo Alavés",
        "Deportivo La Coruña": "RC Deportivo La Coruña",
        "Espanyol Barcelona": "RCD Espanyol de Barcelona",
        "RC Celta": "RC Celta de Vigo",
        "Rayo Vallecano": "Rayo Vallecano de Madrid",
        "Real Betis": "Real Betis Balompié",
        "Real Madrid": "Real Madrid CF",
        "Real Sociedad": "Real Sociedad de Fútbol",
        "Real Valladolid": "Real Valladolid CF",
    },
    "serie_a": {
        "Atalanta": "Atalanta BC",
        "Bologna FC": "Bologna FC 1909",
        "Hellas Verona": "Hellas Verona FC",
        "Inter": "FC Internazionale Milano",
        "Juventus": "Juventus FC",
        "Lazio Roma": "SS Lazio",
        "Parma FC": "Parma Calcio 1913",
        "Sampdoria": "UC Sampdoria",
        "Sassuolo Calcio": "US Sassuolo Calcio",
    },
    "ligue_1": {
        "AS Monaco": "AS Monaco FC",
        "Olympique Marseille": "Olympique de Marseille",
        "Paris Saint-Germain": "Paris Saint-Germain FC",
        "RC Lens": "Racing Club de Lens",
        "RC Strasbourg": "RC Strasbourg Alsace",
        "Stade Rennais": "Stade Rennais FC 1901",
    },
}


# UEFA-file spellings that differ from BOTH the current canonical name and
# the historical league spellings covered by ALIASES. Applied (then ALIASES)
# when mapping cl/el/conf entries to league pools; extended as the fetch
# step reports unmatched top-5-country clubs.
UEFA_ALIASES: dict[str, dict[str, str]] = {}


def canonical(league_key: str, team: str) -> str:
    return ALIASES.get(league_key, {}).get(team, team)


def next_season(season: str) -> str:
    """"2024-25" -> "2025-26"."""
    y = int(season[:4])
    return f"{y + 1}-{str(y + 2)[-2:]}"


def season_for_date(iso_date: str) -> str:
    """European club season containing a date; July 1 is the boundary."""
    year, month = int(iso_date[:4]), int(iso_date[5:7])
    start = year if month >= 7 else year - 1
    return f"{start}-{str(start + 1)[-2:]}"
