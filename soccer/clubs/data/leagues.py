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
    tier: int = 1       # 1 = top flight, 2 = second division
    pool: str = ""      # Elo pool key — the country's tier-1 league key


LEAGUES: dict[str, League] = {
    lg.key: lg
    for lg in [
        League("epl", "Premier League", "en.1", "2010-11", "england", "1-premierleague", "ENG", 1, "epl"),
        League("championship", "Championship", "en.2", "2010-11", "england", "2-championship", "ENG", 2, "epl"),
        League("bundesliga", "Bundesliga", "de.1", "2010-11", "deutschland", "1-bundesliga", "GER", 1, "bundesliga"),
        League("bundesliga_2", "2. Bundesliga", "de.2", "2012-13", "deutschland", "2-bundesliga2", "GER", 2, "bundesliga"),
        League("la_liga", "La Liga", "es.1", "2012-13", "espana", "1-liga", "ESP", 1, "la_liga"),
        League("la_liga_2", "Segunda División", "es.2", "2012-13", "espana", "2-liga2", "ESP", 2, "la_liga"),
        League("serie_a", "Serie A", "it.1", "2013-14", "italy", "1-seriea", "ITA", 1, "serie_a"),
        League("serie_b", "Serie B", "it.2", "2013-14", "italy", "2-serieb", "ITA", 2, "serie_a"),
        League("ligue_1", "Ligue 1", "fr.1", "2014-15", "france", "1-ligue1", "FRA", 1, "ligue_1"),
        League("ligue_2", "Ligue 2", "fr.2", "2014-15", "france", "2-ligue2", "FRA", 2, "ligue_1"),
    ]
}

TIER1 = {k: lg for k, lg in LEAGUES.items() if lg.tier == 1}
POOLS: dict[str, list[str]] = {
    pool: [k for k, lg in LEAGUES.items() if lg.pool == pool] for pool in TIER1
}


def pool_of(league_key: str) -> str:
    return LEAGUES[league_key].pool


# UEFA entries always belong to the country's top flight pool.
COUNTRY_TO_LEAGUE = {lg.country: lg.key for lg in TIER1.values()}

# Historical name -> current canonical name, per league. Only clubs the
# upstream actually renamed appear here; everything else passes through.
# Second-division dicts map to the SAME canonical names as the top flight,
# so a club keeps one Elo identity as it moves between divisions. Italian
# phoenix clubs (Bari, Palermo, Cesena, …) are deliberately treated as one
# continuous identity — the Serie D exile between incarnations shows up as
# seasons of rollover regression, not a fresh club.
ALIASES: dict[str, dict[str, str]] = {
    "epl": {
        "Aston Villa": "Aston Villa FC",
        "Birmingham City": "Birmingham City FC",
        "Blackburn Rovers": "Blackburn Rovers FC",
        "Bolton Wanderers": "Bolton Wanderers FC",
        "Brighton & Hove Albion": "Brighton & Hove Albion FC",
        "Cardiff City": "Cardiff City FC",
        "Crystal Palace": "Crystal Palace FC",
        "Huddersfield Town": "Huddersfield Town AFC",
        "Hull City": "Hull City AFC",
        "Leeds United": "Leeds United FC",
        "Queens Park Rangers": "Queens Park Rangers FC",
        "Stoke City": "Stoke City FC",
        "Swansea City": "Swansea City AFC",
        "Wigan Athletic": "Wigan Athletic FC",
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
    "championship": {
        "Aston Villa": "Aston Villa FC",
        "Birmingham City": "Birmingham City FC",
        "Blackburn Rovers": "Blackburn Rovers FC",
        "Bolton Wanderers": "Bolton Wanderers FC",
        "Brighton & Hove Albion": "Brighton & Hove Albion FC",
        "Bristol City": "Bristol City FC",
        "Cardiff City": "Cardiff City FC",
        "Charlton Athletic": "Charlton Athletic FC",
        "Coventry City": "Coventry City FC",
        "Crystal Palace": "Crystal Palace FC",
        "Derby County": "Derby County FC",
        "Huddersfield Town": "Huddersfield Town AFC",
        "Hull City": "Hull City AFC",
        "Ipswich Town": "Ipswich Town FC",
        "Leeds United": "Leeds United FC",
        "Leicester City": "Leicester City FC",
        "Luton Town": "Luton Town FC",
        "Newcastle United": "Newcastle United FC",
        "Norwich City": "Norwich City FC",
        "Nottingham Forest": "Nottingham Forest FC",
        "Oxford United": "Oxford United FC",
        "Peterborough United": "Peterborough United FC",
        "Plymouth Argyle": "Plymouth Argyle FC",
        "Preston North End": "Preston North End FC",
        "Queens Park Rangers": "Queens Park Rangers FC",
        "Rotherham United": "Rotherham United FC",
        "Sheffield United": "Sheffield United FC",
        "Sheffield Wednesday": "Sheffield Wednesday FC",
        "Stoke City": "Stoke City FC",
        "Swansea City": "Swansea City AFC",
        "West Bromwich Albion": "West Bromwich Albion FC",
        "West Ham United": "West Ham United FC",
        "Wigan Athletic": "Wigan Athletic FC",
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
    "bundesliga_2": {
        "FC St. Pauli": "FC St. Pauli 1910",
        "SpVgg Greuther Fürth": "SpVgg Greuther Fürth 1903",
        "VfL Bochum": "VfL Bochum 1848",
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
    "la_liga_2": {
        "CD Alavés": "Deportivo Alavés",
        "Deportivo La Coruña": "RC Deportivo La Coruña",
        "Espanyol Barcelona": "RCD Espanyol de Barcelona",
        "Rayo Vallecano": "Rayo Vallecano de Madrid",
        "Real Betis": "Real Betis Balompié",
        "Real Valladolid": "Real Valladolid CF",
    },
    "serie_a": {
        "AC Cesena": "Cesena FC",
        "Atalanta": "Atalanta BC",
        "Bologna FC": "Bologna FC 1909",
        "Hellas Verona": "Hellas Verona FC",
        "Inter": "FC Internazionale Milano",
        "Juventus": "Juventus FC",
        "Lazio Roma": "SS Lazio",
        "Parma FC": "Parma Calcio 1913",
        "Sampdoria": "UC Sampdoria",
        "Sassuolo Calcio": "US Sassuolo Calcio",
        "US Palermo": "Palermo FC",
    },
    "serie_b": {
        "AC Cesena": "Cesena FC",
        "AC Pisa": "AC Pisa 1909",
        "AS Avellino": "US Avellino",
        "AS Bari": "SSC Bari",
        "Bologna FC": "Bologna FC 1909",
        "Como Calcio": "Como 1907",
        "FC Bari 1908": "SSC Bari",
        "Hellas Verona": "Hellas Verona FC",
        "Pisa SC": "AC Pisa 1909",
        "Reggina Calcio": "Reggina 1914",
        "Sampdoria": "UC Sampdoria",
        "Sassuolo Calcio": "US Sassuolo Calcio",
        "US Palermo": "Palermo FC",
    },
    "ligue_1": {
        "AS Monaco": "AS Monaco FC",
        "Olympique Marseille": "Olympique de Marseille",
        "Paris Saint-Germain": "Paris Saint-Germain FC",
        "RC Lens": "Racing Club de Lens",
        "RC Strasbourg": "RC Strasbourg Alsace",
        "Stade Rennais": "Stade Rennais FC 1901",
    },
    "ligue_2": {
        "AS Monaco": "AS Monaco FC",
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
