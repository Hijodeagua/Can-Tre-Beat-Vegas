"""
Reusable fixtures modeled on The Odds API v4 /sports/{sport}/odds response shape.

See https://the-odds-api.com/liveapi/guides/v4/ - each game is a dict with
id, sport_key, commence_time, home_team, away_team, and a list of bookmakers,
each with markets (h2h / spreads / totals) containing outcomes.
"""


def make_game(
    home_team="Kansas City Chiefs",
    away_team="Buffalo Bills",
    game_id="abc123def456",
    commence_time="2026-09-10T00:15:00Z",
    bookmakers=None,
):
    """Build a single game payload in The Odds API v4 shape"""
    if bookmakers is None:
        bookmakers = [
            make_bookmaker(
                "DraftKings",
                home_team,
                away_team,
                home_h2h=-150,
                away_h2h=130,
                home_spread=-110,
                away_spread=-110,
                home_point=-3.5,
                away_point=3.5,
                over=-110,
                under=-110,
                total=47.5,
            ),
            make_bookmaker(
                "FanDuel",
                home_team,
                away_team,
                home_h2h=-155,
                away_h2h=135,
                home_spread=-108,
                away_spread=-112,
                home_point=-3.5,
                away_point=3.5,
                over=-112,
                under=-108,
                total=48.5,
            ),
        ]

    return {
        "id": game_id,
        "sport_key": "americanfootball_nfl",
        "sport_title": "NFL",
        "commence_time": commence_time,
        "home_team": home_team,
        "away_team": away_team,
        "bookmakers": bookmakers,
    }


def make_bookmaker(
    title,
    home_team,
    away_team,
    home_h2h=-150,
    away_h2h=130,
    home_spread=-110,
    away_spread=-110,
    home_point=-3.5,
    away_point=3.5,
    over=-110,
    under=-110,
    total=47.5,
):
    """Build a bookmaker entry with h2h, spreads, and totals markets"""
    return {
        "key": title.lower().replace(" ", ""),
        "title": title,
        "last_update": "2026-09-09T12:00:00Z",
        "markets": [
            {
                "key": "h2h",
                "last_update": "2026-09-09T12:00:00Z",
                "outcomes": [
                    {"name": home_team, "price": home_h2h},
                    {"name": away_team, "price": away_h2h},
                ],
            },
            {
                "key": "spreads",
                "last_update": "2026-09-09T12:00:00Z",
                "outcomes": [
                    {"name": home_team, "price": home_spread, "point": home_point},
                    {"name": away_team, "price": away_spread, "point": away_point},
                ],
            },
            {
                "key": "totals",
                "last_update": "2026-09-09T12:00:00Z",
                "outcomes": [
                    {"name": "Over", "price": over, "point": total},
                    {"name": "Under", "price": under, "point": total},
                ],
            },
        ],
    }
