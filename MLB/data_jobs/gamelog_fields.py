"""
Retrosheet game log field schema.

A game log line is a quoted CSV record with 161 fields. The layout is fixed
and documented by Retrosheet ("Game Logs" file description); field names here
follow that spec, prefixed away_/home_ from this pipeline's perspective.
Offensive/pitching/defensive stat blocks are per-team totals for the game.
"""

# Names for all 161 fields, in file order.
_OFFENSE = [
    "ab", "h", "d2", "d3", "hr", "rbi", "sh", "sf", "hbp",
    "bb", "ibb", "so", "sb", "cs", "gidp", "ci", "lob",
]
_PITCHING = ["pitchers_used", "indiv_er", "team_er", "wp", "balks"]
_DEFENSE = ["po", "a", "e", "pb", "dp", "tp"]


def _block(prefix, names):
    return [f"{prefix}_{n}" for n in names]


def _lineup(prefix):
    cols = []
    for slot in range(1, 10):
        cols += [
            f"{prefix}_bat{slot}_id",
            f"{prefix}_bat{slot}_name",
            f"{prefix}_bat{slot}_pos",
        ]
    return cols


GAMELOG_COLUMNS = (
    [
        "date", "game_num", "day_of_week",
        "away_team_retro", "away_league", "away_game_num",
        "home_team_retro", "home_league", "home_game_num",
        "away_score", "home_score",
        "outs_total", "day_night",
        "completion_info", "forfeit_info", "protest_info",
        "park_id", "attendance", "duration_minutes",
        "away_linescore", "home_linescore",
    ]
    + _block("away", _OFFENSE)
    + _block("away", _PITCHING)
    + _block("away", _DEFENSE)
    + _block("home", _OFFENSE)
    + _block("home", _PITCHING)
    + _block("home", _DEFENSE)
    + [
        "ump_hp_id", "ump_hp_name",
        "ump_1b_id", "ump_1b_name",
        "ump_2b_id", "ump_2b_name",
        "ump_3b_id", "ump_3b_name",
        "ump_lf_id", "ump_lf_name",
        "ump_rf_id", "ump_rf_name",
        "away_manager_id", "away_manager_name",
        "home_manager_id", "home_manager_name",
        "winning_pitcher_id", "winning_pitcher_name",
        "losing_pitcher_id", "losing_pitcher_name",
        "save_pitcher_id", "save_pitcher_name",
        "gwrbi_batter_id", "gwrbi_batter_name",
        "away_sp_id", "away_sp_name",
        "home_sp_id", "home_sp_name",
    ]
    + _lineup("away")
    + _lineup("home")
    + ["additional_info", "acquisition_info"]
)

assert len(GAMELOG_COLUMNS) == 161, len(GAMELOG_COLUMNS)

# Numeric columns (parsed to nullable ints; Retrosheet uses -1/blank for
# unknown attendance/duration, normalized to NA by the parser).
NUMERIC_COLUMNS = (
    ["away_game_num", "home_game_num", "away_score", "home_score",
     "outs_total", "attendance", "duration_minutes"]
    + _block("away", _OFFENSE) + _block("away", _PITCHING) + _block("away", _DEFENSE)
    + _block("home", _OFFENSE) + _block("home", _PITCHING) + _block("home", _DEFENSE)
)
