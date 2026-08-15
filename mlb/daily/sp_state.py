"""Live starting-pitcher and rest/travel state for the daily slate.

Rebuilt from scratch on every run - same philosophy as ratings.py: replaying
data/mlb/pitcher_starts.csv (~75k starts) and the games file takes under two
seconds and leaves no incremental state on disk to drift.

The PitcherBook's leakage contract carries straight into production: every
start is buffered via record_start and only commits once advance_to(run_date)
proves it predates the slate being predicted.
"""

from __future__ import annotations

import csv

import pandas as pd

from mlb.adjustments import RestTravelBook
from mlb.daily.config import (
    SP_C, SP_HALF_LIFE, STARTS_CSV, USE_PITCHER_ADJ, USE_REST, USE_TRAVEL,
)
from mlb.pitcher_rating import PitcherBook, game_score


def build_books(games: pd.DataFrame, run_date: str) -> tuple[PitcherBook,
                                                             RestTravelBook]:
    """Pitcher/staff rGS book and rest-travel book, current through the day
    before `run_date`."""
    pbook = PitcherBook(half_life=SP_HALF_LIFE, c=SP_C)
    if STARTS_CSV.exists():
        with open(STARTS_CSV, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                gs = game_score(int(r["outs"]), int(r["h"]), int(r["r"]),
                                int(r["er"]), int(r["bb"]), int(r["so"]))
                pbook.record_start(r["date"], r["team"], r["mlbam_id"], gs)
    pbook.advance_to(run_date)

    rtbook = RestTravelBook(use_rest=USE_REST, use_travel=USE_TRAVEL)
    played = games[games.date < run_date].sort_values(["date", "game_num"])
    for g in played.itertuples(index=False):
        for team in (g.home_fr, g.away_fr):
            rtbook.update(team, g.date, int(g.season), g.home_fr)
    return pbook, rtbook


def slate_adjustments(pbook: PitcherBook, rtbook: RestTravelBook,
                      todays: list[dict]) -> dict[tuple, dict]:
    """Per-game adjustment audit for one day's schedule rows, keyed like the
    slate: (date, game_num, home_fr). Elo points, home and away."""
    out: dict[tuple, dict] = {}
    for g in todays:
        date, season = g["date"], int(g["season"])
        rec: dict[str, float | str] = {}
        for side in ("home", "away"):
            team = g[f"{side}_fr"]
            sp_uid = g.get(f"{side}_sp_id") or None
            if USE_PITCHER_ADJ:
                sp = pbook.pregame_adj(team, sp_uid, date)
            else:
                sp = {"adj": 0.0, "mode": "off", "career_starts": 0}
            rt = rtbook.pregame(team, date, season, g["home_fr"])
            rec[f"{side}_sp_adj"] = round(sp["adj"], 2)
            rec[f"{side}_sp_mode"] = sp["mode"]
            rec[f"{side}_rt_adj"] = round(rt["adj"], 2)
            rec[f"{side}_adj"] = rec[f"{side}_sp_adj"] + rec[f"{side}_rt_adj"]
        out[(date, int(g["game_num"]), g["home_fr"])] = rec
    return out
