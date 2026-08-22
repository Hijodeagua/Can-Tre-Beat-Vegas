"""
Write the site JSON: web/public/data/soccer/latest.json plus a per-day
history snapshot — the club-soccer sibling of the MLB card's data files.
Shaped like the MLB export (ratings / slate / futures / ledger under one
roof with a generated_at stamp), plus `league_rankings`: a cross-league
summary (avg Elo, avg squad value/wage, and — where the market_values
uploads have been backfilled with them — avg squad size/age/foreigners/
value-per-player) that powers the site's soccer rankings page.
"""

import json
from datetime import datetime, timezone

import pandas as pd

from soccer.clubs.daily.config import SITE_DIR, SITE_HISTORY, SITE_LATEST
from soccer.clubs.data.leagues import LEAGUES, pool_of
from soccer.clubs.daily.state import DailyState
from soccer.clubs.model.features import values_available, load_market_values_raw

SQUAD_STAT_COLS = ["squad_size", "avg_age", "foreigners", "avg_value_eur_m"]


def ratings_payload(state: DailyState) -> dict:
    out = {}
    for league, lg in LEAGUES.items():
        engine = state.engines[pool_of(league)]
        latest = state.history[state.history["league"] == league]["season"].max()
        table = engine.table(season=latest, league=league)
        out[league] = {
            "name": lg.name,
            "tier": lg.tier,
            "asOfSeason": latest,
            "clubs": [
                {"team": r["team"], "elo": round(float(r["elo"]), 1),
                 "matches": int(r["matches"])}
                for _, r in table.iterrows()
            ],
        }
    return out


def _round(x) -> float | None:
    return None if pd.isna(x) else round(float(x), 3)


def league_rankings_payload(ratings: dict) -> dict:
    """One row per league: current avg Elo (from `ratings`) plus the most
    recent season's squad economics. Value/wage and squad-composition stats
    are reported against whichever season actually has each — coverage
    grows league by league as more market-value uploads land, so the two
    "as of" seasons for one league can legitimately differ."""
    values = load_market_values_raw() if values_available() else pd.DataFrame()

    out = {}
    for league, lg in LEAGUES.items():
        clubs = ratings.get(league, {}).get("clubs", [])
        entry = {
            "name": lg.name,
            "tier": lg.tier,
            "avgElo": round(sum(c["elo"] for c in clubs) / len(clubs), 1) if clubs else None,
            "eloClubCount": len(clubs),
            "valueSeason": None,
            "avgSquadValueEurM": None,
            "avgWageBillEurM": None,
            "valueClubCount": None,
            "squadStatsSeason": None,
            "avgSquadSize": None,
            "avgAge": None,
            "avgForeigners": None,
            "avgValuePerPlayerEurM": None,
            "squadStatsClubCount": None,
        }
        sub = values[values["league"] == league] if len(values) else values
        if len(sub):
            value_season = sorted(sub["season"].unique())[-1]
            vseason = sub[sub["season"] == value_season]
            entry["valueSeason"] = value_season
            entry["avgSquadValueEurM"] = _round(vseason["squad_value_eur_m"].mean())
            entry["valueClubCount"] = int(len(vseason))
            if "wage_bill_eur_m" in vseason.columns and vseason["wage_bill_eur_m"].notna().any():
                entry["avgWageBillEurM"] = _round(vseason["wage_bill_eur_m"].mean())

            has_stats = sub["squad_size"].notna() if "squad_size" in sub.columns else pd.Series(dtype=bool)
            stats_seasons = sorted(sub.loc[has_stats, "season"].unique()) if has_stats.any() else []
            if stats_seasons:
                s_season = stats_seasons[-1]
                sseason = sub[(sub["season"] == s_season) & has_stats]
                entry["squadStatsSeason"] = s_season
                entry["avgSquadSize"] = _round(sseason["squad_size"].mean())
                entry["avgAge"] = _round(sseason["avg_age"].mean())
                entry["avgForeigners"] = _round(sseason["foreigners"].mean())
                entry["avgValuePerPlayerEurM"] = _round(sseason["avg_value_eur_m"].mean())
                entry["squadStatsClubCount"] = int(len(sseason))
        out[league] = entry
    return out


def export(state: DailyState, run_date: str, slate: pd.DataFrame,
           futures: dict, ledger: dict, graded_today: pd.DataFrame) -> None:
    ratings = ratings_payload(state)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_date": run_date,
        "ratings": ratings,
        "league_rankings": league_rankings_payload(ratings),
        "slate": slate.to_dict(orient="records"),
        "graded_today": graded_today.to_dict(orient="records"),
        "ledger": ledger,
        "futures": futures,
    }
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    SITE_HISTORY.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    SITE_LATEST.write_text(text)
    (SITE_HISTORY / f"{run_date}.json").write_text(text)
