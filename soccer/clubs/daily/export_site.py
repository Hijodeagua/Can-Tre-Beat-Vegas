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
from soccer.clubs.data.leagues import LEAGUES, TIER1, pool_of
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


def elo_history_payload(state: DailyState, run_date: str) -> dict:
    """Per top flight, each club's current-season Elo trajectory: the
    pre-match rating at every match date, closed with today's live rating.
    The replay updates ratings after every matchday and this export runs
    daily, so the final point (and any post-matchday step) moves each day.
    Powers the site's Elo trend chart."""
    out = {}
    for league in TIER1:
        sub = state.history[state.history["league"] == league]
        if sub.empty:
            continue
        season = sub["season"].max()
        sub = sub[sub["season"] == season].sort_values("date")
        engine = state.engines[pool_of(league)]
        series: dict[str, list] = {}
        for r in sub.itertuples():
            series.setdefault(r.home_team, []).append(
                [r.date, round(float(r.elo_home_pre), 1)])
            series.setdefault(r.away_team, []).append(
                [r.date, round(float(r.elo_away_pre), 1)])
        for team, points in series.items():
            current = round(float(engine.rating_for(team, league)), 1)
            if not points or points[-1][0] < run_date:
                points.append([run_date, current])
            else:
                points[-1] = [points[-1][0], current]
        out[league] = {"season": season, "clubs": series}
    return out


def _top3(series) -> float | None:
    """Mean of the three largest values — the league's top-end alongside
    its overall average. None until at least three clubs report."""
    s = pd.Series(series).dropna()
    if len(s) < 3:
        return None
    return round(float(s.nlargest(3).mean()), 3)


def _elo_bands(elos: list[float]) -> dict:
    """Top-4 / middle-10 / bottom-4 mean Elo — the league's ceiling, its
    midtable, and its floor. Mid-10 is the ten clubs centered on the
    median rank, so an 18-club league reads ranks 5-14 and a 24-club
    league 8-17. None where a band doesn't fit the league size."""
    s = sorted(elos, reverse=True)
    n = len(s)
    out = {"top4Elo": None, "mid10Elo": None, "bottom4Elo": None}
    if n >= 4:
        out["top4Elo"] = round(sum(s[:4]) / 4, 1)
        out["bottom4Elo"] = round(sum(s[-4:]) / 4, 1)
    if n >= 10:
        start = (n - 10) // 2
        out["mid10Elo"] = round(sum(s[start:start + 10]) / 10, 1)
    return out


def league_rankings_payload(ratings: dict) -> dict:
    """One row per league: current avg Elo (from `ratings`) plus the most
    recent season's squad economics, each paired with a top-3-club average
    (the top end of the league, not just its middle). Value/wage and
    squad-composition stats are reported against whichever season actually
    has each — coverage grows league by league as more market-value
    uploads land, so the "as of" seasons for one league can legitimately
    differ. Wage bills additionally fall back to the most recent season
    carrying the optional wage column, since value coverage runs ahead of
    wage coverage."""
    values = load_market_values_raw() if values_available() else pd.DataFrame()

    out = {}
    for league, lg in LEAGUES.items():
        clubs = ratings.get(league, {}).get("clubs", [])
        entry = {
            "name": lg.name,
            "tier": lg.tier,
            "avgElo": round(sum(c["elo"] for c in clubs) / len(clubs), 1) if clubs else None,
            **_elo_bands([c["elo"] for c in clubs]),
            "eloClubCount": len(clubs),
            "valueSeason": None,
            "avgSquadValueEurM": None,
            "top3SquadValueEurM": None,
            "avgWageBillEurM": None,
            "top3WageBillEurM": None,
            "wageSeason": None,
            "valueClubCount": None,
            "squadStatsSeason": None,
            "avgSquadSize": None,
            "avgAge": None,
            "avgForeigners": None,
            "avgValuePerPlayerEurM": None,
            "top3ValuePerPlayerEurM": None,
            "squadStatsClubCount": None,
        }
        sub = values[values["league"] == league] if len(values) else values
        if len(sub):
            value_season = sorted(sub["season"].unique())[-1]
            vseason = sub[sub["season"] == value_season]
            entry["valueSeason"] = value_season
            entry["avgSquadValueEurM"] = _round(vseason["squad_value_eur_m"].mean())
            entry["top3SquadValueEurM"] = _top3(vseason["squad_value_eur_m"])
            entry["valueClubCount"] = int(len(vseason))
            if "wage_bill_eur_m" in sub.columns and sub["wage_bill_eur_m"].notna().any():
                has_wage = sub["wage_bill_eur_m"].notna()
                wage_season = sorted(sub.loc[has_wage, "season"].unique())[-1]
                wseason = sub[(sub["season"] == wage_season) & has_wage]
                entry["wageSeason"] = wage_season
                entry["avgWageBillEurM"] = _round(wseason["wage_bill_eur_m"].mean())
                entry["top3WageBillEurM"] = _top3(wseason["wage_bill_eur_m"])

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
                entry["top3ValuePerPlayerEurM"] = _top3(sseason["avg_value_eur_m"])
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
        "elo_history": elo_history_payload(state, run_date),
    }
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    SITE_HISTORY.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    SITE_LATEST.write_text(text)
    (SITE_HISTORY / f"{run_date}.json").write_text(text)
