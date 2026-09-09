"""Write the site JSON: web/public/data/nfl/latest.json plus a per-day
history snapshot — the NFL sibling of the CFB card's data file, shaped
the same way (ratings / divisions / slate / ledger / futures /
elo_history under one roof with a generated_at stamp).

`ratings` is the full 32-team board ranked by Elo with season-to-date
record; `divisions` is the eight-division summary that powers the
division-strength table; `elo_history` is every team's current-season
Elo trajectory for the trend chart.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from NFL.daily.config import SITE_DIR, SITE_HISTORY, SITE_LATEST
from NFL.daily.predict import next_week
from NFL.daily.simulate import current_records
from NFL.daily.state import DailyState, build_state
from NFL.elo.teams import CONFERENCE_OF, DIVISION_OF, DIVISIONS, TEAMS, name, week_label


def _clean(v):
    if v is None or (isinstance(v, float) and v != v):
        return None
    return v


def _records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def preseason_ratings(state: DailyState) -> dict[str, float]:
    """Every team's rating on the morning of the season's first game —
    after the off-season regression, before any result."""
    season_games = state.games[state.games["season"] == state.season]
    if season_games.empty:
        return {}
    first = str(season_games["date"].min())
    pre = build_state(state.games, run_date=first)
    return {t: float(pre.engine.rating_for(t)) for t in TEAMS}


def ratings_payload(state: DailyState,
                    preseason: dict[str, float] | None = None) -> list[dict]:
    rec = current_records(state.games, state.season).set_index("team")
    pre = preseason if preseason is not None else preseason_ratings(state)
    pre_rank = {t: i for i, t in enumerate(sorted(pre, key=lambda t: -pre[t]), start=1)}
    rows = []
    for t in TEAMS:
        rows.append({
            "team": t,
            "name": name(t),
            "conference": CONFERENCE_OF[t],
            "division": DIVISION_OF[t],
            "elo": round(float(state.engine.rating_for(t)), 1),
            "wins": int(rec.loc[t, "wins"]),
            "losses": int(rec.loc[t, "losses"]),
            "ties": int(rec.loc[t, "ties"]),
            "div_wins": int(rec.loc[t, "div_wins"]),
            "div_losses": int(rec.loc[t, "div_losses"]),
            "pts_diff": int(rec.loc[t, "pts_diff"]),
            "games": int(state.engine.games_played.get(t, 0)),
            "preseason_elo": round(pre[t], 1) if t in pre else None,
            "preseason_rank": pre_rank.get(t),
        })
    rows.sort(key=lambda r: -r["elo"])
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


def divisions_payload(ratings: list[dict]) -> dict:
    """Per division: Elo average (current and preseason), best and worst
    team, teams in the top 10 — the NFL analogue of the CFB conference
    table."""
    by_div: dict[str, list[dict]] = {}
    for r in ratings:
        by_div.setdefault(r["division"], []).append(r)
    out = {}
    for div in DIVISIONS:
        members = sorted(by_div.get(div, []), key=lambda m: -m["elo"])
        if not members:
            continue
        pre = [m["preseason_elo"] for m in members if m.get("preseason_elo") is not None]
        out[div] = {
            "name": div,
            "conference": div.split()[0],
            "teams": len(members),
            "avgElo": round(sum(m["elo"] for m in members) / len(members), 1),
            "preseasonAvgElo": round(sum(pre) / len(pre), 1) if pre else None,
            "top10": sum(1 for m in members if m["rank"] <= 10),
            "bestTeam": members[0]["team"],
            "bestElo": members[0]["elo"],
            "worstTeam": members[-1]["team"],
            "worstElo": members[-1]["elo"],
        }
    return out


def elo_history_payload(state: DailyState, run_date: str,
                        preseason: dict[str, float] | None = None) -> dict:
    """Every team's current-season trajectory: the post-regression
    preseason rating, the pre-game Elo at each game date, and today's
    live rating — so the chart moves each day the job runs."""
    season = state.season
    pre = preseason if preseason is not None else preseason_ratings(state)
    season_games = state.games[state.games["season"] == season]
    first = str(season_games["date"].min()) if not season_games.empty else run_date
    sub = state.history[state.history["season"] == season].sort_values(["date", "gametime"]) \
        if "gametime" in state.history else state.history[state.history["season"] == season].sort_values("date")
    series: dict[str, list] = {}
    open_date = min(first, run_date)
    for t in TEAMS:
        series[t] = [[open_date, round(pre[t], 1)]] if t in pre else []
    for r in sub.itertuples():
        for team, elo in ((r.home_team, r.elo_home_pre), (r.away_team, r.elo_away_pre)):
            pts = series.setdefault(team, [])
            if pts and pts[-1][0] == r.date:
                continue
            pts.append([r.date, round(float(elo), 1)])
    for t in TEAMS:
        points = series.setdefault(t, [])
        current = round(float(state.engine.rating_for(t)), 1)
        if not points or points[-1][0] < run_date:
            points.append([run_date, current])
        else:
            points[-1] = [points[-1][0], current]
    return {"season": season, "teams": series}


def current_week(state: DailyState, run_date: str) -> tuple[int | None, str | None]:
    nw = next_week(state, run_date)
    if nw is None:
        g = state.games[state.games["season"] == state.season]
        if g.empty:
            return None, None
        last = g.sort_values(["week"]).iloc[-1]
        return int(last["week"]), week_label(int(last["week"]), last["game_type"])
    season, week = nw
    gt = state.games[(state.games["season"] == season) & (state.games["week"] == week)]["game_type"].iloc[0]
    return week, week_label(week, gt)


def export(state: DailyState, run_date: str, slate: pd.DataFrame,
           futures: dict | None, ledger: dict, graded_today: pd.DataFrame,
           recent: pd.DataFrame) -> dict:
    pre = preseason_ratings(state)
    ratings = ratings_payload(state, pre)
    week, label = current_week(state, run_date)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_date": run_date,
        "season": state.season,
        "week": week,
        "week_label": label,
        "params": {
            "k": state.engine.k,
            "home_advantage": state.engine.home_advantage,
            "season_regression": state.engine.season_regression,
            "playoff_k_mult": state.engine.playoff_k_mult,
            "margin_cap": state.engine.margin_cap,
            "rest_bonus": state.engine.rest_bonus,
            "elo_per_point": round(state.score_params.elo_per_point, 1),
        },
        "ratings": ratings,
        "divisions": divisions_payload(ratings),
        "slate": _records(slate),
        "graded_today": _records(graded_today),
        "graded_recent": _records(recent),
        "ledger": ledger,
        "futures": futures or {"season": state.season, "status": "no_games"},
        "elo_history": elo_history_payload(state, run_date, pre),
    }
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    SITE_HISTORY.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=1, default=_clean) + "\n"
    SITE_LATEST.write_text(text)
    (SITE_HISTORY / f"{run_date}.json").write_text(text)
    return payload
