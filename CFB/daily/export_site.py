"""Write the site JSON: web/public/data/cfb/latest.json plus a per-day
history snapshot — the college-football sibling of the MLB and soccer
cards' data files, shaped like the soccer export (ratings / slate /
futures / ledger / elo_history under one roof with a generated_at stamp).

`ratings` is the full FBS board for the season (every program, ranked by
Elo, with season-to-date record); `conferences` is the cross-conference
summary (avg / top-4 / bottom-4 Elo per conference) that powers the
site's conference-strength table; `elo_history` is every program's
current-season Elo trajectory for the trend chart.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from CFB.daily.config import SITE_DIR, SITE_HISTORY, SITE_LATEST, TOP_N
from CFB.daily.simulate import current_records
from CFB.daily.state import DailyState, build_state
from CFB.data.teams import conference_short
from CFB.model.elo import INDEPENDENT


def _clean(v):
    if v is None or (isinstance(v, float) and v != v):
        return None
    return v


def _records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def preseason_ratings(state: DailyState) -> dict[str, float]:
    """Every program's rating on the morning of the season's first game —
    after the August regression, before any result. The Top 25's "Pre"
    column and the conference table's preseason view read from this."""
    season_games = state.games[state.games["season"] == state.season]
    if season_games.empty:
        return {}
    first = str(season_games["date"].min())
    pre = build_state(state.games, run_date=first)
    return {t: float(pre.engine.rating_for(t)) for t in pre.fbs_teams()}


def ratings_payload(state: DailyState,
                    preseason: dict[str, float] | None = None) -> list[dict]:
    teams = state.fbs_teams()
    rec = current_records(state.games, state.season, teams).set_index("team")
    pre = preseason if preseason is not None else preseason_ratings(state)
    pre_rank = {t: i for i, t in enumerate(
        sorted(pre, key=lambda t: -pre[t]), start=1)}
    rows = []
    for t in teams:
        rows.append({
            "team": t,
            "conference": state.engine.conference.get(t),
            "conference_short": conference_short(state.engine.conference.get(t)),
            "elo": round(float(state.engine.rating_for(t)), 1),
            "wins": int(rec.loc[t, "wins"]),
            "losses": int(rec.loc[t, "losses"]),
            "conf_wins": int(rec.loc[t, "conf_wins"]),
            "conf_losses": int(rec.loc[t, "conf_losses"]),
            "pts_diff": int(rec.loc[t, "pts_diff"]),
            "games": int(state.engine.games_played.get(t, 0)),
            "preseason_elo": round(pre[t], 1) if t in pre else None,
            "preseason_rank": pre_rank.get(t),
        })
    rows.sort(key=lambda r: -r["elo"])
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


def _bands(elos: list[float]) -> dict:
    """Avg / top-4 / median / bottom-4 Elo — a conference's average, its
    ceiling, its middle and its floor (the soccer page's league bands,
    with a median in place of mid-10 since conferences run 8-18 teams)."""
    s = sorted(elos, reverse=True)
    n = len(s)
    mid = (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2) if n else None
    return {
        "avgElo": round(sum(s) / n, 1) if n else None,
        "top4Elo": round(sum(s[:4]) / 4, 1) if n >= 4 else None,
        "medianElo": round(mid, 1) if mid is not None else None,
        "bottom4Elo": round(sum(s[-4:]) / 4, 1) if n >= 4 else None,
    }


def conferences_payload(ratings: list[dict], top_n: int = TOP_N) -> dict:
    """Per conference: member count, Elo bands (current and preseason),
    programs in the top N, best and worst team — the college analogue of
    the soccer league_rankings table."""
    out = {}
    by_conf: dict[str, list[dict]] = {}
    for r in ratings:
        if r["conference"]:
            by_conf.setdefault(r["conference"], []).append(r)
    for conf, members in by_conf.items():
        members = sorted(members, key=lambda m: -m["elo"])
        pre = [m["preseason_elo"] for m in members if m.get("preseason_elo") is not None]
        out[conf] = {
            "name": conf,
            "short": conference_short(conf),
            "teams": len(members),
            **_bands([m["elo"] for m in members]),
            "preseasonAvgElo": _bands(pre)["avgElo"] if pre else None,
            "topN": sum(1 for m in members if m["rank"] <= top_n),
            "bestTeam": members[0]["team"],
            "bestElo": members[0]["elo"],
            "worstTeam": members[-1]["team"],
            "worstElo": members[-1]["elo"],
            "independent": conf == INDEPENDENT,
        }
    return out


def elo_history_payload(state: DailyState, run_date: str) -> dict:
    """Every FBS program's current-season trajectory: pre-game Elo at each
    game date, opened with the post-regression preseason rating and closed
    with today's live rating — so the chart moves each day the job runs."""
    season = state.season
    sub = state.history[state.history["season"] == season].sort_values("date")
    series: dict[str, list] = {}
    for r in sub.itertuples():
        if not r.home_fcs:
            series.setdefault(r.home_team, []).append([r.date, round(float(r.elo_home_pre), 1)])
        if not r.away_fcs:
            series.setdefault(r.away_team, []).append([r.date, round(float(r.elo_away_pre), 1)])
    for t in state.fbs_teams():
        points = series.setdefault(t, [])
        current = round(float(state.engine.rating_for(t)), 1)
        if not points or points[-1][0] < run_date:
            points.append([run_date, current])
        else:
            points[-1] = [points[-1][0], current]
    return {"season": season, "teams": series}


def current_week(state: DailyState, run_date: str) -> int | None:
    g = state.games[(state.games["season"] == state.season)
                    & (state.games["season_type"] == "regular")]
    upcoming = g[~g["completed"].astype(bool) & (g["date"] >= run_date)]
    if upcoming.empty:
        return int(g["week"].max()) if len(g) else None
    return int(upcoming["week"].min())


def export(state: DailyState, run_date: str, slate: pd.DataFrame,
           futures: dict | None, ledger: dict, graded_today: pd.DataFrame,
           recent: pd.DataFrame) -> None:
    ratings = ratings_payload(state)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_date": run_date,
        "season": state.season,
        "week": current_week(state, run_date),
        "params": {
            "k": state.engine.k,
            "home_advantage": state.engine.home_advantage,
            "season_regression": state.engine.season_regression,
            "conf_weight": state.engine.conf_weight,
            "fcs_rating": state.engine.fcs_rating,
            "entry_rating": state.engine.entry_rating,
            "margin_cap": state.engine.margin_cap,
            "elo_per_point": round(state.score_params.elo_per_point, 1),
        },
        "top_n": TOP_N,
        "ratings": ratings,
        "conferences": conferences_payload(ratings),
        "slate": _records(slate),
        "graded_today": _records(graded_today),
        "graded_recent": _records(recent),
        "ledger": ledger,
        "futures": futures or {"season": state.season, "status": "no_games"},
        "elo_history": elo_history_payload(state, run_date),
    }
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    SITE_HISTORY.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=1,
                      default=_clean) + "\n"
    SITE_LATEST.write_text(text)
    (SITE_HISTORY / f"{run_date}.json").write_text(text)
