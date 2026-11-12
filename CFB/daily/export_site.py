"""Write the site JSON: web/public/data/cfb/latest.json plus a per-day
history snapshot — the college-football sibling of the MLB and soccer
cards' data files, shaped like the soccer export (ratings / slate /
futures / ledger / elo_history under one roof with a generated_at stamp).

`ratings` is the full FBS board for the season (every program, ranked by
Elo, with season-to-date record); `conferences` is the cross-conference
summary (avg / top-4 / bottom-4 Elo per conference) that powers the
site's conference-strength table; `elo_history` is every program's
current-season Elo trajectory and `elo_projection` where the
rest-of-season sim expects each to go — the two halves of the trend
chart.
"""

from __future__ import annotations

import json
import math
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


def elo_projection_payload(futures: dict | None, run_date: str) -> dict:
    """Every program's *projected* Elo from the rest-of-season Monte Carlo:
    its live rating at each checkpoint date read as a median simulated
    season with the 10th/90th-percentile band around it, plus a
    few whole simulated seasons — all anchored on today's actual rating so
    the site's chart draws them as the continuation of the `elo_history`
    line rather than a floating second series.

    The headline value is a real simulated season, not an average of
    them: the average is flat by construction (see `simulate.py`), so a
    chart led by it would report that the board stops moving.

    The block is moved out of `futures` (where the sim put it) and up to
    the top level next to `elo_history`: it is chart data on the same
    footing, and the futures table itself never reads it. Empty when there
    is nothing left to simulate.
    """
    proj = (futures or {}).pop("projection", None)
    if not proj or not proj["dates"]:
        return {}
    now = {t["team"]: t["elo"] for t in futures.get("teams", [])}
    teams = {}
    samples = {}
    for team, median in proj["median"].items():
        elo = now.get(team)
        band = proj["band"][team]
        rows = [[d, m, *b] for d, m, b in zip(proj["dates"], median, band)]
        teams[team] = ([[run_date, elo, elo, elo]] if elo is not None else []) + rows
        # Each sample season opens on the same actual rating, so a path
        # lines up point-for-point with the median above it.
        head = [elo] if elo is not None else []
        samples[team] = [head + path for path in proj["samples"][team]]
    return {
        "season": futures.get("season"),
        "sims": futures.get("sims"),
        "from_date": run_date,
        "teams": teams,
        "samples": samples,
    }


def _game_scores(sub: pd.DataFrame) -> pd.DataFrame:
    """Per-game grading of the Elo replay's pre-game probability: was the
    favorite right, log-loss and Brier. Ties (none since OT) score 0.5."""
    p = sub["p_home"].clip(1e-6, 1 - 1e-6)
    y = sub["home_win"]
    out = pd.DataFrame(index=sub.index)
    out["correct"] = ((p >= 0.5) & (y == 1)) | ((p < 0.5) & (y == 0))
    out["ll"] = -(y * p.map(math.log) + (1 - y) * (1 - p).map(math.log))
    out["brier"] = (p - y) ** 2
    return out


def _game_row(r, fcs: bool = False) -> dict:
    row = {
        "game_id": int(r.game_id), "season": int(r.season), "week": int(r.week),
        "season_type": r.season_type, "date": r.date,
        "home_team": r.home_team, "away_team": r.away_team,
        "home_conference": r.home_conference if isinstance(r.home_conference, str) else "FCS",
        "away_conference": r.away_conference if isinstance(r.away_conference, str) else "FCS",
        "neutral": bool(r.neutral),
        "elo_home_pre": round(float(r.elo_home_pre), 1),
        "elo_away_pre": round(float(r.elo_away_pre), 1),
        "elo_diff": round(float(r.elo_diff), 1),
        "p_home": round(float(r.p_home), 4),
        "home_points": int(r.home_points), "away_points": int(r.away_points),
    }
    if fcs:
        row["home_fcs"] = bool(r.home_fcs)
        row["away_fcs"] = bool(r.away_fcs)
    return row


def interconference_payload(history: pd.DataFrame, season: int,
                            seasons_back: int = 1) -> dict:
    """Every FBS-vs-FBS game between two different conferences (the FBS
    independents count as one) for this season and the `seasons_back`
    before it, with the Elo replay's pre-game home probability and the
    result — the site's Interconference tab aggregates these per
    conference, per conference matchup and per team under its season
    filter. `fcs_games` is every FBS-vs-FCS game over the same seasons,
    kept apart so it is never pooled with the FBS-vs-FBS numbers.

    The probabilities are the replay's, i.e. what the engine thought the
    morning of each game with no hindsight (parameters were fit on
    2005-2023, so 2024+ is out of sample). `baseline` is the same
    grading over in-conference games, for the "is the model worse
    across conferences" comparison, as sums so the site can pool any
    combination of seasons and season types.
    """
    if history is None or history.empty:
        return {}
    seasons = list(range(season - seasons_back, season + 1))
    window = history[history["season"].isin(seasons)]
    home_fcs = window["home_fcs"].astype(bool)
    away_fcs = window["away_fcs"].astype(bool)
    sub = window[~home_fcs & ~away_fcs
                 & window["home_conference"].notna()
                 & window["away_conference"].notna()]
    scores = _game_scores(sub)
    inter = sub["home_conference"] != sub["away_conference"]
    games = [_game_row(r) for r in sub[inter].sort_values(["date", "game_id"]).itertuples()]
    # FBS vs FCS, kept in its own list so the site never pools it with
    # the FBS-vs-FBS numbers: every FCS opponent is one pooled rating, so
    # these are a different (and much easier) kind of game.
    fcs = window[home_fcs ^ away_fcs]
    fcs = fcs[fcs["home_conference"].where(~home_fcs, fcs["away_conference"]).notna()]
    fcs_games = [_game_row(r, fcs=True)
                 for r in fcs.sort_values(["date", "game_id"]).itertuples()]
    baseline = []
    conf_games = sub[~inter]
    for (yr, stype), g in conf_games.groupby(["season", "season_type"]):
        sc = scores.loc[g.index]
        baseline.append({
            "season": int(yr), "season_type": stype, "games": int(len(g)),
            "correct": int(sc["correct"].sum()),
            "ll_sum": round(float(sc["ll"].sum()), 4),
            "brier_sum": round(float(sc["brier"].sum()), 4),
        })
    return {"seasons": seasons, "games": games, "fcs_games": fcs_games,
            "conference_baseline": baseline}


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
    # Before the futures block is serialized — this pops its projection.
    elo_projection = elo_projection_payload(futures, run_date)
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
        # Which forecast the slate carries and how fresh its feed is.
        "second_stage": {
            "on": state.second_stage is not None,
            "model": (f"elo+efficiency:{state.second_stage.learner}"
                      if state.second_stage is not None else "elo"),
            "features": list(state.second_stage.features) if state.second_stage is not None else ["elo_logit"],
            "n_train": state.second_stage.n_train if state.second_stage is not None else None,
        },
        "feeds": {k: {"source": r.source, "newest": r.newest, "age_days": r.age_days,
                      "tolerance_days": r.tolerance_days, "rows": r.rows, "fresh": r.fresh}
                  for k, r in state.feeds.items()},
        "elo_history": elo_history_payload(state, run_date),
        "elo_projection": elo_projection,
        "interconference": interconference_payload(state.history, state.season),
    }
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    SITE_HISTORY.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=1,
                      default=_clean) + "\n"
    SITE_LATEST.write_text(text)
    (SITE_HISTORY / f"{run_date}.json").write_text(text)
