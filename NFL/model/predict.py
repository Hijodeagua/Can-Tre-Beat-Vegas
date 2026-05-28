"""Inference: turn the latest odds slate into model picks.

Loads the trained LightGBM artifacts (lgbm_win.txt, lgbm_ats.txt), reads the
upcoming-games odds snapshot (data/odds_api_data_latest.csv), and builds a
feature row per game by reusing the training feature logic from features.py —
but for *future* games, so there are no outcome columns.

Each team's "form" going into its next game is the mean of its last N games of
box-score stats (the same ROLLING_STAT_COLS / window used in training). Schedule
context (week, rest, roof, division) is pulled from the cached nflverse
schedule when the game is found there, otherwise sensible defaults are used.

Output: picks/picks_today.json. If there are no upcoming games, it still writes
a valid file with an empty picks list.

Usage:
    python NFL/model/predict.py
    python NFL/model/predict.py --odds data/odds_api_data_latest.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

# features.py / schedule.py use same-directory imports, so add this dir to path.
MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIR))

from features import ROLLING_STAT_COLS, ROLLING_WINDOW, ROOF_CATEGORIES, load_raw  # noqa: E402
from schedule import load_schedule  # noqa: E402

REPO_ROOT = MODEL_DIR.parents[1]
ARTIFACTS = MODEL_DIR / "artifacts"
DATA_PATH = REPO_ROOT / "data" / "2023-2025W3.csv"
DEFAULT_ODDS = REPO_ROOT / "data" / "odds_api_data_latest.csv"
PICKS_PATH = REPO_ROOT / "picks" / "picks_today.json"

# Full odds-feed team name -> pro-football-reference abbrev (used in stats file).
FULL_TO_PFR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GNB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KAN", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LAR",
    "Las Vegas Raiders": "LVR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New Orleans Saints": "NOR", "New England Patriots": "NWE", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA", "San Francisco 49ers": "SFO", "Tampa Bay Buccaneers": "TAM",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def _team_form(raw: pd.DataFrame) -> dict[str, dict]:
    """Latest rolling form per team: mean of its last N games of box-score stats.

    Unlike the shifted training features, this *includes* the most recent game,
    which is the correct form going into the next (future) game.
    """
    raw = raw.sort_values(["Team", "Date"])
    form: dict[str, dict] = {}
    for team, g in raw.groupby("Team"):
        recent = g.tail(ROLLING_WINDOW)
        roll = {f"roll_{c}": float(recent[c].mean()) for c in ROLLING_STAT_COLS}
        form[team] = {"roll": roll, "games_played": int(len(g))}
    return form


def _load_schedule_index() -> pd.DataFrame | None:
    """Cached nflverse schedule keyed for (home, away, date) lookup. None if unavailable."""
    try:
        sched = load_schedule()
    except Exception as exc:  # pragma: no cover - network/cache miss
        print(f"  [warn] schedule unavailable ({exc}); using defaults for context")
        return None
    sched["date"] = pd.to_datetime(sched["gameday"]).dt.date
    return sched


def _schedule_row(sched: pd.DataFrame | None, home: str, away: str, gdate) -> pd.Series | None:
    if sched is None:
        return None
    hit = sched[(sched["home_team"] == home) & (sched["away_team"] == away) & (sched["date"] == gdate)]
    if hit.empty:
        return None
    return hit.iloc[0]


def _feature_row(
    home: str,
    away: str,
    home_form: dict,
    away_form: dict,
    spread_home: float,
    total: float,
    game_dt: datetime,
    div_game: int,
    sched_row: pd.Series | None,
    feature_cols: list[str],
) -> dict:
    """Assemble one home-perspective feature row matching feature_cols."""
    row: dict[str, float] = {}

    # Own + opponent rolling form.
    row.update(home_form["roll"])
    for k, v in away_form["roll"].items():
        row[f"opp_{k}"] = v

    # Pre-game / line features.
    row["Spread_vg"] = spread_home
    row["Over/Under_vg"] = total
    row["total_line"] = total
    row["games_played"] = home_form["games_played"]
    row["opp_games_played"] = away_form["games_played"]
    row["is_home"] = 1
    row["div_game"] = div_game
    row["day_num"] = game_dt.weekday()  # Mon=0 .. Sun=6, matches training map

    # Schedule context: from nflverse when matched, else defaults.
    if sched_row is not None:
        week = sched_row.get("week")
        rest = sched_row.get("home_rest")
        opp_rest = sched_row.get("away_rest")
        roof = str(sched_row.get("roof") or "outdoors")
        temp = sched_row.get("temp")
        wind = sched_row.get("wind")
        if div_game == 0 and not pd.isna(sched_row.get("div_game")):
            row["div_game"] = int(sched_row.get("div_game"))
    else:
        week, rest, opp_rest, roof, temp, wind = None, 7.0, 7.0, "outdoors", None, None

    row["Week"] = float(week) if week is not None and not pd.isna(week) else 1.0
    row["rest_days"] = float(rest) if rest is not None and not pd.isna(rest) else 7.0
    row["opp_rest_days"] = float(opp_rest) if opp_rest is not None and not pd.isna(opp_rest) else 7.0
    row["rest_diff"] = row["rest_days"] - row["opp_rest_days"]

    # QB change unknown for future games.
    row["qb_change"] = 0
    row["opp_qb_change"] = 0

    # Roof one-hot + dome/closed flag, mirroring features.build_feature_frame.
    roof = roof if roof in ROOF_CATEGORIES else "outdoors"
    for cat in ROOF_CATEGORIES:
        row[f"roof_{cat}"] = 1 if roof == cat else 0
    row["is_dome_or_closed"] = 1 if roof in ("dome", "closed") else 0
    is_indoor = row["is_dome_or_closed"] == 1
    row["temp"] = float(temp) if temp is not None and not pd.isna(temp) else (70.0 if is_indoor else 60.0)
    row["wind"] = float(wind) if wind is not None and not pd.isna(wind) else (0.0 if is_indoor else 8.0)

    # Return only the model's expected columns, in order; fill any gap with NaN
    # (LightGBM handles NaN natively).
    return {c: row.get(c, np.nan) for c in feature_cols}


def _lean(ats_prob: float) -> str:
    """Model's read against the spread. cover=back home, fade=back away, push=too close."""
    if ats_prob >= 0.55:
        return "cover"
    if ats_prob <= 0.45:
        return "fade"
    return "push"


def predict(odds_path: Path) -> dict:
    win_model = lgb.Booster(model_file=str(ARTIFACTS / "lgbm_win.txt"))
    ats_model = lgb.Booster(model_file=str(ARTIFACTS / "lgbm_ats.txt"))
    feature_cols = json.loads((ARTIFACTS / "features_win.json").read_text())

    raw = load_raw(str(DATA_PATH))
    form = _team_form(raw)
    sched = _load_schedule_index()

    odds = pd.read_csv(odds_path)
    odds = odds[odds["League"] == "NFL"].copy()
    odds["game_dt"] = pd.to_datetime(odds["Date of Game (ET)"], errors="coerce")

    # Upcoming games only.
    now = pd.Timestamp.now()
    upcoming = odds[odds["game_dt"] >= now.normalize()].sort_values("game_dt")

    picks: list[dict] = []
    skipped: list[str] = []

    for _, g in upcoming.iterrows():
        home_full, away_full = g["Home Team"], g["Away Team"]
        home = FULL_TO_PFR.get(home_full)
        away = FULL_TO_PFR.get(away_full)
        if home not in form or away not in form:
            skipped.append(f"{away_full} @ {home_full} (no stats for {away if away not in form else home})")
            continue

        game_dt = g["game_dt"].to_pydatetime()
        div_game = 1 if str(g.get("Divisional Matchup", "")).strip().lower() == "yes" else 0
        sched_row = _schedule_row(sched, home, away, game_dt.date())

        feats = _feature_row(
            home, away, form[home], form[away],
            spread_home=float(g["Avg Home Spread Points"]),
            total=float(g["Avg Total Points"]),
            game_dt=game_dt, div_game=div_game,
            sched_row=sched_row, feature_cols=feature_cols,
        )
        X = pd.DataFrame([feats])[feature_cols]
        win_prob = float(win_model.predict(X)[0])
        ats_prob = float(ats_model.predict(X)[0])

        picks.append({
            "game_id": str(g["Game ID"]),
            "home_team": home_full,
            "away_team": away_full,
            "game_time": game_dt.isoformat(),
            "vegas_spread": round(float(g["Avg Home Spread Points"]), 2),
            "model_win_prob": round(win_prob, 4),
            "model_ats_prob": round(ats_prob, 4),
            "model_lean": _lean(ats_prob),
        })

    if skipped:
        print(f"  [warn] skipped {len(skipped)} games: " + "; ".join(skipped[:5]))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_games": len(picks),
        "picks": picks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model picks from the latest odds.")
    parser.add_argument("--odds", default=str(DEFAULT_ODDS), help="Path to odds CSV.")
    args = parser.parse_args()

    odds_path = Path(args.odds)
    if not odds_path.exists():
        print(f"Odds file not found: {odds_path} — writing empty picks.")
        payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "num_games": 0, "picks": []}
    else:
        payload = predict(odds_path)

    PICKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PICKS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {payload['num_games']} picks -> {PICKS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
