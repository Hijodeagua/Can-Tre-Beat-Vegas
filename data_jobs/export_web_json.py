#!/usr/bin/env python3
"""
Export the upcoming-games slate to static JSON for the web front end.

Reads the latest odds snapshot per sport (data/odds_api_data_*.csv and
data/nba/nba_odds_api_data_*.csv), filters to games starting within the
next N hours (default 48), attaches per-bookmaker odds, opening-line
movement, and model predictions when available, and writes:

    web/public/data/slate.json
    web/public/data/meta.json

Usage:
    python -m data_jobs.export_web_json [--hours 48] [--output web/public/data]
"""

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz

REPO_ROOT = Path(__file__).resolve().parent.parent
ET = pytz.timezone("America/New_York")

SPORTS = {
    "nfl": {"name": "NFL", "glob_dir": "data", "pattern": "odds_api_data_*.csv"},
    "nba": {"name": "NBA", "glob_dir": "data/nba", "pattern": "nba_odds_api_data_*.csv"},
    # World Cup ingestion not wired up yet — add here once fetch_odds supports it.
}

BOOK_COL_RE = re.compile(r"^(Home|Away) (.+?) (Spread|H2H|O/U) Odds$")
# Average columns are not per-book; exclude them from the bookmaker scan.
NON_BOOKS = {"Avg"}

LINE_HISTORY_COLS = [
    "Game ID",
    "Timestamp Pulled",
    "Avg Home Spread Points",
    "Avg Total Points",
    "Avg Home H2H Odds",
    "Avg Away H2H Odds",
]


def snapshot_files(sport: str) -> List[Path]:
    """Timestamped snapshot CSVs for a sport, oldest first."""
    cfg = SPORTS[sport]
    files = sorted((REPO_ROOT / cfg["glob_dir"]).glob(cfg["pattern"]))
    return [f for f in files if "_latest" not in f.name]


def safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def american_to_prob(odds: Optional[float]) -> Optional[float]:
    """Implied probability (with vig) of American odds."""
    if odds is None or odds == 0:
        return None
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def devig_home_prob(home_odds: Optional[float], away_odds: Optional[float]) -> Optional[float]:
    """No-vig home win probability from a two-way moneyline."""
    p_home = american_to_prob(home_odds)
    p_away = american_to_prob(away_odds)
    if p_home is None or p_away is None or p_home + p_away == 0:
        return None
    return p_home / (p_home + p_away)


def bookmaker_names(columns: List[str]) -> List[str]:
    books = []
    for col in columns:
        m = BOOK_COL_RE.match(col)
        if m and m.group(2) not in NON_BOOKS and m.group(2) not in books:
            books.append(m.group(2))
    return books


def book_odds(row: pd.Series, book: str) -> Optional[Dict[str, Any]]:
    def col(side: str, market: str) -> Optional[float]:
        return safe_float(row.get(f"{side} {book} {market} Odds"))

    entry = {
        "book": book,
        "home_ml": col("Home", "H2H"),
        "away_ml": col("Away", "H2H"),
        "home_spread_odds": col("Home", "Spread"),
        "away_spread_odds": col("Away", "Spread"),
        "over_odds": col("Home", "O/U"),
        "under_odds": col("Away", "O/U"),
    }
    if all(v is None for k, v in entry.items() if k != "book"):
        return None
    return entry


def opening_lines(sport: str, game_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Earliest snapshot consensus line for each game ID (the 'opener')."""
    missing = set(game_ids)
    openers: Dict[str, Dict[str, Any]] = {}
    for path in snapshot_files(sport):
        if not missing:
            break
        try:
            df = pd.read_csv(path, usecols=lambda c: c in LINE_HISTORY_COLS)
        except (ValueError, pd.errors.ParserError):
            continue
        if "Game ID" not in df.columns:
            continue
        hits = df[df["Game ID"].isin(missing)]
        for _, row in hits.iterrows():
            gid = row["Game ID"]
            openers[gid] = {
                "first_seen": str(row.get("Timestamp Pulled", "")),
                "spread": safe_float(row.get("Avg Home Spread Points")),
                "total": safe_float(row.get("Avg Total Points")),
                "home_ml": safe_float(row.get("Avg Home H2H Odds")),
                "away_ml": safe_float(row.get("Avg Away H2H Odds")),
            }
            missing.discard(gid)
    return openers


def load_predictions() -> pd.DataFrame:
    """All model prediction CSVs, latest prediction per (date, home, away)."""
    frames = []
    for path in sorted((REPO_ROOT / "data" / "predictions").glob("predictions_*.csv")):
        try:
            frames.append(pd.read_csv(path))
        except (pd.errors.ParserError, OSError):
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["Game Date"], errors="coerce").dt.date
    df["pred_date"] = pd.to_datetime(df["Prediction Date"], errors="coerce")
    df = df.sort_values("pred_date")
    return df.drop_duplicates(subset=["game_date", "Home Team", "Away Team"], keep="last")


def model_pick(
    preds: pd.DataFrame, game_date, home: str, away: str, market_home_prob: Optional[float]
) -> Optional[Dict[str, Any]]:
    if preds.empty:
        return None
    match = preds[
        (preds["game_date"] == game_date)
        & (preds["Home Team"] == home)
        & (preds["Away Team"] == away)
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    confidence = safe_float(row.get("Win Confidence %"))
    winner = row.get("Predicted Winner")
    home_prob = None
    if confidence is not None and winner in (home, away):
        home_prob = confidence / 100 if winner == home else 1 - confidence / 100
    edge = None
    if home_prob is not None and market_home_prob is not None:
        edge = round(home_prob - market_home_prob, 4)
    return {
        "predicted_winner": winner,
        "home_win_prob": home_prob,
        "pred_spread": safe_float(row.get("Pred Spread")),
        "pred_home_score": safe_float(row.get("Pred Home Score")),
        "pred_away_score": safe_float(row.get("Pred Away Score")),
        "edge_vs_market": edge,
        "model": row.get("Model (Clf)"),
        "predicted_at": str(row.get("Prediction Date", "")),
    }


def export_sport(sport: str, now_et: datetime, hours: int, preds: pd.DataFrame) -> Dict[str, Any]:
    cfg = SPORTS[sport]
    files = snapshot_files(sport)
    if not files:
        return {"key": sport, "name": cfg["name"], "snapshot": None, "games": []}
    latest = files[-1]
    df = pd.read_csv(latest)
    df["game_dt"] = df["Date of Game (ET)"].apply(
        lambda v: ET.localize(datetime.strptime(str(v), "%Y-%m-%d %H:%M"))
    )
    window = df[(df["game_dt"] >= now_et) & (df["game_dt"] < now_et + timedelta(hours=hours))]
    window = window.sort_values("game_dt")

    books = bookmaker_names(list(df.columns))
    openers = opening_lines(sport, list(window["Game ID"]))

    games = []
    for _, row in window.iterrows():
        home_ml = safe_float(row.get("Avg Home H2H Odds"))
        away_ml = safe_float(row.get("Avg Away H2H Odds"))
        market_home_prob = devig_home_prob(home_ml, away_ml)
        consensus = {
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": safe_float(row.get("Avg Home Spread Points")),
            "home_spread_odds": safe_float(row.get("Avg Home Spread Odds")),
            "away_spread_odds": safe_float(row.get("Avg Away Spread Odds")),
            "total": safe_float(row.get("Avg Total Points")),
            "over_odds": safe_float(row.get("Avg Over Odds")),
            "under_odds": safe_float(row.get("Avg Under Odds")),
            "home_win_prob": market_home_prob,
        }
        opener = openers.get(row["Game ID"])
        movement = None
        if opener:
            movement = dict(opener)
            if opener["spread"] is not None and consensus["home_spread"] is not None:
                movement["spread_delta"] = round(consensus["home_spread"] - opener["spread"], 2)
            if opener["total"] is not None and consensus["total"] is not None:
                movement["total_delta"] = round(consensus["total"] - opener["total"], 2)
        games.append(
            {
                "game_id": row["Game ID"],
                "commence_et": str(row["Date of Game (ET)"]),
                "home_team": row["Home Team"],
                "away_team": row["Away Team"],
                "consensus": consensus,
                "books": [b for b in (book_odds(row, name) for name in books) if b],
                "line_movement": movement,
                "model": model_pick(
                    preds, row["game_dt"].date(), row["Home Team"], row["Away Team"],
                    market_home_prob,
                ),
            }
        )

    pulled = str(df["Timestamp Pulled"].iloc[0]) if len(df) else None
    return {
        "key": sport,
        "name": cfg["name"],
        "snapshot": {"file": latest.name, "pulled_at_et": pulled},
        "games": games,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export upcoming-games slate JSON")
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--output", default="web/public/data")
    args = parser.parse_args()

    now_et = datetime.now(ET)
    preds = load_predictions()
    sports = [export_sport(sport, now_et, args.hours, preds) for sport in SPORTS]

    out_dir = REPO_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    slate = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_hours": args.hours,
        "sports": sports,
    }
    meta = {
        "last_updated": slate["generated_at"],
        "window_hours": args.hours,
        "game_counts": {s["key"]: len(s["games"]) for s in sports},
        "snapshots": {s["key"]: s["snapshot"] for s in sports},
        "label": "TRACKER — bookmaker odds and experimental model picks. Not betting advice.",
    }
    (out_dir / "slate.json").write_text(json.dumps(slate, indent=2))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    total = sum(len(s["games"]) for s in sports)
    print(f"Exported {total} games in the next {args.hours}h -> {out_dir}")
    for s in sports:
        print(f"  {s['name']}: {len(s['games'])} games (snapshot: {s['snapshot']})")


if __name__ == "__main__":
    main()
