"""Grade each sportsbook's closing prices against actual results.

The odds snapshots store per-book moneylines (``Home DraftKings H2H Odds`` and
friends) for 8-10 US books. For every game a book priced, this takes the
book's *last pregame* snapshot as its closing line, strips the vig, and scores
the implied probability against what actually happened.

Two leaderboards come out of it:

- **sharpness** — Brier / log loss of the no-vig closing probability. Lower =
  the book's closing number was closer to the truth.
- **price** — average hold. Lower = the book takes less juice, so the same
  opinion costs a bettor less there.

Usage
    python3 -m NFL.inventory.book_performance --season 2025
    python3 -m NFL.inventory.book_performance --season 2025 --week 12
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from NFL.model.line_movement import NAME_TO_NFLVERSE, SNAPSHOT_GLOB

REPO_ROOT = Path(__file__).resolve().parents[2]
GAMES_PATH = REPO_ROOT / "data" / "schedules" / "nflverse_games.csv"

BOOK_COL_RE = re.compile(r"^(Home|Away) (.+) H2H Odds$")
DATE_CANDIDATES = ["Date of Game (ET)", "Date of Game"]
MIN_GAMES = 20  # don't rank a book off a handful of games


def _melt_books(path: str) -> pd.DataFrame | None:
    """One snapshot file -> long rows of (ts, game, book, home/away odds)."""
    try:
        header = pd.read_csv(path, nrows=0).columns
    except Exception:
        return None
    books = sorted({m.group(2) for c in header if (m := BOOK_COL_RE.match(c))})
    date_col = next((c for c in DATE_CANDIDATES if c in header), None)
    base = ["Timestamp Pulled", "Home Team", "Away Team"]
    if not books or date_col is None or not set(base).issubset(header):
        return None

    usecols = base + [date_col] + [
        f"{side} {b} H2H Odds" for b in books for side in ("Home", "Away")
        if f"{side} {b} H2H Odds" in set(header)
    ]
    if "League" in header:
        usecols.append("League")
    try:
        df = pd.read_csv(path, usecols=usecols)
    except Exception:
        return None
    if "League" in df.columns:
        df = df[df["League"] == "NFL"]

    frames = []
    for b in books:
        hc, ac = f"Home {b} H2H Odds", f"Away {b} H2H Odds"
        if hc not in df.columns or ac not in df.columns:
            continue
        sub = df[base + [date_col]].copy()
        sub["book"] = b
        sub["h2h_home"] = pd.to_numeric(df[hc], errors="coerce")
        sub["h2h_away"] = pd.to_numeric(df[ac], errors="coerce")
        frames.append(sub.dropna(subset=["h2h_home", "h2h_away"]))
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"Timestamp Pulled": "ts", "Home Team": "home_team",
                              "Away Team": "away_team", date_col: "game_date"})
    return out


def load_book_closes(pattern: str = SNAPSHOT_GLOB) -> pd.DataFrame:
    """Last pregame snapshot per (game, book) across every snapshot file."""
    frames = []
    for path in sorted(glob.glob(pattern)):
        if path.endswith("latest.csv"):
            continue
        d = _melt_books(path)
        if d is not None and len(d):
            frames.append(d)
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["ts", "game_date"])
    df = df[df["ts"] < df["game_date"]]  # pregame only — same leak as line_movement

    df["game_key"] = (df["home_team"].str.strip() + "|" + df["away_team"].str.strip()
                      + "|" + df["game_date"].dt.date.astype(str))
    df = df.sort_values("ts")
    close = df.groupby(["game_key", "book"], as_index=False).last()

    hp = np.where(close["h2h_home"] < 0,
                  -close["h2h_home"] / (-close["h2h_home"] + 100),
                  100 / (close["h2h_home"] + 100))
    ap = np.where(close["h2h_away"] < 0,
                  -close["h2h_away"] / (-close["h2h_away"] + 100),
                  100 / (close["h2h_away"] + 100))
    close["hold"] = hp + ap - 1.0
    close["novig_home"] = hp / (hp + ap)
    return close


def join_results(closes: pd.DataFrame) -> pd.DataFrame:
    g = pd.read_csv(GAMES_PATH)
    g["gameday"] = pd.to_datetime(g["gameday"]).dt.normalize()
    g = g.dropna(subset=["home_score"])
    g["home_won"] = (g["home_score"] > g["away_score"]).astype(int)

    c = closes.copy()
    c["home_nfl"] = c["home_team"].map(NAME_TO_NFLVERSE)
    c["away_nfl"] = c["away_team"].map(NAME_TO_NFLVERSE)
    c["gameday"] = c["game_date"].dt.normalize()

    keys_r = ["gameday", "home_team", "away_team"]
    keys_l = ["gameday", "home_nfl", "away_nfl"]
    gcols = g[keys_r + ["game_id", "season", "week", "home_won"]]

    exact = c.merge(gcols, left_on=keys_l, right_on=keys_r, how="inner",
                    suffixes=("", "_sched"))
    # Late kickoffs can roll the snapshot's game date past midnight.
    rest = c[~c.set_index(["game_key", "book"]).index.isin(
        exact.set_index(["game_key", "book"]).index)].copy()
    rest["gameday"] = rest["gameday"] - pd.Timedelta(days=1)
    lagged = rest.merge(gcols, left_on=keys_l, right_on=keys_r, how="inner",
                        suffixes=("", "_sched"))
    out = pd.concat([exact, lagged], ignore_index=True)
    # Ties grade as a home loss for the 'favourite correct' stat; they are rare
    # enough (one in 2025) not to warrant a third outcome.
    return out


def leaderboard(graded: pd.DataFrame, min_games: int = MIN_GAMES) -> pd.DataFrame:
    rows = []
    for book, d in graded.groupby("book"):
        if len(d) < min_games:
            continue
        p = d["novig_home"].clip(1e-6, 1 - 1e-6)
        y = d["home_won"]
        pick_right = ((p >= 0.5) == (y == 1))
        rows.append({
            "book": book,
            "games": len(d),
            "brier": round(float(np.mean((p - y) ** 2)), 4),
            "log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
            "fav_correct": round(float(pick_right.mean()), 4),
            "avg_hold": round(float(d["hold"].mean()), 4),
        })
    lb = pd.DataFrame(rows)
    if len(lb):
        lb = lb.sort_values("brier").reset_index(drop=True)
        lb.insert(0, "rank", lb.index + 1)
    return lb


def consensus_row(graded: pd.DataFrame) -> pd.DataFrame:
    """The average of all books per game — the number the repo already tracks."""
    per_game = graded.groupby("game_id").agg(
        novig_home=("novig_home", "mean"),
        home_won=("home_won", "first"),
        hold=("hold", "mean"),
    ).reset_index()
    p = per_game["novig_home"].clip(1e-6, 1 - 1e-6)
    y = per_game["home_won"]
    return pd.DataFrame([{
        "rank": "—", "book": "CONSENSUS (all books)", "games": len(per_game),
        "brier": round(float(np.mean((p - y) ** 2)), 4),
        "log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
        "fav_correct": round(float(((p >= 0.5) == (y == 1)).mean()), 4),
        "avg_hold": round(float(per_game["hold"].mean()), 4),
    }])


def book_performance(season: int | None = None, week: int | None = None,
                     min_games: int = MIN_GAMES) -> pd.DataFrame:
    closes = load_book_closes()
    if closes.empty:
        return pd.DataFrame()
    graded = join_results(closes)
    if season is not None:
        graded = graded[graded["season"] == season]
    if week is not None:
        graded = graded[graded["week"] == week]
        min_games = min(min_games, 5)
    if graded.empty:
        return pd.DataFrame()
    lb = leaderboard(graded, min_games)
    return pd.concat([lb, consensus_row(graded)], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    args = ap.parse_args()

    lb = book_performance(args.season, args.week)
    if lb.empty:
        raise SystemExit("no graded games for that season/week filter")
    scope = f"season {args.season}" if args.season else "all captured games"
    if args.week:
        scope += f", week {args.week}"
    print(f"Bookmaker closing-line leaderboard — {scope}")
    print("(brier/log_loss: lower = sharper closing prices; avg_hold: lower = cheaper)")
    print()
    print(lb.to_string(index=False))


if __name__ == "__main__":
    main()
