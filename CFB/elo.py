"""Base head-to-head Elo for college football.

Reuses the NFL engine (``NFL/model/v2/elo.py``) — same MOV-adjusted update,
same walk-forward discipline — with CFB constants from
``CFB/DATA_PULL_PLAN.md`` §8. None of these are fitted yet; they are the
plan's starting guesses, to be grid-searched once more than one season of
games is on disk.

What differs from the NFL run:

- **K = 32** — fewer games per season and more true rating movement.
- **HFA = 70 Elo** (~2.5 pts) — college home fields are worth more.
- **Season regression = 0.40** toward 1500 — roster churn. The plan's real
  recommendation is regression toward the *conference* mean, which needs the
  standings pull (§3.1); flat 1500 is the base-model placeholder.
- **Margin cap at 35** before the MOV multiplier — 60-point CFB blowouts
  carry no information a 35-point one doesn't.
- **Postseason** (bowls + conference championships) gets the playoff K
  multiplier, and bowls are treated as neutral-site (they are).

Usage
    python3 -m CFB.ingest                # build the normalised spine first
    python3 -m CFB.elo                   # ratings board to stdout + CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from NFL.model.v2.elo import EloEngine, expected_score

REPO_ROOT = Path(__file__).resolve().parents[1]
AGG_DIR = REPO_ROOT / "data" / "college_football" / "agg"

BASE_RATING = 1500.0
K_FACTOR = 32.0
POSTSEASON_K_MULT = 1.2
HFA_ELO = 70.0
SEASON_REGRESSION = 0.40
ELO_PER_POINT = 28.0  # plan's guess; fit against margins later
MARGIN_CAP = 35.0

POSTSEASON_TYPES = {"BOWL", "CCG"}


def make_engine() -> EloEngine:
    return EloEngine(k=K_FACTOR, hfa=HFA_ELO, regression=SEASON_REGRESSION)


def compute_elo(games: pd.DataFrame) -> pd.DataFrame:
    """Attach pregame Elo columns; update on played games, capped margins."""
    df = games.sort_values(["gameday", "home_team"]).reset_index(drop=True)
    engine = make_engine()

    home_elo, away_elo, probs = [], [], []
    for row in df.itertuples(index=False):
        neutral = str(row.location) != "Home"
        h, a, p = engine.pregame(int(row.season), row.home_team, row.away_team, neutral)
        home_elo.append(h)
        away_elo.append(a)
        probs.append(p)
        if pd.notna(row.home_score) and pd.notna(row.away_score):
            margin = float(row.home_score) - float(row.away_score)
            capped = float(np.clip(margin, -MARGIN_CAP, MARGIN_CAP))
            # Feed the engine capped scores so its MOV multiplier sees the
            # capped margin; the win/loss outcome is unchanged by the cap.
            engine.update(
                row.home_team,
                row.away_team,
                capped if capped > 0 else 0.0,
                0.0 if capped > 0 else -capped,
                neutral=neutral,
                playoff=str(row.game_type) in POSTSEASON_TYPES,
            )

    df["home_elo"] = home_elo
    df["away_elo"] = away_elo
    df["elo_diff"] = (
        df["home_elo"] - df["away_elo"]
        + np.where(df["location"].eq("Home"), HFA_ELO, 0.0)
    )
    df["elo_home_prob"] = probs
    df["elo_spread"] = df["elo_diff"] / ELO_PER_POINT
    return df


def ratings_board(games: pd.DataFrame) -> pd.DataFrame:
    """Final ratings plus each team's season line, sorted."""
    engine = make_engine()
    played = games.dropna(subset=["home_score", "away_score"])
    for row in played.sort_values(["gameday", "home_team"]).itertuples(index=False):
        neutral = str(row.location) != "Home"
        engine.pregame(int(row.season), row.home_team, row.away_team, neutral)
        margin = float(row.home_score) - float(row.away_score)
        capped = float(np.clip(margin, -MARGIN_CAP, MARGIN_CAP))
        engine.update(
            row.home_team,
            row.away_team,
            capped if capped > 0 else 0.0,
            0.0 if capped > 0 else -capped,
            neutral=neutral,
            playoff=str(row.game_type) in POSTSEASON_TYPES,
        )

    recs: dict[str, list[int]] = {}
    for row in played.itertuples(index=False):
        hw = row.home_score > row.away_score
        recs.setdefault(row.home_team, [0, 0])[0 if hw else 1] += 1
        recs.setdefault(row.away_team, [0, 0])[1 if hw else 0] += 1

    board = pd.DataFrame(
        {
            "team": list(engine.ratings),
            "elo": [round(r, 1) for r in engine.ratings.values()],
        }
    )
    board["w"] = board["team"].map(lambda t: recs.get(t, [0, 0])[0])
    board["l"] = board["team"].map(lambda t: recs.get(t, [0, 0])[1])
    board["pts_vs_avg"] = ((board["elo"] - BASE_RATING) / ELO_PER_POINT).round(1)
    # Teams seen fewer than 4 times are FCS opponents passing through; they
    # have real ratings but tiny samples. Flag rather than hide.
    board["games"] = board["w"] + board["l"]
    board["small_sample"] = board["games"] < 4
    return board.sort_values("elo", ascending=False).reset_index(drop=True)


def evaluate(games: pd.DataFrame) -> dict[str, float]:
    """In-sample-honest quick check: log loss and accuracy of pregame probs."""
    df = compute_elo(games).dropna(subset=["home_score", "away_score"])
    y = (df["home_score"] > df["away_score"]).astype(float)
    p = df["elo_home_prob"].clip(1e-6, 1 - 1e-6)
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    acc = float(((p > 0.5) == y.astype(bool)).mean())
    return {"games": len(df), "log_loss": round(ll, 4), "accuracy": round(acc, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    games = pd.read_csv(AGG_DIR / "cfb_games.csv")
    board = ratings_board(games)
    board.to_csv(AGG_DIR / "cfb_elo_ratings.csv", index=False)

    fbs = board[~board["small_sample"]]
    print(fbs.head(args.top).to_string(index=False))
    print()
    print(evaluate(games))


if __name__ == "__main__":
    main()
