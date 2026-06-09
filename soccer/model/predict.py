"""
Predict upcoming international fixtures (the 2026 World Cup schedule ships
inside results.csv with empty scores).

Replays the full Elo history through today, then scores every unplayed
fixture with the trained outcome model.

Usage:
    python -m soccer.model.predict [--days 60]
"""

import argparse
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from soccer.model.elo import HOME_ADVANTAGE, load_results, run_history
from soccer.model.squad import attach_squad_features
from soccer.model.train import FEATURES, KNOCKOUT_TOURNAMENTS

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def upcoming_fixtures(days: int) -> pd.DataFrame:
    df = load_results()
    fixtures = df[df["home_score"].isna()]
    today = datetime.now().strftime("%Y-%m-%d")
    horizon = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
    return fixtures[(fixtures["date"] >= today) & (fixtures["date"] <= horizon)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    args = parser.parse_args()

    with open(ARTIFACTS / "outcome_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    engine, _ = run_history()
    fixtures = upcoming_fixtures(args.days)
    if fixtures.empty:
        print(f"No fixtures in the next {args.days} days.")
        return

    rows = []
    for _, row in fixtures.iterrows():
        adv = 0.0 if bool(row["neutral"]) else HOME_ADVANTAGE
        rows.append(
            {
                "date": row["date"],
                "tournament": row["tournament"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "city": row["city"],
                "country": row["country"],
                "elo_home": round(engine.get(row["home_team"]), 1),
                "elo_away": round(engine.get(row["away_team"]), 1),
                "elo_gap": (engine.get(row["home_team"]) + adv) - engine.get(row["away_team"]),
                "host_home": int(row["country"] == row["home_team"]),
                "host_away": int(row["country"] == row["away_team"]),
                "knockout": int(row["tournament"] in KNOCKOUT_TOURNAMENTS),
            }
        )
    table = attach_squad_features(pd.DataFrame(rows))

    probs = model.predict_proba(table[FEATURES])
    classes = list(model.classes_)  # ['A', 'D', 'H']
    table["p_home"] = probs[:, classes.index("H")].round(4)
    table["p_draw"] = probs[:, classes.index("D")].round(4)
    table["p_away"] = probs[:, classes.index("A")].round(4)

    out_cols = [
        "date", "tournament", "home_team", "away_team", "city", "country",
        "elo_home", "elo_away", "p_home", "p_draw", "p_away",
    ]
    out = table[out_cols].sort_values("date")
    ARTIFACTS.mkdir(exist_ok=True)
    out_path = ARTIFACTS / "upcoming_predictions.csv"
    out.to_csv(out_path, index=False)

    print(f"{len(out)} fixtures in the next {args.days} days -> {out_path}\n")
    with pd.option_context("display.width", 160):
        print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
