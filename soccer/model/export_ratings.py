"""
Export the custom international Elo ratings as a portable JSON snapshot.

This is the bridge that lets *other* projects (notably the World Cup Tickets
site) consume our ratings without re-implementing the Elo math: it replays the
full history with the existing `elo.py` engine and dumps the resulting table.

Nothing about the Elo rules lives here — `run_history()` / `EloEngine` in
`soccer/model/elo.py` remain the single source of truth. This module only
serializes their output.

Usage:
    python -m soccer.model.export_ratings [--min-matches 10] [--out PATH]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from soccer.model.elo import (
    BASE_RATING,
    HOME_ADVANTAGE,
    START_DATE,
    run_history,
)

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DEFAULT_OUT = ARTIFACTS / "elo_ratings.json"


def export(min_matches: int = 10, out: Path = DEFAULT_OUT) -> Path:
    engine, history = run_history()
    table = engine.table(min_matches=min_matches)

    ratings = [
        {
            "team": row["team"],
            "elo": round(float(row["elo"]), 1),
            "matches": int(row["matches"]),
        }
        for _, row in table.iterrows()
    ]

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Can-Tre-Beat-Vegas soccer Elo (soccer/model/elo.py)",
        "startDate": START_DATE,
        "baseRating": BASE_RATING,
        "homeAdvantage": HOME_ADVANTAGE,
        "minMatches": min_matches,
        "matchesProcessed": int(len(history)),
        "ratings": ratings,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-matches", type=int, default=10)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    path = export(args.min_matches, args.out)
    payload = json.loads(path.read_text())
    print(
        f"Wrote {len(payload['ratings'])} team ratings to {path} "
        f"({payload['matchesProcessed']} matches since {payload['startDate']})"
    )
    print("\nTop 10:")
    for r in payload["ratings"][:10]:
        print(f"  {r['elo']:>7.1f}  {r['team']}  ({r['matches']} matches)")


if __name__ == "__main__":
    main()
