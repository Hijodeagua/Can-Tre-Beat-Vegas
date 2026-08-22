"""
Export the club Elo tables for all five leagues as one portable JSON snapshot.

Same bridge pattern as `soccer/model/export_ratings.py`: replay the full
history with `ClubEloEngine` and serialize the resulting tables — none of
the Elo rules live here. By default each league's table lists the clubs who
played in its most recent season on record (i.e. the current league
membership as far as the data knows); `--all-teams` includes every club
ever rated, with its regressed rating and last season.

Usage:
    python -m soccer.clubs.model.export_ratings [--all-teams] [--out PATH]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from soccer.clubs.data.leagues import LEAGUES
from soccer.clubs.model.elo import BASE_RATING, league_params
from soccer.clubs.model.europe import UEFA_WEIGHT, run_all_european

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DEFAULT_OUT = ARTIFACTS / "club_elo_ratings.json"


def export(all_teams: bool = False, out: Path = DEFAULT_OUT) -> Path:
    engines, history = run_all_european()
    history = history[~history["league"].str.startswith("uefa:")]

    leagues = {}
    for key, lg in LEAGUES.items():
        latest = str(history[history["league"] == key]["season"].max())
        table = engines[key].table(season=None if all_teams else latest)
        leagues[key] = {
            "name": lg.name,
            "firstSeason": lg.first_season,
            "latestSeason": latest,
            "params": league_params(key),
            "matchesProcessed": int((history["league"] == key).sum()),
            "ratings": [
                {
                    "team": row["team"],
                    "elo": round(float(row["elo"]), 1),
                    "matches": int(row["matches"]),
                    "lastSeason": row["last_season"],
                }
                for _, row in table.iterrows()
            ],
        }

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Can-Tre-Beat-Vegas club Elo (soccer/clubs/model/elo.py + europe.py)",
        "baseRating": BASE_RATING,
        "uefaGlueWeight": UEFA_WEIGHT,
        "matchesProcessed": int(len(history)),
        "leagues": leagues,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-teams", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    path = export(args.all_teams, args.out)
    payload = json.loads(path.read_text())
    print(f"Wrote {path} ({payload['matchesProcessed']} matches)")
    for key, lg in payload["leagues"].items():
        top = lg["ratings"][0]
        print(
            f"  {lg['name']:>15} ({lg['latestSeason']}): {len(lg['ratings'])} clubs — "
            f"top {top['team']} @ {top['elo']}"
        )


if __name__ == "__main__":
    main()
