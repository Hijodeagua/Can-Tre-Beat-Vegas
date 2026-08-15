"""Per-start pitching lines 2010-2025 from the Chadwick retrosplits
day-by-day files.

    git clone --depth 1 --filter=blob:none --no-checkout \
        https://github.com/chadwickbureau/retrosplits
    (cd retrosplits && git sparse-checkout set --no-cone \
        'daybyday/playing-201*.csv' 'daybyday/playing-202*.csv' \
        && git checkout HEAD)
    python -m mlb.build_starts retrosplits/daybyday

Writes data/mlb/pitcher_starts.csv - one row per (game, team) with that
team's starting pitcher and his full line:

    game_id, source, date, season, game_num, team, opponent, home,
    retro_id, mlbam_id, name, outs, h, r, er, bb, so

- game_id is the Retrosheet game key (e.g. PHI202404010) for these rows;
  rows appended later by the statsapi ingest (2026-) use the MLBAM gamePk
  and source='statsapi'. game_num follows each source's own convention
  (Retrosheet: 0 = single game; statsapi: 1 = single game) - the same split
  the games CSV already carries, so joins against it line up per season.
- team/opponent are canonical franchise codes (FLA->MIA, OAK->ATH),
  matching the Elo engine.
- outs is P_OUT (thirds of an inning, so 6.1 IP = 19 outs); unearned runs
  for the game score are r - er.
- MLBAM ids come from data/mlb/person_id_crosswalk.csv
  (mlb/build_person_crosswalk.py); the ingest fails if any starter is
  missing one - the live pipeline matches probables by MLBAM id, so a gap
  here would silently break the fallback ladder later.

Validations (all fail loudly):
- per season, exactly one starter per (game, team), and the row count is
  within 1% of 2x the games CSV's count for that season;
- starter identity agrees with data/mlb/starting_pitchers_2009_2025.csv
  (independent extraction from the Retrosheet game logs) on >= 99.9% of
  team-games.

Data courtesy of Retrosheet (https://www.retrosheet.org) via the Chadwick
Bureau's retrosplits project.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from mlb.build_games import RS_TO_BREF, franchise

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "mlb"
OUT = DATA / "pitcher_starts.csv"
XWALK = DATA / "person_id_crosswalk.csv"
GAMES = DATA / "games_2009_2026.csv"
STARTERS_GL = DATA / "starting_pitchers_2009_2025.csv"

SEASONS = range(2010, 2026)

FIELDS = ["game_id", "source", "date", "season", "game_num", "team",
          "opponent", "home", "retro_id", "mlbam_id", "name",
          "outs", "h", "r", "er", "bb", "so"]


def load_crosswalk() -> dict[str, dict]:
    with open(XWALK, newline="", encoding="utf-8") as fh:
        return {r["retro_id"]: r for r in csv.DictReader(fh)}


def extract_season(path: Path, season: int, xwalk: dict) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["season.phase"] != "R" or r.get("P_GS") != "1":
                continue
            team = franchise(RS_TO_BREF[r["team.key"]])
            opp = franchise(RS_TO_BREF[r["opponent.key"]])
            person = xwalk.get(r["person.key"])
            rows.append({
                "game_id": r["game.key"],
                "source": "retrosplits",
                "date": r["game.date"],
                "season": season,
                "game_num": int(r["game.number"]),
                "team": team,
                "opponent": opp,
                "home": int(r["team.alignment"]),
                "retro_id": r["person.key"],
                "mlbam_id": person["mlbam_id"] if person else "",
                "name": (f"{person['name_first']} {person['name_last']}"
                         if person else ""),
                "outs": int(r["P_OUT"]),
                "h": int(r["P_H"]),
                "r": int(r["P_R"]),
                "er": int(r["P_ER"]),
                "bb": int(r["P_BB"]),
                "so": int(r["P_SO"]),
            })
    return rows


def count_games_per_season() -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    with open(GAMES, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            counts[int(r["season"])] += 1
    return counts


def validate(rows: list[dict]) -> None:
    # One starter per (game, team).
    seen: dict[tuple, list[str]] = defaultdict(list)
    for r in rows:
        seen[(r["game_id"], r["team"])].append(r["retro_id"])
    multi = {k: v for k, v in seen.items() if len(v) > 1}
    if multi:
        sample = list(multi.items())[:5]
        raise SystemExit(f"{len(multi)} team-games with >1 starter: {sample}")

    # Row count vs 2x games played, within 1% per season.
    game_counts = count_games_per_season()
    by_season: dict[int, int] = defaultdict(int)
    for r in rows:
        by_season[r["season"]] += 1
    for season in SEASONS:
        expect = 2 * game_counts[season]
        got = by_season[season]
        if abs(got - expect) > 0.01 * expect:
            raise SystemExit(
                f"{season}: {got} starter rows vs {expect} expected "
                f"(2 x {game_counts[season]} games) - outside 1%")
        print(f"  {season}: {got} starts / {expect} team-games")

    # No modern starter should lack an MLBAM id.
    missing = sorted({r["retro_id"] for r in rows if not r["mlbam_id"]})
    if missing:
        raise SystemExit(
            f"{len(missing)} starters missing MLBAM ids: {missing[:10]}")

    # Identity cross-check against the game-log extraction (independent
    # Retrosheet file + parser). Keyed by (date, game_num, home team).
    gl: dict[tuple, dict[str, str]] = {}
    with open(STARTERS_GL, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["date"], int(r["game_num"]), r["home_fr"])
            gl[key] = {"home": r["home_sp_retro"], "away": r["away_sp_retro"]}
    checked = agree = 0
    for r in rows:
        home_team = r["team"] if r["home"] else r["opponent"]
        rec = gl.get((r["date"], r["game_num"], home_team))
        if rec is None:
            continue
        checked += 1
        agree += rec["home" if r["home"] else "away"] == r["retro_id"]
    rate = agree / max(checked, 1)
    print(f"  game-log identity agreement: {agree}/{checked} ({rate:.4%})")
    if rate < 0.999:
        raise SystemExit("starter identity disagrees with the game logs")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    daybyday = Path(argv[0])

    xwalk = load_crosswalk()
    rows: list[dict] = []
    for season in SEASONS:
        path = daybyday / f"playing-{season}.csv"
        got = extract_season(path, season, xwalk)
        rows += got
    rows.sort(key=lambda r: (r["date"], r["game_num"], r["team"]))

    validate(rows)

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} starts -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
