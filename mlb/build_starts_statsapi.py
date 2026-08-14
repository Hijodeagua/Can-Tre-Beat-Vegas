"""Per-start pitching lines for the current era from the MLB Stats API.

    python -m mlb.build_starts_statsapi [--start YYYY-MM-DD] [--end YYYY-MM-DD]
    python -m mlb.build_starts_statsapi --spot-check 10

Complements mlb/build_starts.py (retrosplits, 2010-2025): enumerates final
regular-season games in the window via the schedule endpoint, fetches each
game's boxscore, and extracts one row per (gamePk, team) with the starting
pitcher's line, appending/merging into data/mlb/pitcher_starts.csv with
source='statsapi' and game_id=gamePk.

Resumable by design: every fetched game is appended to
data/mlb/raw_statsapi/starts_checkpoint.jsonl as soon as its boxscore is
parsed, and already-checkpointed gamePks are skipped on the next run - a
crash mid-season resumes where it stopped rather than restarting. The final
CSV merge replaces statsapi rows for the window and never touches
retrosplits rows.

--spot-check N: instead of ingesting, sample N random existing rows from
pitcher_starts.csv (2008+, when statsapi boxscores begin), fetch the live
boxscore for each, and compare the stored line field-by-field. Non-zero exit
on any mismatch. This is the independent accuracy audit for the historical
ingest, run where statsapi is reachable (GitHub runner; the dev sandbox
blocks statsapi.mlb.com).

Default window: from the day after the last statsapi row on file (or
2026-03-01 when there is none) through yesterday US/Eastern.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from mlb.build_games import NAME_TO_BREF, franchise
from mlb.build_starts import FIELDS

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "mlb"
OUT = DATA / "pitcher_starts.csv"
XWALK = DATA / "person_id_crosswalk.csv"
CHECKPOINT = DATA / "raw_statsapi" / "starts_checkpoint.jsonl"

SCHEDULE_URL = ("https://statsapi.mlb.com/api/v1/schedule?sportId=1"
                "&gameType=R&startDate={start}&endDate={end}")
BOX_URL = "https://statsapi.mlb.com/api/v1/game/{pk}/boxscore"
REQUEST_GAP_S = 0.6  # stay well under statsapi's informal rate tolerance


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def mlbam_to_retro() -> dict[str, str]:
    with open(XWALK, newline="", encoding="utf-8") as fh:
        return {r["mlbam_id"]: r["retro_id"]
                for r in csv.DictReader(fh) if r["mlbam_id"]}


def final_games(start: str, end: str) -> list[dict]:
    """Final regular-season games in the window, with date/game_num/teams."""
    games = []
    for d in _get(SCHEDULE_URL.format(start=start, end=end)).get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("codedGameState") != "F":
                continue
            away = g["teams"]["away"]["team"]["name"]
            home = g["teams"]["home"]["team"]["name"]
            if away not in NAME_TO_BREF or home not in NAME_TO_BREF:
                continue
            games.append({
                "gamePk": g["gamePk"],
                "date": d["date"],
                "game_num": int(g.get("gameNumber", 1)),
                "away": franchise(NAME_TO_BREF[away]),
                "home": franchise(NAME_TO_BREF[home]),
            })
    return games


def parse_boxscore(box: dict, meta: dict, retro_map: dict) -> list[dict]:
    """Two rows (away, home) for one final game. The starter is the pitcher
    with gamesStarted == 1 in this game's own pitching line."""
    rows = []
    for side in ("away", "home"):
        team_box = box["teams"][side]
        starter = None
        for pid in team_box.get("pitchers", []):
            p = team_box["players"].get(f"ID{pid}")
            st = (p or {}).get("stats", {}).get("pitching", {})
            if st.get("gamesStarted") == 1:
                starter = (pid, p, st)
                break
        if starter is None:
            raise ValueError(f"gamePk {meta['gamePk']}: no starter for {side}")
        pid, p, st = starter
        team = meta[side]
        opponent = meta["home" if side == "away" else "away"]
        outs = st.get("outs")
        if outs is None:  # innings_pitched "6.1" -> 19 outs
            whole, _, frac = str(st["inningsPitched"]).partition(".")
            outs = int(whole) * 3 + int(frac or 0)
        rows.append({
            "game_id": str(meta["gamePk"]),
            "source": "statsapi",
            "date": meta["date"],
            "season": int(meta["date"][:4]),
            "game_num": meta["game_num"],
            "team": team,
            "opponent": opponent,
            "home": int(side == "home"),
            "retro_id": retro_map.get(str(pid), ""),
            "mlbam_id": str(pid),
            "name": p.get("person", {}).get("fullName", ""),
            "outs": int(outs),
            "h": int(st.get("hits", 0)),
            "r": int(st.get("runs", 0)),
            "er": int(st.get("earnedRuns", 0)),
            "bb": int(st.get("baseOnBalls", 0)),
            "so": int(st.get("strikeOuts", 0)),
        })
    return rows


def load_checkpoint() -> dict[str, list[dict]]:
    done: dict[str, list[dict]] = {}
    if CHECKPOINT.exists():
        with open(CHECKPOINT, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                done[str(rec["gamePk"])] = rec["rows"]
    return done


def ingest(start: str | None, end: str | None) -> int:
    existing: list[dict] = []
    last_statsapi = None
    if OUT.exists():
        with open(OUT, newline="", encoding="utf-8") as fh:
            existing = list(csv.DictReader(fh))
        statsapi_dates = [r["date"] for r in existing if r["source"] == "statsapi"]
        last_statsapi = max(statsapi_dates) if statsapi_dates else None

    if start is None:
        start = ((date.fromisoformat(last_statsapi) + timedelta(days=1)).isoformat()
                 if last_statsapi else "2026-03-01")
    if end is None:
        end = (datetime.now(ZoneInfo("America/New_York")).date()
               - timedelta(days=1)).isoformat()
    if start > end:
        print(f"nothing to do: start {start} > end {end}")
        return 0

    retro_map = mlbam_to_retro()
    games = final_games(start, end)
    done = load_checkpoint()
    todo = [g for g in games if str(g["gamePk"]) not in done]
    print(f"{len(games)} final games {start}..{end}; "
          f"{len(done)} checkpointed, {len(todo)} to fetch")

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT, "a", encoding="utf-8") as ck:
        for i, g in enumerate(todo, 1):
            rows = parse_boxscore(_get(BOX_URL.format(pk=g["gamePk"])), g,
                                  retro_map)
            ck.write(json.dumps({"gamePk": g["gamePk"], "rows": rows}) + "\n")
            ck.flush()
            done[str(g["gamePk"])] = rows
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} fetched")
            time.sleep(REQUEST_GAP_S)

    # Merge: keep everything except statsapi rows for games we (re)fetched.
    fetched_ids = {str(g["gamePk"]) for g in games}
    kept = [r for r in existing
            if not (r["source"] == "statsapi" and r["game_id"] in fetched_ids)]
    new_rows = [row for g in games for row in done[str(g["gamePk"])]]
    merged = kept + [{k: str(v) for k, v in row.items()} for row in new_rows]
    merged.sort(key=lambda r: (r["date"], int(r["game_num"]), r["team"]))
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(merged)
    print(f"wrote {len(merged)} rows ({len(new_rows)} statsapi in window) -> {OUT}")
    return 0


def spot_check(n: int, seed: int | None = None) -> int:
    with open(OUT, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["season"]) >= 2008]
    rng = random.Random(seed)
    sample = rng.sample(rows, n)
    retro_map = mlbam_to_retro()
    failures = 0
    for r in sample:
        gs = final_games(r["date"], r["date"])
        match = [g for g in gs
                 if g["home"] == (r["team"] if int(r["home"]) else r["opponent"])
                 and (g["game_num"] == int(r["game_num"]) or int(r["game_num"]) == 0)]
        if not match:
            print(f"FAIL {r['date']} {r['team']}: game not found in statsapi")
            failures += 1
            continue
        g = match[0]
        live = parse_boxscore(_get(BOX_URL.format(pk=g["gamePk"])), g, retro_map)
        side = [x for x in live if x["team"] == r["team"]][0]
        diffs = {f: (r[f], str(side[f]))
                 for f in ("mlbam_id", "outs", "h", "r", "er", "bb", "so")
                 if str(r[f]) != str(side[f])}
        status = "FAIL " + str(diffs) if diffs else "ok"
        failures += bool(diffs)
        print(f"{r['date']} {r['team']} {r['name']}: {status}")
        time.sleep(REQUEST_GAP_S)
    print(f"spot check: {n - failures}/{n} matched")
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--spot-check", type=int, metavar="N")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args(argv)
    if args.spot_check:
        return spot_check(args.spot_check, args.seed)
    return ingest(args.start, args.end)


if __name__ == "__main__":
    sys.exit(main())
