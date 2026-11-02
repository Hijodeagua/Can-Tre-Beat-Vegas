"""
Overlay final scores from secondary sources onto the openfootball results.

openfootball (`fetch_results.py`) is the fixture backbone of
`results.csv`: it publishes every league's full-season fixture list and,
for most of its history, every final. It stalled on 2026-09-20 for the
2026-27 season (no played match in any of the six leagues after that
date, upstream last touched 2026-09-22), which froze the Elo, the grades,
the forecasts and the emails at that point while every pipeline run
stayed green. Two secondary sources carry the scores it is missing:

- **Understat** (`understat_matches.csv`, refreshed by `fetch_xg` /
  `understat.py` every run): final scores for the five top flights, the
  same table the xG features already come from. It does not cover the
  second divisions.
- **football-data.co.uk** (`fd_results.csv`, refreshed by `fd_results.py`):
  the publisher the shots feed already uses, which also posts the
  Championship (E1). It is the only overlay for that league.

The merge is deterministic and never duplicates a fixture:

1. Sources are applied in a fixed precedence - openfootball (primary),
   then Understat, then football-data. A later source never overwrites a
   score an earlier one supplied; it only cross-checks it.
2. A source row is matched to the primary fixture on the exact
   (league, date, home, away) key. If the fixture is still unplayed, the
   score is filled in.
3. With no exact hit, the same (league, season, home, away) pairing is
   looked for within WINDOW_DAYS of the source date. openfootball lists
   provisional kick-off dates (a Sunday for the whole matchday) that the
   leagues later move by a day or two; the source's date is the one the
   match was actually played on, so an unplayed primary row is moved to
   it and filled. A pairing further away than the window is left alone -
   that is how a reverse fixture (home and away swapped by one upstream)
   stays out of the wrong slot.
4. A score that disagrees between a played primary row and a source is a
   conflict: it is logged and counted, the primary (openfootball) score
   is kept, and the row is listed in the report so it can be checked by
   hand. A source row with no primary pairing at all is never appended -
   it is listed as unmatched - because an unmapped club name is the
   likeliest cause and a duplicate fixture under a second spelling would
   split a club's Elo.

Every run writes `results_overlay_report.json` next to the CSV with the
counts, the conflicts and the unmatched rows, so the state of the overlay
is visible in the repo and in the weekly model review.

    python -m soccer.clubs.data.results_overlay            # apply to results.csv
    python -m soccer.clubs.data.results_overlay --dry-run  # report only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RESULTS_CSV = DATA_DIR / "results.csv"
UNDERSTAT_CSV = DATA_DIR / "understat_matches.csv"
FD_CSV = DATA_DIR / "fd_results.csv"
REPORT_JSON = DATA_DIR / "results_overlay_report.json"

COLUMNS = [
    "date", "season", "league", "home_team", "away_team",
    "home_score", "away_score",
]

# How far (in days) a secondary source's date may sit from the primary
# fixture date and still be the same match. Provisional openfootball dates
# drift by one or two days; a week is generous without reaching the
# reverse fixture, which is never closer than a few weeks.
WINDOW_DAYS = 7

# Precedence order and the leagues each source may fill.
TOP_FLIGHTS = ("epl", "la_liga", "bundesliga", "serie_a", "ligue_1")
SOURCE_LEAGUES: dict[str, tuple[str, ...]] = {
    "understat": TOP_FLIGHTS,
    "football-data": TOP_FLIGHTS + ("championship",),
}


def _score(v) -> str:
    """Normalise a score cell to the CSV's own spelling: '' or an int string."""
    if v is None or v == "":
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if f != f:          # NaN
        return ""
    return str(int(f))


def _days(a: str, b: str) -> int:
    return abs((date.fromisoformat(a) - date.fromisoformat(b)).days)


# --------------------------------------------------------------------------
# source readers
# --------------------------------------------------------------------------
def understat_rows(path: Path = UNDERSTAT_CSV) -> list[dict]:
    """Final scores from the processed Understat table."""
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            hs, as_ = _score(r.get("goals_home")), _score(r.get("goals_away"))
            if not hs or not as_:
                continue
            out.append({
                "league": r["league"], "season": r["season"], "date": r["date"][:10],
                "home_team": r["home_team"], "away_team": r["away_team"],
                "home_score": hs, "away_score": as_,
            })
    return out


def fd_rows(path: Path = FD_CSV) -> list[dict]:
    """Final scores from the committed football-data.co.uk table."""
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            hs, as_ = _score(r.get("home_score")), _score(r.get("away_score"))
            if not hs or not as_:
                continue
            out.append({
                "league": r["league"], "season": r["season"], "date": r["date"][:10],
                "home_team": r["home_team"], "away_team": r["away_team"],
                "home_score": hs, "away_score": as_,
            })
    return out


def default_sources() -> list[tuple[str, list[dict]]]:
    return [("understat", understat_rows()), ("football-data", fd_rows())]


# --------------------------------------------------------------------------
# merge
# --------------------------------------------------------------------------
def apply(rows: list[dict], sources: list[tuple[str, list[dict]]],
          window_days: int = WINDOW_DAYS, log=print) -> tuple[list[dict], dict]:
    """Return (rows with overlays applied, report). `rows` is mutated in
    place and also returned; its order is preserved."""
    for r in rows:
        r["home_score"] = _score(r.get("home_score"))
        r["away_score"] = _score(r.get("away_score"))

    exact: dict[tuple, int] = {}
    pairing: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        exact[(r["league"], r["date"], r["home_team"], r["away_team"])] = i
        pairing.setdefault((r["league"], r["season"], r["home_team"], r["away_team"]), []).append(i)

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_days": window_days,
        "precedence": ["openfootball"] + [name for name, _ in sources],
        "sources": {},
        "by_league": {},
        "conflicts": [],
        "unmatched": [],
    }

    def _league_stat(league: str) -> dict:
        return report["by_league"].setdefault(
            league, {"filled": 0, "redated": 0, "agree": 0, "conflicts": 0,
                     "unmatched": 0, "newest_filled": None})

    def _fill(i: int, src: dict, name: str, redate: bool) -> None:
        r = rows[i]
        if redate:
            old_key = (r["league"], r["date"], r["home_team"], r["away_team"])
            exact.pop(old_key, None)
            r["date"] = src["date"]
            exact[(r["league"], r["date"], r["home_team"], r["away_team"])] = i
        r["home_score"], r["away_score"] = src["home_score"], src["away_score"]
        ls = _league_stat(r["league"])
        ls["filled"] += 1
        if redate:
            ls["redated"] += 1
        if ls["newest_filled"] is None or r["date"] > ls["newest_filled"]:
            ls["newest_filled"] = r["date"]

    for name, src_rows in sources:
        allowed = set(SOURCE_LEAGUES.get(name, ()))
        stat = {"rows": 0, "filled": 0, "redated": 0, "agree": 0,
                "conflicts": 0, "unmatched": 0, "skipped_league": 0}
        report["sources"][name] = stat
        ordered = sorted(src_rows, key=lambda s: (s["date"], s["league"], s["home_team"], s["away_team"]))
        for src in ordered:
            if src["league"] not in allowed:
                stat["skipped_league"] += 1
                continue
            stat["rows"] += 1
            key = (src["league"], src["date"], src["home_team"], src["away_team"])
            i = exact.get(key)
            how = "exact"
            if i is None:
                cands = pairing.get((src["league"], src["season"], src["home_team"], src["away_team"]), [])
                near = [(_days(rows[j]["date"], src["date"]), rows[j]["date"], j) for j in cands]
                near = [c for c in near if c[0] <= window_days]
                if near:
                    near.sort()
                    i = near[0][2]
                    how = "window"
            if i is None:
                # Reverse pairing inside the window = one upstream has home
                # and away swapped. Report it as a conflict (nothing is
                # changed); anything else is an unmatched row.
                rev = pairing.get((src["league"], src["season"], src["away_team"], src["home_team"]), [])
                swapped = [j for j in rev if _days(rows[j]["date"], src["date"]) <= window_days]
                entry = {"source": name, "league": src["league"], "season": src["season"],
                         "date": src["date"], "home_team": src["home_team"],
                         "away_team": src["away_team"],
                         "source_score": [int(src["home_score"]), int(src["away_score"])]}
                ls = _league_stat(src["league"])
                if swapped:
                    j = swapped[0]
                    entry.update({"kind": "home_away_swapped",
                                  "primary_date": rows[j]["date"],
                                  "primary_score": [_int_or_none(rows[j]["home_score"]),
                                                    _int_or_none(rows[j]["away_score"])]})
                    report["conflicts"].append(entry)
                    stat["conflicts"] += 1
                    ls["conflicts"] += 1
                    log(f"  ! overlay[{name}] {src['league']} {src['date']} "
                        f"{src['home_team']} v {src['away_team']}: openfootball has the "
                        f"reverse fixture on {rows[j]['date']} - home/away disagree, left as is")
                else:
                    entry["kind"] = "no_fixture"
                    report["unmatched"].append(entry)
                    stat["unmatched"] += 1
                    ls["unmatched"] += 1
                continue

            r = rows[i]
            if r["home_score"] == "" or r["away_score"] == "":
                _fill(i, src, name, redate=(how == "window"))
                stat["filled"] += 1
                if how == "window":
                    stat["redated"] += 1
            elif (r["home_score"], r["away_score"]) == (src["home_score"], src["away_score"]):
                stat["agree"] += 1
                _league_stat(r["league"])["agree"] += 1
            else:
                stat["conflicts"] += 1
                _league_stat(r["league"])["conflicts"] += 1
                report["conflicts"].append({
                    "source": name, "kind": "score_mismatch", "league": r["league"],
                    "season": r["season"], "date": r["date"], "home_team": r["home_team"],
                    "away_team": r["away_team"],
                    "primary_score": [int(r["home_score"]), int(r["away_score"])],
                    "source_score": [int(src["home_score"]), int(src["away_score"])],
                    "source_date": src["date"],
                })
                log(f"  ! overlay[{name}] {r['league']} {r['date']} {r['home_team']} v "
                    f"{r['away_team']}: openfootball {r['home_score']}-{r['away_score']} vs "
                    f"{name} {src['home_score']}-{src['away_score']} - keeping openfootball")
        log(f"  + overlay[{name}]: {stat['rows']} rows, filled {stat['filled']} "
            f"({stat['redated']} re-dated), {stat['agree']} agree, "
            f"{stat['conflicts']} conflicts, {stat['unmatched']} unmatched")
    return rows, report


def _int_or_none(v: str):
    return int(v) if v not in ("", None) else None


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------
def read_results(path: Path = RESULTS_CSV) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_results(rows: list[dict], path: Path = RESULTS_CSV) -> None:
    rows = sorted(rows, key=lambda r: (r["date"], r["league"], r["home_team"]))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_report(report: dict, path: Path = REPORT_JSON) -> None:
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report only; do not rewrite results.csv")
    args = ap.parse_args(argv)
    if not RESULTS_CSV.exists():
        print("results_overlay: no results.csv to overlay")
        return 1
    rows = read_results()
    played_before = sum(1 for r in rows if _score(r["home_score"]) != "")
    print("Overlaying secondary-source finals onto results.csv…")
    rows, report = apply(rows, default_sources())
    played_after = sum(1 for r in rows if r["home_score"] != "")
    report["played_before"], report["played_after"] = played_before, played_after
    if args.dry_run:
        print(f"dry run: {played_after - played_before} finals would be filled")
        return 0
    write_results(rows)
    write_report(report)
    print(f"Wrote {len(rows)} rows ({played_after} played, {played_after - played_before} newly "
          f"filled) to {RESULTS_CSV}; report at {REPORT_JSON.name}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
