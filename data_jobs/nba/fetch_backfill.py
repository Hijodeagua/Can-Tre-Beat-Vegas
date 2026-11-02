"""
One-off backfill of every free NBA input a model would need, plus two
probes for other sports. Runs from GitHub Actions (open egress); the dev
sandbox cannot reach these hosts.

Sources, all free:

- SportsDataverse release store (CC BY 4.0,
  github.com/sportsdataverse/sportsdataverse-data): ESPN and NBA Stats
  tables per season, the same publisher family the CFB pipeline already
  reads (`CFB/data/fetch_weekly.py`). Seasons are labelled by the year a
  season ends (2025 = 2024-25); every table is tried for every season in
  SEASONS and a 404 is recorded, not fatal.
- FiveThirtyEight NBA Elo history and RAPTOR player metrics
  (github.com/fivethirtyeight/data, CC BY 4.0).
- nflverse injury reports (CC BY 4.0), the availability input the NFL
  model does not have yet.
- FBref: two pages fetched only to record whether second-division xG and
  squad wage tables are still published after the January 2026 loss of
  its Opta data. Nothing from FBref is stored beyond that yes/no report.

Output:
- data/nba/raw/<table>/<file>.parquet|csv   every file fetched (gitignored;
  uploaded as a workflow artifact)
- data/nba/sdv/...                          the same files when small
  enough to commit (<= COMMIT_MAX_MB each)
- data/nba/backfill_manifest.json           per table: seasons found and
  missing, rows, columns, bytes, sample row, where the file lives
- data/nba/fbref_probe.json                 the FBref yes/no report

    python -m data_jobs.nba.fetch_backfill
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "nba"
RAW = OUT / "raw"
COMMIT_DIR = OUT / "sdv"
MANIFEST = OUT / "backfill_manifest.json"
FBREF_PROBE = OUT / "fbref_probe.json"
COMMIT_MAX_MB = 20

SDV = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download/"
FTE = "https://raw.githubusercontent.com/fivethirtyeight/"
SEASONS = list(range(2002, 2028))

# name -> release path template. Order is rough model value.
SDV_TABLES = {
    # ESPN side (2002 on): results spine, team and player box scores, who dressed.
    "espn_schedule": "espn_nba_schedules/nba_schedule_{s}.parquet",
    "espn_team_box": "espn_nba_team_boxscores/team_box_{s}.parquet",
    "espn_player_box": "espn_nba_player_boxscores/player_box_{s}.parquet",
    "espn_game_rosters": "espn_nba_game_rosters/game_rosters_{s}.parquet",
    "espn_standings": "espn_nba_standings/standings_{s}.parquet",
    "espn_team_season_stats": "espn_nba_team_season_stats/team_season_stats_{s}.parquet",
    "espn_player_season_stats": "espn_nba_player_season_stats/player_season_stats_{s}.parquet",
    # NBA Stats side: official box scores, game logs, lineups, injury reports.
    "stats_schedule": "nba_stats_schedules/nba_schedule_{s}.parquet",
    "stats_team_box": "nba_stats_team_boxscores/team_boxscores_{s}.parquet",
    "stats_player_box": "nba_stats_player_boxscores/player_boxscores_{s}.parquet",
    "stats_player_game_logs": "nba_stats_player_game_logs/player_game_logs_{s}.parquet",
    "stats_game_lineups": "nba_stats_game_lineups/nba_lineups_{s}.parquet",
    "stats_game_rosters": "nba_stats_game_rosters/game_rosters_{s}.parquet",
    "stats_injury_report": "nba_stats_injury_report/injury_report_{s}.parquet",
    "stats_player_season": "nba_stats_player_season_stats/player_season_stats_{s}.parquet",
    "stats_team_season": "nba_stats_team_season_stats/team_season_stats_{s}.parquet",
    # Player value and id crosswalks.
    "player_impact": "nba_player_impact/nba_player_impact_{s}.parquet",
    "player_crosswalk": "nba_crosswalk/nba_player_crosswalk_{s}.parquet",
    "team_crosswalk": "nba_crosswalk/nba_team_crosswalk_{s}.parquet",
    "schedule_crosswalk": "nba_crosswalk/nba_schedule_crosswalk_{s}.parquet",
}

SINGLE_FILES = {
    "fte_nba_elo_history": FTE + "data/master/nba-elo/nbaallelo.csv",
    "fte_raptor_by_team": FTE + "data/master/nba-raptor/modern_RAPTOR_by_team.csv",
    "fte_raptor_by_player": FTE + "data/master/nba-raptor/modern_RAPTOR_by_player.csv",
    "fte_raptor_historical": FTE + "data/master/nba-raptor/historical_RAPTOR_by_player.csv",
    "fte_player_advanced_historical": FTE + "nba-player-advanced-metrics/master/nba-data-historical.csv",
}

NFL_INJURY_SEASONS = list(range(2009, 2027))

UA = {"User-Agent": "Mozilla/5.0 (can-tre-beat-vegas backfill)"}


def _get(url: str, timeout: int = 120, tries: int = 3) -> requests.Response | None:
    for i in range(tries):
        try:
            r = requests.get(url, timeout=timeout, headers=UA)
        except requests.RequestException as exc:
            if i == tries - 1:
                print(f"    ! {url}: {exc}")
                return None
            time.sleep(2 ** (i + 1))
            continue
        if r.status_code == 404:
            return r
        if r.status_code >= 500 and i < tries - 1:
            time.sleep(2 ** (i + 1))
            continue
        return r
    return None


def _describe(path: Path) -> dict:
    try:
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
    except Exception as exc:  # noqa: BLE001 - report, don't die
        return {"error": f"unreadable: {exc}"}
    sample = df.head(1).to_dict(orient="records")
    return {
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "sample": json.loads(pd.Series(sample[0] if sample else {}).to_json(default_handler=str)) if sample else {},
    }


def _store(name: str, filename: str, content: bytes) -> dict:
    raw_path = RAW / name / filename
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(content)
    mb = len(content) / 1e6
    entry = {"file": filename, "bytes": len(content), **_describe(raw_path)}
    if mb <= COMMIT_MAX_MB:
        dest = COMMIT_DIR / name / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(raw_path, dest)
        entry["location"] = str(dest.relative_to(REPO))
    else:
        entry["location"] = "artifact only (too large to commit)"
    return entry


def fetch_sdv() -> dict:
    report = {}
    for name, tmpl in SDV_TABLES.items():
        found, missing, files = [], [], []
        for s in SEASONS:
            url = SDV + tmpl.format(s=s)
            r = _get(url)
            if r is None or r.status_code != 200:
                missing.append(s)
                continue
            files.append(_store(name, url.rsplit("/", 1)[1], r.content))
            found.append(s)
        rows = sum(f.get("rows", 0) for f in files)
        print(f"  + {name}: {len(found)} seasons ({found[0] if found else '-'}..{found[-1] if found else '-'}), "
              f"{rows:,} rows; missing {len(missing)}")
        report[name] = {"source": SDV + tmpl, "seasons_found": found, "seasons_missing": missing,
                        "rows": rows, "files": files}
    return report


def fetch_singles() -> dict:
    report = {}
    for name, url in SINGLE_FILES.items():
        r = _get(url)
        if r is None or r.status_code != 200:
            print(f"  ! {name}: HTTP {getattr(r, 'status_code', 'error')}")
            report[name] = {"source": url, "error": f"HTTP {getattr(r, 'status_code', 'error')}"}
            continue
        entry = _store(name, url.rsplit("/", 1)[1], r.content)
        print(f"  + {name}: {entry.get('rows', 0):,} rows")
        report[name] = {"source": url, "files": [entry], "rows": entry.get("rows", 0)}
    return report


def fetch_nfl_injuries() -> dict:
    found, missing, files = [], [], []
    for s in NFL_INJURY_SEASONS:
        url = NFLVERSE + f"injuries/injuries_{s}.parquet"
        r = _get(url)
        if r is None or r.status_code != 200:
            missing.append(s)
            continue
        files.append(_store("nfl_injuries", f"injuries_{s}.parquet", r.content))
        found.append(s)
    rows = sum(f.get("rows", 0) for f in files)
    print(f"  + nfl_injuries: {len(found)} seasons, {rows:,} rows; missing {missing}")
    return {"nfl_injuries": {"source": NFLVERSE + "injuries/injuries_{s}.parquet",
                             "seasons_found": found, "seasons_missing": missing,
                             "rows": rows, "files": files}}


def probe_fbref() -> dict:
    """Record, without storing any FBref data, whether the pages a soccer
    upgrade would need still exist: second-division xG and squad wages."""
    pages = {
        "championship_squad_stats": "https://fbref.com/en/comps/10/Championship-Stats",
        "championship_shooting": "https://fbref.com/en/comps/10/shooting/Championship-Stats",
        "championship_wages": "https://fbref.com/en/comps/10/wages/Championship-Wages",
        "premier_league_squad_stats": "https://fbref.com/en/comps/9/Premier-League-Stats",
    }
    out = {"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "pages": {}}
    for key, url in pages.items():
        r = _get(url, timeout=30, tries=2)
        text = r.text if r is not None and r.status_code == 200 else ""
        out["pages"][key] = {
            "url": url,
            "http": getattr(r, "status_code", None),
            "has_xg_column": ('data-stat="xg' in text) or ('data-stat="npxg' in text),
            "has_wages_column": 'data-stat="annual_wages' in text or 'data-stat="weekly_wages' in text,
            "mentions_opta": "Opta" in text,
        }
        print(f"  ? fbref {key}: HTTP {out['pages'][key]['http']}, xG {out['pages'][key]['has_xg_column']}, "
              f"wages {out['pages'][key]['has_wages_column']}")
        time.sleep(6)   # FBref asks for well under 20 requests a minute
    return out


def main() -> int:
    if COMMIT_DIR.exists():
        shutil.rmtree(COMMIT_DIR)
    RAW.mkdir(parents=True, exist_ok=True)
    print("SportsDataverse NBA tables…")
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "commit_max_mb": COMMIT_MAX_MB, "tables": {}}
    report["tables"].update(fetch_sdv())
    print("FiveThirtyEight NBA history…")
    report["tables"].update(fetch_singles())
    print("nflverse injury reports…")
    report["tables"].update(fetch_nfl_injuries())
    MANIFEST.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print("FBref availability probe…")
    FBREF_PROBE.write_text(json.dumps(probe_fbref(), indent=1) + "\n", encoding="utf-8")
    got = [k for k, v in report["tables"].items() if v.get("rows")]
    print(f"Done: {len(got)} of {len(report['tables'])} tables have data. Manifest: {MANIFEST.relative_to(REPO)}")
    return 0 if got else 1


if __name__ == "__main__":
    sys.exit(main())
