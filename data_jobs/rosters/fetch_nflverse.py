"""Download the nflverse data the squad-quality features need.

Three release assets, all free and keyless:

- ``players/players.csv`` — the player master. Carries ``draft_year``,
  ``draft_round``, ``draft_pick`` and the id crosswalk (gsis / pfr / otc).
- ``weekly_rosters/roster_weekly_{season}.csv`` — who was actually on the
  active roster in a given week, 2002-present. This is what makes the
  features per-game rather than per-season.
- ``stats_player/stats_player_week_{season}.csv`` — weekly player stats,
  used for quarterback quality (passing EPA per attempt).

Raw files are large (weekly rosters run ~15 MB a season, ~360 MB for the full
history), so they land in ``data/rosters/nflverse/`` which is gitignored. Only
the small aggregated feature table built by ``NFL/model/v2/squad.py`` is
committed.

Usage
    python3 -m data_jobs.rosters.fetch_nflverse                  # 2002-present
    python3 -m data_jobs.rosters.fetch_nflverse --seasons 2025 2026
    python3 -m data_jobs.rosters.fetch_nflverse --force          # ignore cache
"""

from __future__ import annotations

import argparse
import shutil
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "rosters" / "nflverse"
BASE = "https://github.com/nflverse/nflverse-data/releases/download"

FIRST_SEASON = 2002
TIMEOUT = 120

# GitHub release downloads 302 to a CDN; urlretrieve follows redirects.
STATIC_ASSETS = {
    "players.csv": f"{BASE}/players/players.csv",
    "draft_picks.csv": f"{BASE}/draft_picks/draft_picks.csv",
}
PER_SEASON = {
    "roster_weekly_{season}.csv": f"{BASE}/weekly_rosters/roster_weekly_{{season}}.csv",
    "stats_player_week_{season}.csv": f"{BASE}/stats_player/stats_player_week_{{season}}.csv",
}


def _download(url: str, dest: Path, force: bool = False) -> bool:
    """Returns True if the file is on disk afterwards."""
    if dest.exists() and not force and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"  MISS {dest.name}: {exc}")
        return False
    shutil.move(tmp, dest)
    print(f"  got  {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def fetch(seasons: list[int], force: bool = False) -> dict[str, int]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"static": 0, "rosters": 0, "stats": 0, "missing": 0}

    print("static assets:")
    for name, url in STATIC_ASSETS.items():
        counts["static" if _download(url, CACHE_DIR / name, force) else "missing"] += 1

    print(f"per-season assets ({min(seasons)}-{max(seasons)}):")
    for season in seasons:
        for pattern, url_tpl in PER_SEASON.items():
            name = pattern.format(season=season)
            ok = _download(url_tpl.format(season=season), CACHE_DIR / name, force)
            key = "rosters" if "roster" in pattern else "stats"
            counts[key if ok else "missing"] += 1
    return counts


def latest_available_season() -> int:
    """Newest season with a schedule — rosters generally track it."""
    import pandas as pd
    games = REPO_ROOT / "data" / "schedules" / "nflverse_games.csv"
    if games.exists():
        return int(pd.read_csv(games, usecols=["season"])["season"].max())
    from datetime import date
    return date.today().year


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="*", default=None)
    ap.add_argument("--first-season", type=int, default=FIRST_SEASON)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seasons = args.seasons or list(range(args.first_season, latest_available_season() + 1))
    counts = fetch(seasons, args.force)
    print(f"\nrosters={counts['rosters']} stats={counts['stats']} "
          f"static={counts['static']} missing={counts['missing']}")
    print(f"cache -> {CACHE_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
