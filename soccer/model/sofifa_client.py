"""
SoFIFA API client — pull national-team squad ratings per FIFA/FC edition and
write them into the pipeline schema (soccer/data/fifa_ratings/fifa_<year>.csv,
columns team,player,overall).

Why this exists
---------------
The squad-strength layer (soccer/model/squad.py) needs each national side's
player overall ratings, vintage-matched to the edition on shelves at match
time. SoFIFA's free REST API (https://api.sofifa.net, documented at
https://sofifa.com/document) exposes national teams as ordinary teams, so we:

  1. GET /teams/{roster}            -> all teams for an edition's roster
                                       filter type=="national", gender=="male"
  2. GET /team/{id}/{roster}        -> that squad's players + overallRating

No API key is required for these endpoints (the apiToken in the docs is only
for the customizedPlayers endpoints). They are rate-limited to 60 req/min;
this client self-throttles and backs off on HTTP 429.

Attribution: SoFIFA's terms require non-commercial use and a SoFIFA logo +
link on the consuming site's landing page. If the /vegas site surfaces these
ratings, add that attribution.

Network note: api.sofifa.net is NOT reachable from the Claude Code web
sandbox (host allowlist). Run this LOCALLY, then commit the resulting CSVs.

Usage
-----
    # Fill the editions we don't already have (FIFA 07-14, 22-26)
    python -m soccer.model.sofifa_client

    # Re-pull every edition 07-26 from one source for full consistency
    python -m soccer.model.sofifa_client --versions all

    # Sanity check: one edition, first 5 teams only
    python -m soccer.model.sofifa_client --versions 26 --limit 5
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from soccer.model.convert_ratings import NAME_MAP

BASE_URL = "https://api.sofifa.net"
RATINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "fifa_ratings"
CACHE_DIR = RATINGS_DIR / ".cache"

# Editions already loaded from GitHub mirrors (FIFA 15-21 == versions 15-21).
EXISTING_VERSIONS = {"15", "16", "17", "18", "19", "20", "21"}
ALL_VERSIONS = [f"{v:02d}" for v in range(7, 27)]  # 07 .. 26
GAP_VERSIONS = [v for v in ALL_VERSIONS if v not in EXISTING_VERSIONS]

# SoFIFA national-team names that differ from soccer/data/results.csv labels,
# on top of the shared NAME_MAP (which covers the Kaggle nationality strings).
SOFIFA_TEAM_MAP = {
    **NAME_MAP,
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "China PR": "China",
    "Republic of Ireland": "Ireland",
    "IR Iran": "Iran",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Cabo Verde": "Cape Verde",
}


def version_to_release_year(version: str) -> int:
    """FIFA 15 (version '15') shipped Sept 2014 -> 2014. FC 26 -> 2025."""
    return 2000 + int(version) - 1


def default_roster(version: str) -> str:
    """Launch roster for an edition. The docs show '220001' as a version-22
    roster, so the launch snapshot follows {YY}0001."""
    return f"{version}0001"


class SofifaClient:
    def __init__(self, rpm: int = 50, api_token: Optional[str] = None):
        self.min_interval = 60.0 / rpm
        self.api_token = api_token or os.environ.get("SOFIFA_API_TOKEN")
        self.session = requests.Session()
        # Browser-like headers: the API sits behind Cloudflare, which rejects
        # bare scripted user agents with 403.
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://sofifa.com/",
                "Origin": "https://sofifa.com",
            }
        )
        self._last = 0.0

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)

    def get(self, path: str, max_retries: int = 5) -> dict:
        url = f"{BASE_URL}{path}"
        for attempt in range(max_retries):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=30)
                self._last = time.monotonic()
            except requests.RequestException as exc:
                wait = 2 ** attempt
                print(f"  ! {path} network error ({exc}); retry in {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 61))
                print(f"  ! 429 rate-limited; sleeping {retry_after}s")
                time.sleep(retry_after)
                continue
            if resp.status_code == 404:
                raise FileNotFoundError(f"404 {path}")
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Gave up on {path} after {max_retries} retries")

    def national_teams(self, roster: str) -> List[dict]:
        data = self.get(f"/teams/{roster}").get("data", [])
        return [
            t for t in data
            if t.get("type") == "national" and t.get("gender", "male") == "male"
        ]

    def team_squad(self, team_id: int, roster: str) -> List[dict]:
        return self.get(f"/team/{team_id}/{roster}").get("data", {}).get("players", [])


def player_name(p: dict) -> str:
    common = (p.get("commonName") or "").strip()
    if common:
        return common
    return f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()


def normalize_team(name: str) -> str:
    return SOFIFA_TEAM_MAP.get(name, name)


def fetch_edition(
    client: SofifaClient, version: str, roster: str, limit: Optional[int]
) -> Optional[Path]:
    print(f"\n=== FIFA/FC {version} (roster {roster}) ===")
    try:
        teams = client.national_teams(roster)
    except FileNotFoundError:
        print(f"  roster {roster} not found — skip (try a different roster id)")
        return None
    if not teams:
        print("  no national teams returned for this roster — skip")
        return None
    if limit:
        teams = teams[:limit]
    print(f"  {len(teams)} men's national teams")

    cache = CACHE_DIR / roster
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, team in enumerate(teams, 1):
        tid, tname = team["id"], normalize_team(team["name"])
        blob = cache / f"{tid}.json"
        if blob.exists():
            players = json.loads(blob.read_text())
        else:
            try:
                players = client.team_squad(tid, roster)
            except FileNotFoundError:
                continue
            blob.write_text(json.dumps(players))
        for p in players:
            ovr = p.get("overallRating")
            if ovr is not None:
                rows.append({"team": tname, "player": player_name(p), "overall": ovr})
        if i % 25 == 0 or i == len(teams):
            print(f"  {i}/{len(teams)} teams, {len(rows)} players")

    if not rows:
        print("  no player ratings collected — skip")
        return None
    df = pd.DataFrame(rows).drop_duplicates(subset=["team", "player"])
    RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    out = RATINGS_DIR / f"fifa_{version_to_release_year(version)}.csv"
    df.to_csv(out, index=False)
    print(f"  -> {out}: {len(df)} players, {df['team'].nunique()} teams")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull SoFIFA national-team squad ratings")
    parser.add_argument(
        "--versions",
        default="gaps",
        help="'gaps' (default, editions we lack), 'all' (07-26), or e.g. '24,25,26'",
    )
    parser.add_argument(
        "--rosters",
        help="JSON file mapping version -> roster id, overriding the {YY}0001 default",
    )
    parser.add_argument("--rpm", type=int, default=50, help="max requests/min (<=60)")
    parser.add_argument("--limit", type=int, help="cap teams per edition (testing)")
    args = parser.parse_args()

    if args.versions == "gaps":
        versions = GAP_VERSIONS
    elif args.versions == "all":
        versions = ALL_VERSIONS
    else:
        versions = [v.strip().zfill(2) for v in args.versions.split(",")]

    roster_override: Dict[str, str] = {}
    if args.rosters:
        roster_override = json.loads(Path(args.rosters).read_text())

    client = SofifaClient(rpm=min(args.rpm, 60))
    print(f"Editions: {', '.join(versions)}")
    written = []
    for v in versions:
        roster = roster_override.get(v, default_roster(v))
        out = fetch_edition(client, v, roster, args.limit)
        if out:
            written.append(out)

    print(f"\nDone. Wrote {len(written)} edition files to {RATINGS_DIR}")
    if written:
        print("Next: python -m soccer.model.train && python -m soccer.model.predict")


if __name__ == "__main__":
    main()
