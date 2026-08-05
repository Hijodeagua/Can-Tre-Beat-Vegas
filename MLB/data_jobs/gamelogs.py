"""
Retrosheet game log downloader / parser / cache.

Layout:
    MLB/data/raw/gamelogs/GL{year}.TXT.gz   raw season files, gzipped verbatim
    MLB/data/games.csv.gz                   parsed table, one row per game
    MLB/data/manifest.json                  per-season ETag + row counts

Refresh semantics (python -m MLB.data_jobs.refresh):
    - a season already in the raw cache is only re-downloaded when the mirror
      reports a different ETag (completed seasons are immutable in practice,
      so a weekly run costs one conditional request per season);
    - the first missing season after the newest cached one is probed, so the
      next Retrosheet annual release is picked up automatically;
    - --force re-downloads everything.
"""

import gzip
import io
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd
import requests

from .config import (
    FIRST_SEASON,
    GAMELOG_URL_TEMPLATE,
    GAMES_PATH,
    MANIFEST_PATH,
    RAW_GAMELOG_DIR,
    REFERENCE_DIR,
    REFERENCE_FILES,
    RETRO_TO_FRANCHISE,
)
from .gamelog_fields import GAMELOG_COLUMNS, NUMERIC_COLUMNS

USER_AGENT = "can-tre-beat-vegas MLB pipeline (github.com/Hijodeagua/Can-Tre-Beat-Vegas)"

# Postseason logs live at the mirror root under gamelog/, one file per round
# covering all years. GLAS.TXT (All-Star game) is deliberately not modeled.
POSTSEASON_FILES = {
    "GLWC.TXT": "wildcard",
    "GLDV.TXT": "division",
    "GLLC.TXT": "lcs",
    "GLWS.TXT": "worldseries",
}
POSTSEASON_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/chadwickbureau/retrosheet/master/gamelog/{name}"
)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"seasons": {}}


def save_manifest(manifest: dict) -> None:
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _raw_path(year: int) -> str:
    return os.path.join(RAW_GAMELOG_DIR, f"GL{year}.TXT.gz")


def download_season(year: int, session: requests.Session, manifest: dict,
                    force: bool = False) -> str:
    """
    Ensure GL{year}.TXT.gz is cached. Returns one of:
    'downloaded', 'cached', 'unchanged', 'missing' (season not on the mirror).
    """
    path = _raw_path(year)
    entry = manifest["seasons"].get(str(year), {})
    headers = {"User-Agent": USER_AGENT}

    # The mirror is inconsistent about casing (GL2025.TXT vs gl2024.txt).
    urls = [
        GAMELOG_URL_TEMPLATE.format(year=year),
        GAMELOG_URL_TEMPLATE.format(year=year).replace(
            f"GL{year}.TXT", f"gl{year}.txt"
        ),
    ]

    if os.path.exists(path) and not force and entry.get("etag"):
        # Conditional GET against the stored ETag; any 200 without a match
        # falls through to a re-download.
        headers["If-None-Match"] = entry["etag"]

    resp = None
    for url in urls:
        resp = session.get(url, headers=headers, timeout=60)
        if resp.status_code != 404:
            break
    if resp.status_code == 304:
        return "unchanged"
    if resp.status_code == 404:
        return "missing"
    resp.raise_for_status()
    if entry.get("etag") and resp.headers.get("ETag") == entry["etag"]:
        return "cached"

    os.makedirs(RAW_GAMELOG_DIR, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(resp.content)
    manifest["seasons"][str(year)] = {
        "etag": resp.headers.get("ETag"),
        "bytes": len(resp.content),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return "downloaded"


def download_postseason(session: requests.Session, manifest: dict,
                        force: bool = False) -> dict:
    """Ensure the postseason gamelog files are cached. Returns name->status."""
    statuses = {}
    os.makedirs(RAW_GAMELOG_DIR, exist_ok=True)
    for name in POSTSEASON_FILES:
        url = POSTSEASON_URL_TEMPLATE.format(name=name)
        path = os.path.join(RAW_GAMELOG_DIR, name + ".gz")
        entry = manifest["seasons"].get(name, {})
        headers = {"User-Agent": USER_AGENT}
        if os.path.exists(path) and not force and entry.get("etag"):
            headers["If-None-Match"] = entry["etag"]
        resp = session.get(url, headers=headers, timeout=60)
        if resp.status_code == 304:
            statuses[name] = "unchanged"
            continue
        resp.raise_for_status()
        with gzip.open(path, "wb") as f:
            f.write(resp.content)
        manifest["seasons"][name] = {
            "etag": resp.headers.get("ETag"),
            "bytes": len(resp.content),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        statuses[name] = "downloaded"
    return statuses


def download_reference_files(session: requests.Session) -> list:
    """Fetch small reference tables (ballparks, teams). Always refreshed."""
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    fetched = []
    for name, url in REFERENCE_FILES.items():
        resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        resp.raise_for_status()
        with open(os.path.join(REFERENCE_DIR, name), "wb") as f:
            f.write(resp.content)
        fetched.append(name)
    return fetched


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

_LINESCORE_TOKEN = re.compile(r"\((\d+)\)|(\d)|x")


def innings_from_linescore(linescore: str) -> int:
    """
    Count innings in a Retrosheet line score. Multi-run innings are wrapped
    in parens ("010000(10)0x"); 'x' (home half not played) still counts as
    an inning slot. The away line score always has one entry per inning
    played, which is what refresh uses.
    """
    if not linescore or pd.isna(linescore):
        return 0
    return sum(1 for _ in _LINESCORE_TOKEN.finditer(linescore))


def parse_gamelog_file(path: str, game_type: str) -> pd.DataFrame:
    """Parse a cached raw gamelog file (gzipped) into the games table schema."""
    with gzip.open(path, "rt", encoding="latin-1") as f:
        df = pd.read_csv(
            f, header=None, names=GAMELOG_COLUMNS, dtype=str,
            keep_default_na=False, na_values=[""],
        )

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    # Retrosheet writes unknown attendance/duration as 0 or -1.
    for col in ("attendance", "duration_minutes"):
        df.loc[df[col] <= 0, col] = pd.NA

    df.insert(0, "season", df["date"].str[:4].astype(int))
    df.insert(1, "game_type", game_type)
    df.insert(
        2, "game_id",
        df["date"] + "_" + df["game_num"] + "_" + df["home_team_retro"],
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.date.astype(str)

    df["home_team"] = df["home_team_retro"].map(RETRO_TO_FRANCHISE)
    df["away_team"] = df["away_team_retro"].map(RETRO_TO_FRANCHISE)

    df["innings"] = df["away_linescore"].map(innings_from_linescore).astype("Int64")
    # A shortened (rain / COVID-doubleheader) game has < 51 outs; extras > 54.
    df["extra_innings"] = (df["innings"] > 9).astype("Int64")
    df["home_win"] = (df["home_score"] > df["away_score"]).astype("Int64")
    df["run_diff_home"] = df["home_score"] - df["away_score"]

    return df


def parse_season(year: int) -> pd.DataFrame:
    """Parse a cached regular-season file."""
    return parse_gamelog_file(_raw_path(year), "regular")


def parse_postseason() -> pd.DataFrame:
    """Parse cached postseason files, filtered to FIRST_SEASON+."""
    frames = []
    for name, game_type in POSTSEASON_FILES.items():
        path = os.path.join(RAW_GAMELOG_DIR, name + ".gz")
        if os.path.exists(path):
            frames.append(parse_gamelog_file(path, game_type))
    if not frames:
        return pd.DataFrame()
    post = pd.concat(frames, ignore_index=True)
    return post[post["season"] >= FIRST_SEASON].reset_index(drop=True)


def build_games_table(years) -> pd.DataFrame:
    """Parse all cached seasons + postseason and write MLB/data/games.csv.gz."""
    frames = [parse_season(y) for y in years]
    post = parse_postseason()
    if not post.empty:
        frames.append(post)
    games = pd.concat(frames, ignore_index=True)
    games = games.sort_values(["date", "game_num", "home_team_retro"]).reset_index(drop=True)

    # Join-coverage assertions (house rule 7): every retro code must map to a
    # franchise, and game_ids must be unique.
    unmapped = games.loc[
        games["home_team"].isna() | games["away_team"].isna(),
        ["season", "home_team_retro", "away_team_retro"],
    ]
    if not unmapped.empty:
        codes = sorted(
            set(unmapped["home_team_retro"]) | set(unmapped["away_team_retro"])
        )
        raise AssertionError(
            f"{len(unmapped)} games have Retrosheet team codes missing from "
            f"the franchise crosswalk: {codes}"
        )
    dupes = games["game_id"].duplicated().sum()
    if dupes:
        raise AssertionError(f"{dupes} duplicate game_ids after concat")

    os.makedirs(os.path.dirname(GAMES_PATH), exist_ok=True)
    games.to_csv(GAMES_PATH, index=False, compression="gzip")
    return games


def cached_seasons() -> list:
    if not os.path.isdir(RAW_GAMELOG_DIR):
        return []
    years = []
    for name in os.listdir(RAW_GAMELOG_DIR):
        m = re.fullmatch(r"GL(\d{4})\.TXT\.gz", name)
        if m:
            years.append(int(m.group(1)))
    return sorted(y for y in years if y >= FIRST_SEASON)
