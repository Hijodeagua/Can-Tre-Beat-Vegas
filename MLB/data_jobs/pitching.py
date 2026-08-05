"""
Per-pitcher, per-game lines from the Chadwick Bureau's retrosplits project
(github.com/chadwickbureau/retrosplits): daily player splits computed from
Retrosheet event data. This is what the game logs alone can't provide —
each starter's own IP/K/BB/ER per start (rolling ERA, FIP, K-BB%) and each
reliever's innings by date (bullpen fatigue), seasons 2005-2025.

The upstream playing-{year}.csv files are ~25MB each (batting+pitching+
fielding for every player-game), so the raw files are not kept: each season
is downloaded, filtered to pitcher-games, trimmed to the pitching columns,
and cached as MLB/data/pitching/pitching_{year}.csv.gz (~0.5MB). Season
files are immutable once published; the manifest ETag makes re-runs cheap.

Join keys: person.key is the same Retrosheet player ID the game logs use
for starting pitchers, and game.key is {home}{yyyymmdd}{game_num}, which
maps 1:1 onto games.csv.gz game_ids. refresh.py asserts the join.
"""

import gzip
import io
import os
from datetime import datetime, timezone

import pandas as pd
import requests

from .config import DATA_DIR, FIRST_SEASON

PITCHING_DIR = os.path.join(DATA_DIR, "pitching")
RETROSPLITS_URL = (
    "https://raw.githubusercontent.com/chadwickbureau/retrosplits/master"
    "/daybyday/playing-{year}.csv"
)
USER_AGENT = "can-tre-beat-vegas MLB pipeline (github.com/Hijodeagua/Can-Tre-Beat-Vegas)"

KEY_COLUMNS = [
    "game.key", "game.date", "game.number", "team.alignment", "team.key",
    "person.key", "seq",
]
PITCHING_COLUMNS = [
    "P_G", "P_GS", "P_CG", "P_GF", "P_W", "P_L", "P_SV",
    "P_OUT", "P_TBF", "P_AB", "P_R", "P_ER", "P_H", "P_2B", "P_3B",
    "P_HR", "P_BB", "P_IBB", "P_SO", "P_HP", "P_WP", "P_BK",
    "P_IR", "P_PITCH", "P_STRIKE",
]


def _season_path(year: int) -> str:
    return os.path.join(PITCHING_DIR, f"pitching_{year}.csv.gz")


def download_season(year: int, session: requests.Session, manifest: dict,
                    force: bool = False) -> str:
    """
    Ensure pitching_{year}.csv.gz is cached. Returns 'downloaded', 'cached',
    'unchanged', or 'missing'.
    """
    key = f"retrosplits-{year}"
    path = _season_path(year)
    entry = manifest["seasons"].get(key, {})
    headers = {"User-Agent": USER_AGENT}
    if os.path.exists(path) and not force:
        if not entry.get("etag"):
            return "cached"
        headers["If-None-Match"] = entry["etag"]

    resp = session.get(
        RETROSPLITS_URL.format(year=year), headers=headers, timeout=300
    )
    if resp.status_code == 304:
        return "unchanged"
    if resp.status_code == 404:
        return "missing"
    resp.raise_for_status()

    df = pd.read_csv(io.BytesIO(resp.content), dtype=str)
    keep = [c for c in KEY_COLUMNS + PITCHING_COLUMNS if c in df.columns]
    pitchers = df.loc[pd.to_numeric(df.get("P_G"), errors="coerce") > 0, keep].copy()
    if pitchers.empty:
        raise AssertionError(f"retrosplits {year}: no pitcher rows parsed")

    os.makedirs(PITCHING_DIR, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        pitchers.to_csv(f, index=False)
    manifest["seasons"][key] = {
        "etag": resp.headers.get("ETag"),
        "bytes": len(resp.content),
        "rows": len(pitchers),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return "downloaded"


def cached_seasons() -> list:
    if not os.path.isdir(PITCHING_DIR):
        return []
    years = []
    for name in os.listdir(PITCHING_DIR):
        if name.startswith("pitching_") and name.endswith(".csv.gz"):
            years.append(int(name[len("pitching_"):-len(".csv.gz")]))
    return sorted(y for y in years if y >= FIRST_SEASON)


def load_season(year: int) -> pd.DataFrame:
    df = pd.read_csv(_season_path(year))
    for col in PITCHING_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


# The game log and the event data occasionally disagree about who "started"
# (opener situations, Retrosheet errata applied to one release but not the
# other). More than this many per season means a broken join, not errata.
MAX_STARTER_MISMATCHES_PER_SEASON = 3


def assert_join_coverage(games: pd.DataFrame, years: list) -> tuple:
    """
    House rule 7: after the merge, prove it.

    Hard requirements, per cached season: every REGULAR-season game joins to
    exactly one P_GS=1 row per side, and starter mismatches vs the game log
    stay within the errata allowance. Postseason gaps are reported as notes
    (retrosplits lacks the single-game wild-card rounds before 2022) rather
    than failures. Returns (problems, notes).
    """
    problems, notes = [], []
    for year in years:
        p = load_season(year)
        g = games[games["season"] == year]
        if g.empty:
            continue
        # game.key is {home_retro}{yyyymmdd}{game_num}; games.csv.gz game_id
        # is {yyyymmdd}_{game_num}_{home_retro}.
        p = p.assign(
            game_id=(
                p["game.key"].str[3:11] + "_" + p["game.key"].str[11:]
                + "_" + p["game.key"].str[:3]
            )
        )
        starters = p[p["P_GS"] == 1]
        per_side = starters.groupby(["game_id", "team.alignment"]).size()
        bad = per_side[per_side != 1]
        if len(bad):
            problems.append(f"{year}: {len(bad)} game-sides without exactly 1 starter")

        # team.alignment: 0 = away, 1 = home.
        home_starters = starters[starters["team.alignment"] == 1]
        merged = g.merge(
            home_starters[["game_id", "person.key"]], on="game_id", how="left",
        )
        is_reg = merged["game_type"] == "regular"

        reg_missing = int((is_reg & merged["person.key"].isna()).sum())
        if reg_missing:
            problems.append(
                f"{year}: {reg_missing} regular-season games with no starter row"
            )
        post_missing = int((~is_reg & merged["person.key"].isna()).sum())
        if post_missing:
            notes.append(
                f"{year}: {post_missing} postseason games not in retrosplits"
            )

        mismatch = int((
            merged["person.key"].notna()
            & (merged["person.key"] != merged["home_sp_id"])
        ).sum())
        if mismatch > MAX_STARTER_MISMATCHES_PER_SEASON:
            problems.append(
                f"{year}: {mismatch} starter mismatches vs the game log "
                f"(allowance {MAX_STARTER_MISMATCHES_PER_SEASON})"
            )
        elif mismatch:
            notes.append(
                f"{year}: {mismatch} starter mismatch(es) vs the game log "
                f"(opener/errata)"
            )
    return problems, notes
