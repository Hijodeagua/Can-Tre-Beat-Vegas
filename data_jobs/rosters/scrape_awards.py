"""Scrape All-Pro, Pro Bowl, and NFL Top 100 honors. **Run this locally.**

## Why local-only

Both hosts are refused by this project's CI/agent egress policy (the proxy
answers 403 to CONNECT for ``pro-football-reference.com`` and
``en.wikipedia.org``), and PFR additionally rate-limits and Cloudflare-blocks
datacenter traffic. So this module is written to run on your own machine, at
a polite crawl, and to commit its *output* CSVs. Do not add it to a workflow.

## Belt and braces: raw HTML is saved

Every fetched page is written to ``data/rosters/awards/raw/`` before parsing.
If a parser turns out to be wrong for a given year's markup, re-run with
``--reparse`` and nothing gets re-fetched. This matters because the parsers
below were written against PFR's documented table structure but could not be
verified against live HTML from the development sandbox — treat the first run
as something to eyeball, and check ``--report`` output for unresolved names.

## No-network path (save the pages by hand)

``--reparse`` never touches the network: it just reads whatever HTML is
already in ``data/rosters/awards/raw/``. So if the machine running this can't
reach the sites either, open the pages in a browser, "Save Page As" into that
folder using these exact names, and run ``--reparse``:

    data/rosters/awards/raw/allpro_2024.html      <- .../years/2024/allpro.htm
    data/rosters/awards/raw/probowl_2024.html     <- .../years/2024/probowl.htm
    data/rosters/awards/raw/top100_2024.html      <- Wikipedia NFL Top 100 2024

Partial coverage is fine — any season whose file is missing is simply skipped,
and the honor features fall back to NaN for the rosters of that season.

## Output

``data/rosters/awards/allpro.csv``  — season, pfr_id, player, team_level
``data/rosters/awards/probowl.csv`` — season, pfr_id, player
``data/rosters/awards/top100.csv``  — season, rank, player, pfr_id

``pfr_id`` is the join key the squad features use (e.g. ``MahoPa00``). It comes
free from PFR player links; for Wikipedia's Top 100 it is resolved by name
against the nflverse player master, and anything unresolved is reported so you
can patch it by hand.

Usage
    python3 -m data_jobs.rosters.scrape_awards --first 2002 --last 2025
    python3 -m data_jobs.rosters.scrape_awards --reparse      # no network
    python3 -m data_jobs.rosters.scrape_awards --report       # coverage check
"""

from __future__ import annotations

import argparse
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
AWARDS_DIR = REPO_ROOT / "data" / "rosters" / "awards"
RAW_DIR = AWARDS_DIR / "raw"
PLAYERS_CSV = REPO_ROOT / "data" / "rosters" / "nflverse" / "players.csv"

PFR_ALLPRO = "https://www.pro-football-reference.com/years/{season}/allpro.htm"
PFR_PROBOWL = "https://www.pro-football-reference.com/years/{season}/probowl.htm"
WIKI_TOP100 = "https://en.wikipedia.org/wiki/NFL_Top_100_Players_of_{season}"

# PFR asks for a real UA and punishes speed. 4s is comfortably under their
# published 20-requests-per-minute ceiling.
DELAY_SECONDS = 4.0
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
TIMEOUT = 45

PLAYER_LINK_RE = re.compile(r"/players/[A-Z]/([A-Za-z0-9]+)\.htm")
FIRST_TOP100_SEASON = 2011  # the player-voted list does not exist before this


def _fetch(url: str, dest: Path, force: bool = False) -> str | None:
    """Fetch to disk (cached), return the HTML text."""
    if dest.exists() and not force and dest.stat().st_size > 0:
        return dest.read_text(encoding="utf-8", errors="ignore")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            html = r.read().decode("utf-8", errors="ignore")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        print(f"  FAIL {url}: {exc}")
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html, encoding="utf-8")
    time.sleep(DELAY_SECONDS)
    return html


def _soup(html: str):
    from bs4 import BeautifulSoup
    # PFR hides most tables inside HTML comments to deter scrapers; unwrapping
    # the comments makes them visible to the parser.
    html = html.replace("<!--", "").replace("-->", "")
    return BeautifulSoup(html, "html.parser")


def parse_pfr_players(html: str) -> pd.DataFrame:
    """Every distinct player linked from the page's tables."""
    soup = _soup(html)
    rows = {}
    for a in soup.find_all("a", href=True):
        m = PLAYER_LINK_RE.search(a["href"])
        if not m:
            continue
        name = a.get_text(strip=True)
        if name:
            rows.setdefault(m.group(1), name)
    return pd.DataFrame({"pfr_id": list(rows), "player": list(rows.values())})


def parse_allpro(html: str) -> pd.DataFrame:
    """All-Pro selections, keeping first/second team where the markup says so."""
    soup = _soup(html)
    out = {}
    for tr in soup.find_all("tr"):
        link = tr.find("a", href=PLAYER_LINK_RE)
        if not link:
            continue
        m = PLAYER_LINK_RE.search(link["href"])
        text = tr.get_text(" ", strip=True).lower()
        level = 2 if ("2nd" in text or "second" in text) else 1
        pid = m.group(1)
        # First team wins if a player appears twice.
        if pid not in out or level < out[pid][1]:
            out[pid] = (link.get_text(strip=True), level)
    if not out:
        base = parse_pfr_players(html)
        base["team_level"] = 1
        return base
    return pd.DataFrame([{"pfr_id": k, "player": v[0], "team_level": v[1]}
                         for k, v in out.items()])


MIN_TOP100_ROWS = 50


def parse_wikipedia_top100(html: str) -> pd.DataFrame:
    """Rank + player name from the list table.

    Parsed with BeautifulSoup rather than ``pandas.read_html``. read_html needs
    lxml or html5lib, neither of which this project declares — it happened to
    work in a dev environment where lxml had arrived as somebody's transitive
    dependency, and returned an empty frame in CI. bs4 is already a declared
    dependency and already used elsewhere in this module.
    """
    soup = _soup(html)
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < MIN_TOP100_ROWS:
            continue

        header = [c.get_text(strip=True).lower()
                  for c in rows[0].find_all(["th", "td"])]
        rank_i = next((i for i, h in enumerate(header) if "rank" in h or h == "#"), None)
        name_i = next((i for i, h in enumerate(header)
                       if "player" in h or "name" in h), None)
        if rank_i is None or name_i is None:
            continue

        out = []
        for tr in rows[1:]:
            cells = tr.find_all(["td", "th"])
            if len(cells) <= max(rank_i, name_i):
                continue
            rank = pd.to_numeric(cells[rank_i].get_text(strip=True), errors="coerce")
            if pd.isna(rank):
                continue
            name = re.sub(r"\[.*?\]", "", cells[name_i].get_text(strip=True)).strip()
            out.append({"rank": rank, "player": name})

        if len(out) >= MIN_TOP100_ROWS:
            return pd.DataFrame(out)
    return pd.DataFrame(columns=["rank", "player"])


SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b", re.IGNORECASE)

# Wikipedia's preferred name vs the one nflverse files the player under.
# Mostly players who changed the name they go by mid-career; verified against
# players.csv one at a time rather than fuzzy-matched, because a wrong id here
# silently credits the wrong roster.
NAME_ALIASES = {
    "michaelvick": "mikevick",
    "robertgriffiniii": "robertgriffin",
    "stevensmithsr": "stevensmith",
    "dariusshaquilleleonard": "shaquilleleonard",  # PFR LeonDa00
    "justinmadubuike": "nnamdimadubuike",          # PFR MaduJu00
    "tariqwoolen": "riqwoolen",                    # PFR WoolTa00
}


def normalize_name(s: pd.Series) -> pd.Series:
    """Fold the ways the same player gets written across sources.

    Wikipedia writes ``B. J. Raji``, ``Pierre Garçon``, ``Chris Harris Jr.``;
    nflverse writes ``B.J. Raji``, ``Pierre Garcon``, ``Chris Harris``. Dropping
    accents, suffixes and *all* separators (spaces included, so spaced initials
    collapse the same way unspaced ones do) reconciles them.
    """
    import unicodedata

    def _strip_accents(x: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFKD", str(x))
                       if not unicodedata.combining(c))

    out = s.fillna("").map(_strip_accents).str.lower()
    out = out.str.replace(r"[.\-']", " ", regex=True)
    out = out.map(lambda x: SUFFIX_RE.sub(" ", x))
    out = out.str.replace(r"[^a-z0-9]", "", regex=True)
    return out.map(lambda x: NAME_ALIASES.get(x, x))


def resolve_pfr_ids(names: pd.Series) -> pd.Series:
    """Map display names to pfr_id via the nflverse player master.

    Wikipedia carries no player ids, so this is the join that decides whether
    a Top 100 entry can reach a roster at all. An unresolved name does not
    error — it silently vanishes from ``n_top100`` — so the match rate is
    reported by ``--report`` and worth watching.
    """
    if not PLAYERS_CSV.exists():
        return pd.Series([pd.NA] * len(names), index=names.index)
    p = pd.read_csv(PLAYERS_CSV, low_memory=False,
                    usecols=["display_name", "pfr_id"]).dropna()
    p = p.assign(k=normalize_name(p["display_name"]))
    # Prefer the first listing for a name collision; ambiguous names are rare
    # among Top 100 players and a wrong id would only mis-credit one roster.
    key = p.drop_duplicates("k").set_index("k")["pfr_id"]
    return normalize_name(names).map(key)


def scrape(first: int, last: int, reparse: bool = False, force: bool = False) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    allpro, probowl, top100 = [], [], []

    for season in range(first, last + 1):
        print(f"{season}:")
        for label, url_tpl, parser, sink in (
            ("allpro", PFR_ALLPRO, parse_allpro, allpro),
            ("probowl", PFR_PROBOWL, parse_pfr_players, probowl),
        ):
            dest = RAW_DIR / f"{label}_{season}.html"
            html = (dest.read_text(encoding="utf-8", errors="ignore")
                    if reparse and dest.exists() else _fetch(url_tpl.format(season=season), dest, force))
            if not html:
                continue
            d = parser(html)
            if len(d):
                d["season"] = season
                sink.append(d)
            print(f"  {label}: {len(d)} players")

        if season >= FIRST_TOP100_SEASON:
            dest = RAW_DIR / f"top100_{season}.html"
            html = (dest.read_text(encoding="utf-8", errors="ignore")
                    if reparse and dest.exists() else _fetch(WIKI_TOP100.format(season=season), dest, force))
            if html:
                d = parse_wikipedia_top100(html)
                if len(d):
                    d["season"] = season
                    d["pfr_id"] = resolve_pfr_ids(d["player"])
                    top100.append(d)
                print(f"  top100: {len(d)} players "
                      f"({d['pfr_id'].notna().sum() if len(d) else 0} id-matched)")

    AWARDS_DIR.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, frames, cols in (
        ("allpro", allpro, ["season", "pfr_id", "player", "team_level"]),
        ("probowl", probowl, ["season", "pfr_id", "player"]),
        ("top100", top100, ["season", "rank", "player", "pfr_id"]),
    ):
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df = df[[c for c in cols if c in df.columns]]
        path = AWARDS_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        written[name] = len(df)
        print(f"wrote {path.relative_to(REPO_ROOT)} ({len(df)} rows)")
    return written


def report() -> None:
    """Sanity-check what was scraped; flags years that look wrong."""
    for name, expected in (("allpro", (40, 130)), ("probowl", (60, 200)),
                           ("top100", (95, 105))):
        path = AWARDS_DIR / f"{name}.csv"
        if not path.exists():
            print(f"{name}: MISSING — run the scrape")
            continue
        d = pd.read_csv(path)
        per = d.groupby("season").size()
        odd = per[(per < expected[0]) | (per > expected[1])]
        print(f"{name}: {len(d)} rows, {per.index.min()}-{per.index.max()}, "
              f"median {per.median():.0f}/season")
        if "pfr_id" in d.columns:
            print(f"  unresolved pfr_id: {d['pfr_id'].isna().sum()}")
        if len(odd):
            print(f"  seasons outside expected {expected}: {odd.to_dict()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first", type=int, default=2002)
    ap.add_argument("--last", type=int, default=2025)
    ap.add_argument("--reparse", action="store_true",
                    help="re-parse cached HTML without any network access")
    ap.add_argument("--force", action="store_true", help="re-fetch even if cached")
    ap.add_argument("--report", action="store_true", help="only summarise existing CSVs")
    args = ap.parse_args()

    if args.report:
        report()
        return
    scrape(args.first, args.last, args.reparse, args.force)
    print()
    report()


if __name__ == "__main__":
    main()
