"""Public archive of every report email, by league.

Each pipeline already writes its email as HTML under reports/<pipeline>/
and hashes it for the send ledger; nothing kept a browsable record of what
went out, let alone one you could sort by league. This module does, for
all five senders (MLB, soccer, CFB, NFL, the weekly models check):

- `publish(league, entries)` — called by a pipeline after it renders its
  emails and BEFORE the send ledger hashes them. For each entry it appends
  the archive footer (permalink, the previous email of the same league and
  type, the league's archive page) to the HTML on disk, copies the file to
  web/public/emails/<league>/<date>_<type>.html so the site serves it at a
  stable URL, and upserts the entry into web/public/emails/index.json.
  The footer is appended before hashing so the delivered copy and the
  archived copy are byte-identical.
- `index.json` — one record per email: league, type, date, subject,
  repo path, public URL, archived_at. The site's /emails page reads it
  and sorts/filters by league; the record is keyed so a rerun of the same
  day updates in place rather than duplicating.
- `backfill` (CLI) — walks the existing reports/ trees once so emails sent
  before the archive existed are in it too (they keep their original
  bytes: no footer is added retroactively).

    python -m data_jobs.email_archive backfill
    python -m data_jobs.email_archive publish --league nfl \
        --manifest reports/nfl/manifest_latest.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = REPO_ROOT / "web" / "public" / "emails"
INDEX_FILE = ARCHIVE_DIR / "index.json"
SITE_BASE = "https://whosyurgoat.app/vegas"

# League key -> display name, and where each pipeline keeps its reports
# (for the backfill). The models check spans every sport; it archives as
# its own "league" so the archive page can still group it.
LEAGUES = {
    "mlb": {"name": "MLB", "reports": "reports/mlb_daily", "emoji": "⚾"},
    "soccer": {"name": "Soccer", "reports": "reports/soccer", "emoji": "⚽"},
    "cfb": {"name": "College Football", "reports": "reports/cfb", "emoji": "🎓"},
    "nfl": {"name": "NFL", "reports": "reports/nfl", "emoji": "🏈"},
    "models": {"name": "Models check", "reports": "reports/models_check", "emoji": "📊"},
}
# Report-file stem -> email type, for the backfill.
FILE_TYPES = {"futures": "futures", "slate": "slate", "grade": "grade",
              "update": "update", "check": "models"}

FOOTER_MARK = "<!-- email-archive-footer -->"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _key(league: str, email_type: str, date: str) -> str:
    return f"{league}:{email_type}:{date}"


def load_index(index_file: Path = INDEX_FILE) -> dict:
    if index_file.exists():
        return json.loads(index_file.read_text(encoding="utf-8"))
    return {"generated_at": None, "emails": []}


def save_index(index: dict, index_file: Path = INDEX_FILE) -> None:
    emails = sorted(index["emails"], key=lambda e: (e["league"], e["date"], e["type"]),
                    reverse=True)
    index = {"generated_at": _now(), "leagues": {k: v["name"] for k, v in LEAGUES.items()},
             "emails": emails}
    index_file.parent.mkdir(parents=True, exist_ok=True)
    index_file.write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n",
                          encoding="utf-8")


def public_path(league: str, email_type: str, date: str) -> str:
    """Site-relative path of an archived email (under the /vegas basePath)."""
    return f"emails/{league}/{date}_{email_type}.html"


def public_url(league: str, email_type: str, date: str) -> str:
    return f"{SITE_BASE}/{public_path(league, email_type, date)}"


def archive_page_url(league: str) -> str:
    return f"{SITE_BASE}/emails?league={league}"


def previous_entry(index: dict, league: str, email_type: str, date: str) -> dict | None:
    """The most recent archived email of the same league and type dated
    strictly before `date`."""
    prior = [e for e in index["emails"]
             if e["league"] == league and e["type"] == email_type and e["date"] < date]
    return max(prior, key=lambda e: e["date"]) if prior else None


def footer_html(league: str, email_type: str, date: str, prev: dict | None) -> str:
    name = LEAGUES.get(league, {}).get("name", league)
    prev_link = (
        f' &middot; Previous: <a href="{prev["url"]}">{prev["date"]}</a>'
        if prev else ""
    )
    return (
        f"{FOOTER_MARK}"
        f'<p style="font-family:Arial,Helvetica,sans-serif;color:#999;font-size:11px;'
        f'margin-top:8px;">'
        f'This email online: <a href="{public_url(league, email_type, date)}">'
        f"{date} {email_type}</a>{prev_link} &middot; "
        f'<a href="{archive_page_url(league)}">All {name} emails</a></p>'
    )


def _upsert(index: dict, record: dict) -> None:
    k = _key(record["league"], record["type"], record["date"])
    index["emails"] = [e for e in index["emails"]
                       if _key(e["league"], e["type"], e["date"]) != k]
    index["emails"].append(record)


def publish(league: str, entries: dict, repo_root: Path = REPO_ROOT,
            archive_dir: Path | None = None, index_file: Path | None = None) -> dict:
    """Archive each email in `entries` ({type: {path, subject, date}} — the
    manifest shape every pipeline builds). Appends the archive footer to
    the HTML on disk (once), copies it into the archive, and updates the
    index. Adds `url` to each entry; mutates and returns `entries`."""
    if league not in LEAGUES:
        raise ValueError(f"unknown league {league!r}; known: {sorted(LEAGUES)}")
    archive_dir = archive_dir or ARCHIVE_DIR
    index_file = index_file or (archive_dir / "index.json")
    index = load_index(index_file)
    for email_type, entry in entries.items():
        src = repo_root / entry["path"]
        date = entry["date"]
        html = src.read_text(encoding="utf-8")
        if FOOTER_MARK not in html:
            prev = previous_entry(index, league, email_type, date)
            html = html + footer_html(league, email_type, date, prev)
            src.write_text(html, encoding="utf-8")
        dest = archive_dir / league / f"{date}_{email_type}.html"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        entry["url"] = public_url(league, email_type, date)
        _upsert(index, {
            "league": league, "type": email_type, "date": date,
            "subject": entry.get("subject", ""),
            "path": entry["path"],
            "public_path": public_path(league, email_type, date),
            "url": entry["url"],
            "archived_at": _now(),
        })
    save_index(index, index_file)
    return entries


def backfill(repo_root: Path = REPO_ROOT, archive_dir: Path | None = None,
             index_file: Path | None = None) -> int:
    """Archive every email already on disk under reports/. Files are copied
    as they were sent (no footer); subjects come from the send ledgers
    where one exists. Existing index records are kept as-is."""
    archive_dir = archive_dir or ARCHIVE_DIR
    index_file = index_file or (archive_dir / "index.json")
    index = load_index(index_file)
    have = {_key(e["league"], e["type"], e["date"]) for e in index["emails"]}
    added = 0
    for league, spec in LEAGUES.items():
        root = repo_root / spec["reports"]
        if not root.exists():
            continue
        sent = {}
        ledger = root / "sent.json"
        if ledger.exists():
            sent = json.loads(ledger.read_text(encoding="utf-8"))
        for day_dir in sorted(p for p in root.iterdir() if p.is_dir() and _DATE_RE.match(p.name)):
            for html in sorted(day_dir.glob("*.html")):
                email_type = FILE_TYPES.get(html.stem)
                if not email_type:
                    continue
                # The MLB grade email is keyed by the graded date (the day
                # before the run) in its ledger; the file lives under the run
                # date. Use the ledger key when one matches, else the folder.
                date = day_dir.name
                subject = ""
                for k, v in sent.items():
                    t, _, d = k.partition(":")
                    if t == email_type and (d == date or (
                            email_type == "grade" and _prev_day(date) == d)):
                        date, subject = d, v.get("subject", "")
                        break
                k = _key(league, email_type, date)
                if k in have:
                    continue
                dest = archive_dir / league / f"{date}_{email_type}.html"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(html, dest)
                _upsert(index, {
                    "league": league, "type": email_type, "date": date,
                    "subject": subject,
                    "path": str(html.relative_to(repo_root)),
                    "public_path": public_path(league, email_type, date),
                    "url": public_url(league, email_type, date),
                    "archived_at": _now(),
                })
                have.add(k)
                added += 1
    save_index(index, index_file)
    return added


def _prev_day(date: str) -> str:
    from datetime import date as _d, timedelta
    return (_d.fromisoformat(date) - timedelta(days=1)).isoformat()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pub = sub.add_parser("publish")
    pub.add_argument("--league", required=True, choices=sorted(LEAGUES))
    pub.add_argument("--manifest", required=True, help="manifest path, relative to the repo root")
    sub.add_parser("backfill")
    args = ap.parse_args(argv)
    if args.cmd == "backfill":
        n = backfill()
        print(f"email_archive: backfilled {n} emails -> {INDEX_FILE}")
        return 0
    manifest = json.loads((REPO_ROOT / args.manifest).read_text(encoding="utf-8"))
    publish(args.league, manifest["emails"])
    print(f"email_archive: published {len(manifest['emails'])} {args.league} emails")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
