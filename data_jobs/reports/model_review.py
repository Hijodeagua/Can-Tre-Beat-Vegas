"""Weekly model-review drafts: a stats-heavy one-pager per model that a
Claude session then assesses and sends to TW.

    python -m data_jobs.reports.model_review --model nfl|cfb|soccer|basketball|all
                                             [--date YYYY-MM-DD] [--out DIR]

For each model the draft is written as JSON and as a self-contained HTML
page (`model_review_html.render`) under reports/model_review/<model>/:
<date>.json, <date>.html, plus a stable latest.json / latest.html copy.
The Claude session reads `latest`, confirms the date matches the day it
runs, checks key numbers against the files named in each section, and
writes the final assessment into the same template.

What is in a draft, and the rule for every number in it: it comes from a
file that is committed in this repo, and the section says which file and
how old it is. Nothing is estimated, and no model metric is invented for
a model that does not exist (basketball has odds data and no model, and
its page says exactly that).

Sections (fixed order, rendered by the template):

1. Current status - run health (from `_runs/<workflow>.json`, written by
   the workflow with `gh run list`; "unavailable" when generated outside
   Actions), last successful run, feed freshness, the last graded date,
   known failures, and the verdict: live / stale / incomplete.
2. Key features - model type, target, training window, inputs, tuned
   parameters, and the committed importance study where one exists
   (`web/public/data/importance.json`, built by data_jobs/build_importance).
3. Interesting findings - the biggest misses and the best upset calls in
   the last 30 days, rating movers over the week, forecast shifts against
   the site snapshot nearest seven days ago, calibration drift, and
   segment splits.
4. Report card - accuracy, log loss, Brier, the always-home baseline and
   the paired delta with its standard error, for the last 7 days, the
   last 30 days and the season, each with its sample size; plus the
   calibration table.
5. Potential improvements - observations with the number that motivates
   each, phrased as candidates for the reviewer to weigh, never as
   decisions.

The schedule (`.github/workflows/weekly-model-review.yml`): Monday CFB
and soccer, Tuesday basketball, Wednesday NFL. "Football" in that
schedule means the NFL model; college football has its own Monday slot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data_jobs.reports.model_review_html import render

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "reports" / "model_review"
RUNS_DIR = OUT_DIR / "_runs"
SITE = REPO / "web" / "public" / "data"
IMPORTANCE_JSON = SITE / "importance.json"
SUMMARY_JSON = SITE / "summary.json"
SLATE_JSON = SITE / "slate.json"
DOCS_FEATURES = REPO / "docs" / "MODEL_FEATURES.md"
DOCS_SOURCES = REPO / "docs" / "MODEL_SOURCES.md"

MODELS = ("cfb", "soccer", "basketball", "nfl")
# Weekday (Mon=0) -> models reviewed that day. Football = NFL; college has
# its own slot.
SCHEDULE = {0: ("cfb", "soccer"), 1: ("basketball",), 2: ("nfl",)}

WINDOWS = ((7, "Last 7 days"), (30, "Last 30 days"), (None, "Season to date"))
CAL_BINS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
TOP_N = 5


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _mtime(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d")
    except OSError:
        return None


def _pct(x, digits=1) -> str | None:
    return None if x is None or x != x else f"{100 * x:.{digits}f}%"


def _r(x, d=4):
    if x is None:
        return None
    try:
        if x != x:
            return None
        return round(float(x), d)
    except (TypeError, ValueError):
        return None


def _days_between(a: str, b: str) -> int | None:
    try:
        return (date.fromisoformat(b[:10]) - date.fromisoformat(a[:10])).days
    except (TypeError, ValueError):
        return None


def _window(df: pd.DataFrame, run_date: str, days: int | None) -> pd.DataFrame:
    if days is None:
        return df
    start = (date.fromisoformat(run_date) - timedelta(days=days)).isoformat()
    return df[(df["date"] >= start) & (df["date"] <= run_date)]


def _se(x: pd.Series) -> float | None:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 2:
        return None
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def _nearest_snapshot(history_dir: Path, run_date: str, back_days: int = 7) -> Path | None:
    """The site snapshot dated closest to run_date - back_days (never after
    run_date), so "a week ago" survives a missed daily run."""
    target = date.fromisoformat(run_date) - timedelta(days=back_days)
    best, best_gap = None, None
    for p in history_dir.glob("*.json"):
        try:
            d = date.fromisoformat(p.stem)
        except ValueError:
            continue
        if d > date.fromisoformat(run_date) or d == date.fromisoformat(run_date):
            continue
        gap = abs((d - target).days)
        if best is None or gap < best_gap:
            best, best_gap = p, gap
    return best


def _run_health(workflows: list[str]) -> tuple[list[dict], list[str]]:
    """Rows for the run-health table from the gh dumps, plus notes.
    Each dump is `gh run list --json conclusion,status,createdAt,url,
    displayTitle --limit 10` for one workflow file."""
    rows, notes = [], []
    for wf in workflows:
        p = RUNS_DIR / f"{wf}.json"
        runs = _read_json(p)
        if not isinstance(runs, list):
            rows.append([wf, "unavailable", None, None, None])
            notes.append(f"{wf}: run history unavailable (no `{_rel(p)}`; the workflow writes it from "
                         f"`gh run list` and a local run has no token)")
            continue
        completed = [r for r in runs if r.get("status") == "completed"]
        last = completed[0] if completed else None
        last_ok = next((r for r in completed if r.get("conclusion") == "success"), None)
        fails = sum(1 for r in completed if r.get("conclusion") not in ("success", "skipped", None))
        rows.append([wf, (last or {}).get("conclusion") or "none",
                     (last or {}).get("createdAt", "")[:16].replace("T", " ") or None,
                     (last_ok or {}).get("createdAt", "")[:16].replace("T", " ") or None,
                     f"{fails}/{len(completed)}"])
        if fails:
            notes.append(f"{wf}: {fails} of the last {len(completed)} completed runs failed")
    return rows, notes


def _importance_block(key: str) -> dict | None:
    imp = _read_json(IMPORTANCE_JSON) or {}
    m = (imp.get("models") or {}).get(key)
    if not m:
        return None
    rows = [[f["name"], _r(f.get("value"), 5), f.get("detail", "")] for f in m.get("features", [])]
    rows.sort(key=lambda r: -(r[1] or 0))
    return {"kind": "table", "heading": f"Measured worth of each input ({m.get('method', 'importance')})",
            "columns": ["Input", m.get("metric", "value"), "What was removed / shuffled"],
            "rows": rows[:12],
            "note": f"Baseline log loss {m.get('baseline')} on n={m.get('n')} ({m.get('window', '')}). "
                    f"A positive value means the model gets worse without that input.",
            "source": _rel(IMPORTANCE_JSON), "as_of": (imp.get("generated_at") or "")[:10]}


def _doc_section(path: Path, heading_regex: str, max_lines: int = 40) -> list[str]:
    """Bullet lines from one section of a markdown doc, for the feature
    list, so the draft quotes the maintained description rather than a
    second copy that can drift."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    m = re.search(heading_regex, text)
    if not m:
        return []
    body = text[m.end():]
    nxt = re.search(r"\n## ", body)
    body = body[:nxt.start()] if nxt else body
    out: list[str] = []
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("* ") or s.startswith("- "):
            if len(out) >= max_lines:
                break
            out.append(re.sub(r"[*`_]", "", s[2:]))
        elif s and out and line.startswith("  "):
            # a wrapped bullet continues on indented lines
            out[-1] += " " + re.sub(r"[*`_]", "", s)
        elif not s and out and len(out) >= max_lines:
            break
    return out


# ---------------------------------------------------------------------------
# binary-outcome ledgers (NFL, CFB)
# ---------------------------------------------------------------------------
def _binary_windows(g: pd.DataFrame, run_date: str) -> list[list]:
    rows = []
    for days, label in WINDOWS:
        w = _window(g, run_date, days)
        n = int(len(w))
        if n == 0:
            rows.append([label, 0, None, None, None, None, None, None])
            continue
        d = pd.to_numeric(w["d_ll"], errors="coerce")
        rows.append([label, n, _pct(w["pick_correct"].astype(bool).mean()),
                     _r(w["log_loss"].mean()), _r(w["brier"].mean()),
                     _r(w["home_log_loss"].mean()),
                     _r(d.mean(), 4), _r(_se(d), 4)])
    return rows


def _binary_calibration(g: pd.DataFrame) -> list[list]:
    rows = []
    p = pd.to_numeric(g["pick_prob"], errors="coerce")
    hit = g["pick_correct"].astype(bool)
    for lo, hi in CAL_BINS:
        m = (p >= lo) & (p < hi)
        n = int(m.sum())
        rows.append([f"{int(lo * 100)}-{min(int(hi * 100), 100)}%", n,
                     _pct(p[m].mean()) if n else None, _pct(hit[m].mean()) if n else None,
                     _r((hit[m].mean() - p[m].mean()) * 100, 1) if n else None])
    return rows


def _binary_misses(g: pd.DataFrame, run_date: str, home_pts: str, away_pts: str) -> tuple[list, list]:
    w = _window(g, run_date, 30)
    if w.empty:
        return [], []
    w = w.copy()
    w["ll"] = pd.to_numeric(w["log_loss"], errors="coerce")
    w["pp"] = pd.to_numeric(w["pick_prob"], errors="coerce")

    def row(r):
        return [r["date"], f"{r['away_team']} @ {r['home_team']}", r["pick"], _pct(r["pp"], 0),
                f"{int(r[home_pts])}-{int(r[away_pts])}", _r(r["ll"], 2)]
    misses = [row(r) for _, r in w[~w["pick_correct"].astype(bool)].nlargest(TOP_N, "ll").iterrows()]
    hits = [row(r) for _, r in w[w["pick_correct"].astype(bool)].nsmallest(TOP_N, "pp").iterrows()]
    return misses, hits


def _binary_segments(g: pd.DataFrame, run_date: str, extra: dict[str, pd.Series] | None = None) -> list[list]:
    season = g
    segs: dict[str, pd.Series] = {
        "Home-side picks": season["pick"] == season["home_team"],
        "Away-side picks": season["pick"] != season["home_team"],
        "Picks at 50-65%": pd.to_numeric(season["pick_prob"], errors="coerce") < 0.65,
        "Picks at 65-80%": pd.to_numeric(season["pick_prob"], errors="coerce").between(0.65, 0.8, inclusive="left"),
        "Picks at 80%+": pd.to_numeric(season["pick_prob"], errors="coerce") >= 0.8,
    }
    if "neutral" in season:
        segs["Neutral site"] = season["neutral"].astype(str).str.lower().isin(["true", "1"])
    segs.update(extra or {})
    rows = []
    for label, mask in segs.items():
        sub = season[mask.fillna(False)]
        n = int(len(sub))
        rows.append([label, n, _pct(sub["pick_correct"].astype(bool).mean()) if n else None,
                     _r(sub["log_loss"].mean()) if n else None,
                     _r(pd.to_numeric(sub["d_ll"], errors="coerce").mean()) if n else None])
    return rows


def _binary_improvements(windows: list[list], cal: list[list], segs: list[list], feeds: dict,
                         stage: dict) -> list[str]:
    out = []
    season = windows[-1]
    if season[1] and season[6] is not None and season[7]:
        z = season[6] / season[7] if season[7] else 0
        if z > -2:
            out.append(f"Edge over always-home is not clear at 2 SE on the season (delta log loss "
                       f"{season[6]:+.4f} ± {season[7]:.4f}, n={season[1]}): the honest baseline comparison "
                       f"to keep in view before adding inputs.")
    for b in cal:
        if b[1] and b[1] >= 15 and b[4] is not None and abs(b[4]) >= 8:
            direction = "under" if b[4] > 0 else "over"
            out.append(f"Calibration: picks quoted at {b[0]} hit {b[3]} (n={b[1]}, mean quote {b[2]}), "
                       f"{abs(b[4]):.0f} points {direction}-confident. Candidate: a calibration layer "
                       f"on the pick probability, measured on held-out weeks first.")
    by_label = {r[0]: r for r in segs}
    h, a = by_label.get("Home-side picks"), by_label.get("Away-side picks")
    if h and a and h[1] >= 15 and a[1] >= 15 and h[2] and a[2]:
        hv, av = float(h[2].rstrip("%")), float(a[2].rstrip("%"))
        if abs(hv - av) >= 12:
            out.append(f"Segment split: home-side picks {h[2]} (n={h[1]}) vs away-side picks {a[2]} "
                       f"(n={a[1]}). Candidate: re-check the home edge and rest handling on the side that lags.")
    for key, f in (feeds or {}).items():
        if f and not f.get("fresh", True):
            out.append(f"Feed {f.get('source', key)} is stale (newest {f.get('newest')}, "
                       f"{f.get('age_days')} days old): the second stage falls back to Elo alone while it "
                       f"is out. Candidate: fix or replace the feed before judging the model on these weeks.")
    if stage and not stage.get("on", True):
        out.append("The efficiency second stage was off on the latest run (slate `model` column = elo): "
                   "picks are pure Elo until its feed is current.")
    return out


# ---------------------------------------------------------------------------
# NFL and CFB builders
# ---------------------------------------------------------------------------
def _football_draft(key: str, run_date: str) -> dict:
    cfg = {
        "nfl": dict(title="NFL Elo + efficiency", site=SITE / "nfl", grades=REPO / "data" / "nfl" / "predictions" / "grades.csv",
                    workflows=["nfl-daily.yml", "weekly-nfl-report.yml"], home_pts="home_score", away_pts="away_score",
                    doc_re=r"## Football model features — NFL", imp_key="nfl",
                    futures_cols=[("p_playoffs", "Playoffs"), ("p_sb", "Super Bowl")], rating_pre="preseason_elo",
                    conf_key="division"),
        "cfb": dict(title="College football Elo + efficiency", site=SITE / "cfb", grades=REPO / "data" / "college_football" / "predictions" / "grades.csv",
                    workflows=["cfb-daily.yml"], home_pts="home_points", away_pts="away_points",
                    doc_re=r"## Football model features — CFB", imp_key="cfb",
                    futures_cols=[("p_conf_title", "Conf. title"), ("p_undefeated", "Undefeated")], rating_pre=None,
                    conf_key="conference"),
    }[key]
    latest_p = cfg["site"] / "latest.json"
    latest = _read_json(latest_p) or {}
    grades_p = cfg["grades"]
    g = pd.read_csv(grades_p) if grades_p.exists() else pd.DataFrame()
    if not g.empty:
        g = g[g["date"] <= run_date]
    ledger = latest.get("ledger") or {}
    feeds = latest.get("feeds") or {}
    stage = latest.get("second_stage") or {}
    site_run = latest.get("run_date")
    last_graded = str(g["date"].max()) if len(g) else None
    graded_age = _days_between(last_graded, run_date) if last_graded else None
    site_age = _days_between(site_run, run_date) if site_run else None

    # ---- status
    run_rows, run_notes = _run_health(cfg["workflows"])
    feed_rows = [[f.get("source", k), f.get("newest"), f.get("age_days"), f.get("tolerance_days"),
                  "fresh" if f.get("fresh") else "STALE"] for k, f in feeds.items()]
    problems = list(run_notes)
    stale_feeds = [k for k, f in feeds.items() if f and not f.get("fresh", True)]
    if site_age is None:
        state, problems = "incomplete", problems + ["no site JSON (latest.json missing)"]
    elif site_age > 3:
        state = "stale"
        problems.append(f"site data last run {site_run} ({site_age} days before this draft)")
    elif graded_age is not None and graded_age > 10:
        state = "stale"
        problems.append(f"nothing graded since {last_graded} ({graded_age} days)")
    else:
        state = "live"
    if stale_feeds:
        problems.append(f"stale feed(s): {', '.join(stale_feeds)} - second stage falls back to Elo")
        if state == "live":
            state = "incomplete"
    status = {
        "state": state,
        "summary": (f"{cfg['title']}: last pipeline run {site_run}, last graded game {last_graded}, "
                    f"{ledger.get('graded', 0)} graded this season; second stage "
                    f"{'on' if stage.get('on') else 'off'} ({stage.get('model', 'elo')})."),
        "source": _rel(latest_p), "as_of": site_run,
        "blocks": [
            {"kind": "kv", "heading": "Pipeline", "rows": [
                ["Verdict", state],
                ["Last pipeline run (site JSON)", site_run],
                ["Last graded game", last_graded],
                ["Season / week on the site", f"{latest.get('season')} / {latest.get('week_label') or latest.get('week')}"],
                ["Pick model on latest slate", stage.get("model", "elo")],
                ["Second-stage training rows", stage.get("n_train")],
            ], "source": _rel(latest_p), "as_of": site_run},
            {"kind": "table", "heading": "Workflow run health (last 10 runs each)",
             "columns": ["Workflow", "Last conclusion", "Last run (UTC)", "Last success (UTC)", "Failed/completed"],
             "rows": run_rows, "source": _rel(RUNS_DIR) + "/<workflow>.json (gh run list)", "as_of": run_date},
            {"kind": "table", "heading": "Feed freshness",
             "columns": ["Feed", "Newest row", "Age (days)", "Tolerance", "State"], "rows": feed_rows,
             "source": _rel(latest_p) + " → feeds", "as_of": site_run},
            {"kind": "bullets", "heading": "Known failures and caveats", "items": problems or ["None detected from the committed files."]},
        ],
    }

    # ---- features
    params = latest.get("params") or {}
    feat_lines = _doc_section(DOCS_FEATURES, cfg["doc_re"])
    features = {
        "source": _rel(DOCS_FEATURES) + " and " + _rel(latest_p), "as_of": _mtime(DOCS_FEATURES),
        "blocks": [
            {"kind": "kv", "heading": "Model", "rows": [
                ["Type", f"Betting-blind Elo, with a random-forest second stage ({stage.get('model', 'elo')}) "
                         f"over Elo plus efficiency features for the pick probability"],
                ["Target", "Home win (binary); picks are the side at >= 50% from the pick model"],
                ["Training window", "NFL: replay 2002 → last completed week, ties excluded" if key == "nfl"
                                    else "CFB: replay 2005 → last completed week"],
                ["Second-stage inputs", f"{len(stage.get('features') or [])} columns: "
                                        + ", ".join((stage.get("features") or [])[:12])
                                        + (" …" if len(stage.get("features") or []) > 12 else "")],
                ["Tuned Elo parameters", ", ".join(f"{k}={v}" for k, v in params.items())],
            ]},
            {"kind": "bullets", "heading": "Inputs (from docs/MODEL_FEATURES.md)", "items": feat_lines[:14]},
        ],
    }
    imp = _importance_block(cfg["imp_key"])
    if imp:
        features["blocks"].append(imp)
    else:
        features["blocks"].append({"kind": "note", "text": f"No committed importance study for '{cfg['imp_key']}' "
                                   f"in {_rel(IMPORTANCE_JSON)}; the second stage's feature list above is what it sees."})

    # ---- report card
    if g.empty:
        windows, cal, cal30, segs, misses, hits = [], [], [], [], [], []
    else:
        windows = _binary_windows(g, run_date)
        cal = _binary_calibration(g)
        cal30 = _binary_calibration(_window(g, run_date, 30))
        extra = {}
        if key == "cfb" and "home_fcs" in g:
            fcs = g["home_fcs"].astype(str).str.lower().eq("true") | g["away_fcs"].astype(str).str.lower().eq("true")
            extra = {"FBS vs FBS only": ~fcs, "Games with an FCS side": fcs}
        if key == "nfl" and "game_type" in g:
            extra = {"Divisional games": g.get("div_game", pd.Series(False, index=g.index)).astype(str).str.lower().eq("true")} if "div_game" in g else {}
        segs = _binary_segments(g, run_date, extra)
        misses, hits = _binary_misses(g, run_date, cfg["home_pts"], cfg["away_pts"])
    report = {
        "summary": (f"Season: {ledger.get('correct')}/{ledger.get('graded')} picks correct "
                    f"({_pct(ledger.get('accuracy'))}), log loss {ledger.get('log_loss')}, Brier {ledger.get('brier')}, "
                    f"always-home log loss {ledger.get('home_log_loss')}, paired delta {ledger.get('d_ll_mean')} "
                    f"± {ledger.get('d_ll_se')} (negative = beating the baseline)."),
        "source": _rel(grades_p), "as_of": last_graded,
        "blocks": [
            {"kind": "table", "heading": "Windows (by game date)",
             "columns": ["Window", "n", "Accuracy", "Log loss", "Brier", "Always-home LL", "Δ LL", "± SE"],
             "rows": windows, "note": "Δ LL is the paired per-game log-loss difference against the fixed "
             "always-pick-home forecast (p = 0.55 NFL / 0.632 CFB, coin flip at neutral sites); negative beats it."},
            {"kind": "table", "heading": "Calibration, season (pick probability vs hit rate)",
             "columns": ["Quoted", "n", "Mean quote", "Hit rate", "Hit − quote (pts)"], "rows": cal},
            {"kind": "table", "heading": "Calibration, last 30 days",
             "columns": ["Quoted", "n", "Mean quote", "Hit rate", "Hit − quote (pts)"], "rows": cal30},
            {"kind": "table", "heading": "Segments, season",
             "columns": ["Segment", "n", "Accuracy", "Log loss", "Δ LL vs always-home"], "rows": segs},
        ],
    }
    by_week = ledger.get("by_week") or {}
    if by_week:
        report["blocks"].append({"kind": "table", "heading": "By week (site ledger)",
                                 "columns": ["Week", "n", "Correct", "Accuracy", "Log loss", "Δ LL"],
                                 "rows": [[w, v.get("graded"), v.get("correct"), _pct(v.get("accuracy")),
                                           v.get("log_loss"), v.get("d_ll_mean")] for w, v in by_week.items()],
                                 "source": _rel(latest_p) + " → ledger.by_week", "as_of": site_run})
    if key == "cfb" and ledger.get("fbs_only"):
        f = ledger["fbs_only"]
        report["blocks"].insert(1, {"kind": "kv", "heading": "FBS vs FBS only (site ledger)", "rows": [
            ["n", f.get("graded")], ["Accuracy", _pct(f.get("accuracy"))], ["Log loss", f.get("log_loss")],
            ["Brier", f.get("brier")], ["Δ LL vs always-home", f"{f.get('d_ll_mean')} ± {f.get('d_ll_se')}"]],
            "source": _rel(latest_p) + " → ledger.fbs_only", "as_of": site_run})

    # ---- findings
    ratings = latest.get("ratings") or []
    movers_week = []
    snap_p = _nearest_snapshot(cfg["site"] / "history", run_date)
    snap = _read_json(snap_p) if snap_p else None
    if snap and snap.get("ratings"):
        prev = {r["team"]: r for r in snap["ratings"]}
        for r in ratings:
            p = prev.get(r["team"])
            if p:
                movers_week.append((r["team"], r.get("elo"), p.get("elo"), r.get("elo", 0) - p.get("elo", 0),
                                    r.get("rank"), p.get("rank")))
        movers_week.sort(key=lambda t: -abs(t[3]))
    movers_rows = [[t, _r(e, 1), _r(pe, 1), _r(d, 1), f"{pr} → {rk}"] for t, e, pe, d, rk, pr in movers_week[:8]]
    pre_rows = []
    if cfg["rating_pre"]:
        pre = [(r["team"], r.get("elo"), r.get(cfg["rating_pre"]), (r.get("elo") or 0) - (r.get(cfg["rating_pre"]) or 0))
               for r in ratings if r.get(cfg["rating_pre"]) is not None]
        pre.sort(key=lambda t: -abs(t[3]))
        pre_rows = [[t, _r(e, 1), _r(p, 1), _r(d, 1)] for t, e, p, d in pre[:8]]
    shifts = []
    fut = (latest.get("futures") or {}).get("teams") or []
    if snap and (snap.get("futures") or {}).get("teams"):
        prevf = {t["team"]: t for t in snap["futures"]["teams"]}
        col, label = cfg["futures_cols"][0]
        for t in fut:
            p = prevf.get(t["team"])
            if p and t.get(col) is not None and p.get(col) is not None:
                shifts.append((t["team"], t[col], p[col], t[col] - p[col]))
        shifts.sort(key=lambda s: -abs(s[3]))
    shift_rows = [[t, _pct(now), _pct(then), f"{100 * d:+.1f} pts"] for t, now, then, d in shifts[:8]]
    cal_drift = []
    for s, m in zip(cal, cal30):
        if s[1] and m[1]:
            cal_drift.append([s[0], s[1], s[4], m[1], m[4]])
    findings = {
        "source": f"{_rel(grades_p)}, {_rel(latest_p)}" + (f", {_rel(snap_p)}" if snap_p else ""),
        "as_of": run_date,
        "blocks": [
            {"kind": "table", "heading": "Biggest misses, last 30 days (by log loss)",
             "columns": ["Date", "Game (away @ home)", "Pick", "Quoted", "Final (home-away)", "Log loss"], "rows": misses},
            {"kind": "table", "heading": "Best upset calls, last 30 days (correct picks with the lowest quote)",
             "columns": ["Date", "Game (away @ home)", "Pick", "Quoted", "Final (home-away)", "Log loss"], "rows": hits},
            {"kind": "table", "heading": f"Rating movers since {snap_p.stem if snap_p else 'n/a'} (site snapshot nearest 7 days back)",
             "columns": ["Team", "Elo now", "Elo then", "Δ", "Rank then → now"], "rows": movers_rows},
            {"kind": "table", "heading": f"Forecast shifts: {cfg['futures_cols'][0][1]} vs {snap_p.stem if snap_p else 'n/a'}",
             "columns": ["Team", "Now", "Then", "Shift"], "rows": shift_rows},
            {"kind": "table", "heading": "Calibration drift (season vs last 30 days)",
             "columns": ["Quoted", "n season", "Hit − quote, season", "n 30d", "Hit − quote, 30d"], "rows": cal_drift},
        ],
    }
    if pre_rows:
        findings["blocks"].insert(2, {"kind": "table", "heading": "Rating movers since preseason",
                                      "columns": ["Team", "Elo now", "Preseason", "Δ"], "rows": pre_rows})
    improvements = {
        "summary": "Candidates only, each with the number that raises it; the reviewer weighs them.",
        "blocks": [{"kind": "bullets", "items": _binary_improvements(windows, cal, segs, feeds, stage)
                    or ["No rule-based candidate fired this week: baseline edge, calibration bins and segment splits "
                        "are all inside their thresholds."]}],
    }
    return {
        "model": key, "title": cfg["title"], "date": run_date, "kind": "draft",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": status["summary"],
        "status": status, "features": features, "findings": findings,
        "report_card": report, "improvements": improvements,
        "sources": sorted({_rel(grades_p), _rel(latest_p), _rel(DOCS_FEATURES), _rel(IMPORTANCE_JSON)}),
    }


# ---------------------------------------------------------------------------
# soccer builder (three-way outcome)
# ---------------------------------------------------------------------------
def _soccer_windows(g: pd.DataFrame, run_date: str, base_p: dict[str, float]) -> list[list]:
    rows = []
    for days, label in WINDOWS:
        w = _window(g, run_date, days)
        n = int(len(w))
        if n == 0:
            rows.append([label, 0, None, None, None, None, None, None])
            continue
        brier = w.apply(lambda r: sum((float(r[f"p_{c}"]) - (1.0 if r["outcome"] == c else 0.0)) ** 2
                                      for c in ("H", "D", "A")), axis=1)
        base_ll = w["outcome"].map(lambda o: -math.log(max(base_p.get(o, 1 / 3), 1e-9)))
        d = pd.to_numeric(w["log_loss"], errors="coerce") - base_ll
        rows.append([label, n, _pct(w["pick_correct"].astype(bool).mean()), _r(w["log_loss"].mean()),
                     _r(brier.mean()), _r(base_ll.mean()), _r(d.mean()), _r(_se(d))])
    return rows


def _soccer_draft(run_date: str) -> dict:
    latest_p = SITE / "soccer" / "latest.json"
    latest = _read_json(latest_p) or {}
    grades_p = REPO / "data" / "soccer_clubs" / "predictions" / "grades.csv"
    g = pd.read_csv(grades_p) if grades_p.exists() else pd.DataFrame()
    if not g.empty:
        g = g[g["date"] <= run_date]
    results_p = REPO / "soccer" / "clubs" / "data" / "results.csv"
    overlay_p = REPO / "soccer" / "clubs" / "data" / "results_overlay_report.json"
    overlay = _read_json(overlay_p) or {}
    ledger = latest.get("ledger") or {}
    feeds = latest.get("feeds") or {}
    fresh = latest.get("results_freshness") or {}
    site_run = latest.get("run_date")
    site_age = _days_between(site_run, run_date) if site_run else None
    last_graded = str(g["date"].max()) if len(g) else None
    graded_age = _days_between(last_graded, run_date) if last_graded else None
    stale_leagues = fresh.get("stale_leagues") or []

    # ---- status
    run_rows, run_notes = _run_health(["soccer-daily.yml"])
    problems = list(run_notes)
    league_rows = []
    for k, e in (fresh.get("leagues") or {}).items():
        league_rows.append([e.get("name", k), e.get("newest_played"), e.get("played_season"),
                            e.get("fixtures_season"), e.get("overdue"), "STALE" if e.get("stale") else (e.get("reason") or "ok")])
    if site_age is None:
        state = "incomplete"
        problems.append("no site JSON (latest.json missing)")
    elif site_age > 3:
        state = "stale"
        problems.append(f"site data last run {site_run} ({site_age} days before this draft)")
    elif not fresh:
        state = "incomplete"
        problems.append("latest.json carries no results_freshness block (written by a pipeline older than "
                        "the freshness export); staleness cannot be judged from it")
    elif stale_leagues:
        state = "stale"
        for k in stale_leagues:
            e = (fresh.get("leagues") or {}).get(k, {})
            problems.append(f"{e.get('name', k)} results stale: {e.get('reason')}")
    elif graded_age is not None and graded_age > 10:
        state = "stale"
        problems.append(f"nothing graded since {last_graded} ({graded_age} days)")
    else:
        state = "live"
    stale_feeds = [k for k, f in feeds.items() if f and not f.get("fresh", True)]
    if stale_feeds:
        problems.append(f"stale form feed(s): {', '.join(stale_feeds)} (feature zeroed by its guard)")
    season_start = f"{int(run_date[:4]) - (0 if int(run_date[5:7]) >= 7 else 1)}-07-01"
    current_conflicts = [c for c in (overlay.get("conflicts") or []) if str(c.get("date", "")) >= season_start]
    for c in current_conflicts[-5:]:
        problems.append(f"results overlay conflict ({c.get('kind')}): {c.get('league')} {c.get('date')} "
                        f"{c.get('home_team')} v {c.get('away_team')} "
                        f"openfootball {c.get('primary_score')} vs {c.get('source')} {c.get('source_score')}")
    status = {
        "state": state,
        "summary": (f"Club soccer: last pipeline run {site_run}, results as of {fresh.get('as_of')}, last graded match "
                    f"{last_graded}, {ledger.get('graded', 0)} graded this season"
                    + (f"; stale leagues: {', '.join(stale_leagues)}" if stale_leagues else "; no league stale") + "."),
        "source": _rel(latest_p), "as_of": site_run,
        "blocks": [
            {"kind": "kv", "heading": "Pipeline", "rows": [
                ["Verdict", state], ["Last pipeline run (site JSON)", site_run],
                ["Results as of (newest final, openfootball leagues)", fresh.get("as_of")],
                ["Last graded match", last_graded],
                ["Overlay (last run)", ", ".join(f"{k}: filled {v.get('filled')}, agree {v.get('agree')}, conflicts {v.get('conflicts')}"
                                                 for k, v in (overlay.get("sources") or {}).items()) or "no report"],
            ], "source": f"{_rel(latest_p)}, {_rel(overlay_p)}", "as_of": site_run},
            {"kind": "table", "heading": "Workflow run health (last 10 runs)",
             "columns": ["Workflow", "Last conclusion", "Last run (UTC)", "Last success (UTC)", "Failed/completed"],
             "rows": run_rows, "source": _rel(RUNS_DIR) + "/soccer-daily.yml.json (gh run list)", "as_of": run_date},
            {"kind": "table", "heading": "Results freshness by league",
             "columns": ["League", "Newest final", "Played", "Fixtures", "Overdue", "State"], "rows": league_rows,
             "source": _rel(latest_p) + " → results_freshness", "as_of": site_run,
             "note": "Overdue = fixtures past their date with no score; a league is stale above the configured limit "
                     "(soccer/clubs/daily/config.py)."},
            {"kind": "table", "heading": "Form feed freshness",
             "columns": ["Feed", "Newest row", "Age (days)", "Tolerance", "State"],
             "rows": [[f.get("source", k), f.get("newest"), f.get("age_days"), f.get("tolerance_days"),
                       "fresh" if f.get("fresh") else "STALE"] for k, f in feeds.items()],
             "source": _rel(latest_p) + " → feeds", "as_of": site_run},
            {"kind": "bullets", "heading": "Known failures and caveats", "items": problems or ["None detected from the committed files."]},
        ],
    }

    # ---- features
    feat_lines = _doc_section(DOCS_FEATURES, r"## Soccer model features")
    features = {
        "source": _rel(DOCS_FEATURES), "as_of": _mtime(DOCS_FEATURES),
        "blocks": [
            {"kind": "kv", "heading": "Model", "rows": [
                ["Type", "Glued club Elo (one pool per country, UEFA cross-play) + random-forest outcome model over Elo, "
                         "squad economics, rolling xG / shots form and the Understat advanced layer"],
                ["Target", "Three-way match outcome (home win / draw / away win); pick = the likeliest class"],
                ["Training window", "Replay history across all pools (2010-11 onward by league); outcome model refit in-run daily"],
                ["Leagues", ", ".join(k for k in (latest.get("ratings") or {}).keys())],
            ]},
            {"kind": "bullets", "heading": "Inputs (from docs/MODEL_FEATURES.md)", "items": feat_lines[:16]},
        ],
    }
    imp = _importance_block("soccer")
    features["blocks"].append(imp or {"kind": "note", "text": f"No committed importance study for 'soccer' in {_rel(IMPORTANCE_JSON)}."})

    # ---- report card
    if g.empty:
        windows, cal, cal30, segs, misses, hits, base_p = [], [], [], [], [], [], {}
    else:
        counts = g["outcome"].value_counts(normalize=True)
        base_p = {c: float(counts.get(c, 0.0)) for c in ("H", "D", "A")}
        windows = _soccer_windows(g, run_date, base_p)
        g2 = g.copy()
        g2["pick_prob"] = g2.apply(lambda r: float(r[f"p_{r['pick']}"]), axis=1)
        cal_bins = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]
        cal, cal30 = [], []
        for src, dst in ((g2, cal), (_window(g2, run_date, 30), cal30)):
            p, hit = src["pick_prob"], src["pick_correct"].astype(bool)
            for lo, hi in cal_bins:
                m = (p >= lo) & (p < hi)
                n = int(m.sum())
                dst.append([f"{int(lo * 100)}-{min(int(hi * 100), 100)}%", n, _pct(p[m].mean()) if n else None,
                            _pct(hit[m].mean()) if n else None, _r((hit[m].mean() - p[m].mean()) * 100, 1) if n else None])
        segs = []
        for label, mask in {"Home-win picks": g2["pick"] == "H", "Draw picks": g2["pick"] == "D",
                            "Away-win picks": g2["pick"] == "A"}.items():
            sub = g2[mask]
            segs.append([label, int(len(sub)), _pct(sub["pick_correct"].astype(bool).mean()) if len(sub) else None,
                         _r(sub["log_loss"].mean()) if len(sub) else None])
        for lg, sub in g2.groupby("league"):
            segs.append([f"League: {lg}", int(len(sub)), _pct(sub["pick_correct"].astype(bool).mean()), _r(sub["log_loss"].mean())])
        w30 = _window(g2, run_date, 30)
        def row(r):
            return [r["date"], r["league"], f"{r['home_team']} v {r['away_team']}", r["pick"], _pct(r["pick_prob"], 0),
                    f"{int(r['home_score'])}-{int(r['away_score'])}", _r(r["log_loss"], 2)]
        misses = [row(r) for _, r in w30[~w30["pick_correct"].astype(bool)].nlargest(TOP_N, "log_loss").iterrows()]
        hits = [row(r) for _, r in w30[w30["pick_correct"].astype(bool)].nsmallest(TOP_N, "pick_prob").iterrows()]
    report = {
        "summary": (f"Season: {ledger.get('graded', 0)} graded, pick accuracy {_pct(ledger.get('accuracy'))}, "
                    f"log loss {ledger.get('log_loss')} (three-way, so not comparable to the binary sports). "
                    f"Baseline = season outcome frequencies H/D/A "
                    + "/".join(_pct(base_p.get(c), 0) or "—" for c in ("H", "D", "A")) + "."),
        "source": _rel(grades_p), "as_of": last_graded,
        "blocks": [
            {"kind": "table", "heading": "Windows (by match date)",
             "columns": ["Window", "n", "Pick accuracy", "Log loss", "Brier (3-way)", "Frequency-baseline LL", "Δ LL", "± SE"],
             "rows": windows, "note": "Δ LL is the per-match log-loss difference against a fixed forecast equal to the "
             "season's own H/D/A frequencies (in-sample for the baseline, so it flatters the baseline slightly); negative beats it."},
            {"kind": "table", "heading": "Calibration, season (pick probability vs hit rate)",
             "columns": ["Quoted", "n", "Mean quote", "Hit rate", "Hit − quote (pts)"], "rows": cal},
            {"kind": "table", "heading": "Calibration, last 30 days",
             "columns": ["Quoted", "n", "Mean quote", "Hit rate", "Hit − quote (pts)"], "rows": cal30},
            {"kind": "table", "heading": "Segments, season", "columns": ["Segment", "n", "Pick accuracy", "Log loss"], "rows": segs},
        ],
    }
    rolling = ledger.get("rolling") or {}
    if rolling:
        report["blocks"].append({"kind": "kv", "heading": "Site ledger rolling windows",
                                 "rows": [[k, f"n={v.get('graded')}, accuracy {_pct(v.get('accuracy'))}, log loss {v.get('log_loss')}"]
                                          for k, v in rolling.items()],
                                 "source": _rel(latest_p) + " → ledger.rolling", "as_of": site_run})

    # ---- findings
    snap_p = _nearest_snapshot(SITE / "soccer" / "history", run_date)
    snap = _read_json(snap_p) if snap_p else None
    movers = []
    if snap and snap.get("ratings"):
        for lg, block in (latest.get("ratings") or {}).items():
            prev = {c["team"]: c["elo"] for c in (snap["ratings"].get(lg) or {}).get("clubs", [])}
            for c in block.get("clubs", []):
                if c["team"] in prev:
                    movers.append((lg, c["team"], c["elo"], prev[c["team"]], c["elo"] - prev[c["team"]]))
        movers.sort(key=lambda t: -abs(t[4]))
    mover_rows = [[lg, t, _r(e, 1), _r(p, 1), _r(d, 1)] for lg, t, e, p, d in movers[:10]]
    shifts = []
    if snap and snap.get("futures"):
        for lg, sim in (latest.get("futures") or {}).items():
            prev = {c["team"]: c for c in (snap["futures"].get(lg) or {}).get("clubs", [])}
            for c in sim.get("clubs", []) if isinstance(sim, dict) else []:
                p = prev.get(c["team"])
                if p:
                    for col in ("p_title", "p_top4", "p_relegation"):
                        if c.get(col) is not None and p.get(col) is not None:
                            shifts.append((lg, c["team"], col, c[col], p[col], c[col] - p[col]))
        shifts.sort(key=lambda s: -abs(s[5]))
    shift_rows = [[lg, t, col, _pct(now), _pct(then), f"{100 * d:+.1f} pts"] for lg, t, col, now, then, d in shifts[:10]]
    cal_drift = [[s[0], s[1], s[4], m[1], m[4]] for s, m in zip(cal, cal30) if s[1] and m[1]]
    findings = {
        "source": f"{_rel(grades_p)}, {_rel(latest_p)}" + (f", {_rel(snap_p)}" if snap_p else ""), "as_of": run_date,
        "blocks": [
            {"kind": "table", "heading": "Biggest misses, last 30 days (by log loss)",
             "columns": ["Date", "League", "Match", "Pick", "Quoted", "Final", "Log loss"], "rows": misses},
            {"kind": "table", "heading": "Best low-confidence calls, last 30 days (correct picks with the lowest quote)",
             "columns": ["Date", "League", "Match", "Pick", "Quoted", "Final", "Log loss"], "rows": hits},
            {"kind": "table", "heading": f"Rating movers since {snap_p.stem if snap_p else 'n/a'}",
             "columns": ["League", "Club", "Elo now", "Elo then", "Δ"], "rows": mover_rows},
            {"kind": "table", "heading": f"Forecast shifts vs {snap_p.stem if snap_p else 'n/a'}",
             "columns": ["League", "Club", "Odds", "Now", "Then", "Shift"], "rows": shift_rows},
            {"kind": "table", "heading": "Calibration drift (season vs last 30 days)",
             "columns": ["Quoted", "n season", "Hit − quote, season", "n 30d", "Hit − quote, 30d"], "rows": cal_drift},
        ],
    }
    if stale_leagues:
        findings["blocks"].insert(0, {"kind": "warn", "text": f"Stale leagues ({', '.join(stale_leagues)}) are frozen at their "
                                      f"newest final; their movers, shifts and grades below are not current."})
    cands = []
    if windows and windows[-1][1] and windows[-1][6] is not None and windows[-1][7]:
        if windows[-1][6] / windows[-1][7] > -2:
            cands.append(f"Edge over the frequency baseline is not clear at 2 SE on the season (Δ LL {windows[-1][6]:+.4f} "
                         f"± {windows[-1][7]:.4f}, n={windows[-1][1]}).")
    for b in cal:
        if b[1] >= 20 and b[4] is not None and abs(b[4]) >= 8:
            cands.append(f"Calibration: picks quoted at {b[0]} hit {b[3]} (n={b[1]}), {abs(b[4]):.0f} points "
                         f"{'under' if b[4] > 0 else 'over'}-confident. Candidate: calibration layer, measured out of sample first.")
    draws = next((s for s in segs if s[0] == "Draw picks"), None)
    if draws and draws[1] == 0 and len(g):
        cands.append("The model never picks the draw (0 draw picks this season) although draws are "
                     f"{_pct(base_p.get('D'))} of outcomes: expected for an argmax pick, worth knowing when reading accuracy.")
    for s in segs:
        if s[0].startswith("League:") and s[1] >= 40 and s[2] and float(s[2].rstrip('%')) < 45:
            cands.append(f"{s[0][8:]}: pick accuracy {s[2]} on n={s[1]}, below the other leagues. Candidate: check that "
                         f"league's results freshness and feature coverage before changing anything.")
    for k in stale_leagues:
        cands.append(f"{(fresh.get('leagues') or {}).get(k, {}).get('name', k)}: results stale; nothing about this league "
                     f"can be judged until its source (fd_results from football-data.co.uk) lands.")
    for k in stale_feeds:
        cands.append(f"Form feed {k} is stale; the feature is zeroed by its guard until it is current.")
    improvements = {"summary": "Candidates only, each with the number that raises it; the reviewer weighs them.",
                    "blocks": [{"kind": "bullets", "items": cands or ["No rule-based candidate fired this week."]}]}
    return {
        "model": "soccer", "title": "Club soccer Elo + outcome model", "date": run_date, "kind": "draft",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": status["summary"], "status": status, "features": features, "findings": findings,
        "report_card": report, "improvements": improvements,
        "sources": sorted({_rel(grades_p), _rel(latest_p), _rel(results_p), _rel(overlay_p), _rel(DOCS_FEATURES)}),
    }


# ---------------------------------------------------------------------------
# basketball: odds data, no model
# ---------------------------------------------------------------------------
def _basketball_draft(run_date: str) -> dict:
    nba_dir = REPO / "data" / "nba"
    files = sorted(p for p in nba_dir.glob("nba_odds_api_data_*.csv") if "latest" not in p.name)
    latest_p = nba_dir / "nba_odds_api_data_latest.csv"
    stamps = []
    for p in files:
        m = re.search(r"(\d{4}-\d{2}-\d{2})-(\d{4})", p.name)
        if m:
            stamps.append(m.group(1))
    latest = pd.read_csv(latest_p) if latest_p.exists() else pd.DataFrame()
    pulled = str(latest["Timestamp Pulled"].max())[:16] if "Timestamp Pulled" in latest and len(latest) else None
    pulled_age = _days_between(pulled, run_date) if pulled else None
    books = sorted({c.split(" ")[1] for c in latest.columns if c.startswith("Home ") and c.endswith(" H2H Odds")
                    and "Avg" not in c}) if len(latest) else []
    summary = _read_json(SUMMARY_JSON) or {}
    nba_sum = next((m for m in summary.get("models", []) if m.get("sport") == "NBA"), {})
    slate = _read_json(SLATE_JSON) or {}
    nba_slate = next((s for s in slate.get("sports", []) if s.get("key") == "nba"), {})
    actual = sorted((nba_dir / "actual_games").glob("*.csv"))
    run_rows, run_notes = _run_health(["unified-odds.yml"])
    state = "none"
    problems = list(run_notes)
    if pulled_age is None:
        problems.append("no NBA odds snapshot on disk")
    elif pulled_age > 2:
        problems.append(f"newest NBA odds snapshot is {pulled_age} days old ({pulled})")
    odds_rows = []
    if len(latest):
        for _, r in latest.head(12).iterrows():
            p = r.get("Avg Home H2H Odds")
            try:
                p = float(p)
                prob = (-p / (-p + 100)) if p < 0 else (100 / (p + 100))
            except (TypeError, ValueError):
                prob = None
            odds_rows.append([str(r.get("Date of Game (ET)", ""))[:16], f"{r.get('Away Team')} @ {r.get('Home Team')}",
                              r.get("Avg Home Spread Points"), r.get("Avg Total Points"), _pct(prob)])
    needs = _doc_section(DOCS_SOURCES, r"## NBA — nothing exists yet", max_lines=12)
    status = {
        "state": state,
        "summary": ("Basketball has no predictive model in this repo. What exists is the NBA half of the unified odds "
                    f"snapshot: {len(files)} snapshot files from {stamps[0] if stamps else '—'} to {stamps[-1] if stamps else '—'}, "
                    f"newest pull {pulled}, {len(latest)} games in the latest file across {len(books)} books."),
        "source": _rel(nba_dir), "as_of": pulled,
        "blocks": [
            {"kind": "warn", "text": "No NBA model, no picks, no grades. Every number on this page describes the odds feed, "
                                     "not a forecast. Do not read a win probability below as the repo's prediction: it is the "
                                     "bookmakers' consensus."},
            {"kind": "kv", "heading": "What exists", "rows": [
                ["Predictive model", "none (no NBA model folder, no NBA daily workflow)"],
                ["Odds snapshots on disk", f"{len(files)} files, {stamps[0] if stamps else '—'} → {stamps[-1] if stamps else '—'}"],
                ["Newest snapshot pulled (ET)", pulled],
                ["Games in newest snapshot", int(len(latest))],
                ["Bookmakers in newest snapshot", ", ".join(books) or "—"],
                ["Hand-dropped results files", ", ".join(p.name for p in actual) or "none"],
                ["Cross-sport summary entry", f"status {nba_sum.get('status')}, record {nba_sum.get('record') or '—'}, last graded {nba_sum.get('last_graded') or '—'}"],
                ["Site slate block (48h window)", f"{len(nba_slate.get('games', []))} games, snapshot {((nba_slate.get('snapshot') or {}).get('pulled_at_et'))}"],
            ], "source": f"{_rel(nba_dir)}, {_rel(SUMMARY_JSON)}, {_rel(SLATE_JSON)}", "as_of": pulled},
            {"kind": "table", "heading": "Workflow run health (unified odds, last 10 runs)",
             "columns": ["Workflow", "Last conclusion", "Last run (UTC)", "Last success (UTC)", "Failed/completed"],
             "rows": run_rows, "source": _rel(RUNS_DIR) + "/unified-odds.yml.json (gh run list)", "as_of": run_date},
            {"kind": "bullets", "heading": "Known failures and caveats", "items": problems or ["None detected from the committed files."]},
        ],
    }
    features = {
        "summary": "There is no model to describe. This is what a basketball board would need, from docs/MODEL_SOURCES.md.",
        "source": _rel(DOCS_SOURCES), "as_of": _mtime(DOCS_SOURCES),
        "blocks": [{"kind": "bullets", "heading": "What would be needed to build one", "items": needs or [
            "A results spine (nba_api or Basketball Reference) with dates, scores, home/away and rest days",
            "A betting-blind Elo tuned on held-out seasons, as the other four sports have",
            "Then roster/availability inputs, the biggest signal in basketball"]}],
    }
    findings = {
        "summary": "Odds-feed observations only.",
        "source": _rel(latest_p), "as_of": pulled,
        "blocks": [{"kind": "table", "heading": "Newest snapshot: consensus lines",
                    "columns": ["Tip-off (ET)", "Game (away @ home)", "Home spread", "Total", "Home win prob (book consensus)"],
                    "rows": odds_rows, "note": "Win probability is the de-vigged-free conversion of the average home moneyline; "
                    "it is the market's number, not a model's."}],
    }
    report = {
        "summary": "No report card: nothing is predicted, so nothing is graded. Accuracy, log loss, Brier and calibration do not exist for basketball.",
        "source": _rel(SUMMARY_JSON), "as_of": (summary.get("generated_at") or "")[:10],
        "blocks": [{"kind": "kv", "rows": [["Graded games", 0], ["Accuracy", None], ["Log loss", None], ["Brier", None]]}],
    }
    improvements = {
        "summary": "The only honest candidate is whether to build a model at all.",
        "blocks": [{"kind": "bullets", "items": [
            "Decide whether basketball stays an odds-tracking page or gets a model (open question for TW).",
            f"If it gets one: the odds archive already holds {len(files)} snapshots since {stamps[0] if stamps else '—'}, "
            "so a closing-line benchmark exists from day one.",
        ]}],
    }
    return {
        "model": "basketball", "title": "Basketball (NBA odds only, no model)", "date": run_date, "kind": "draft",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": status["summary"], "status": status, "features": features, "findings": findings,
        "report_card": report, "improvements": improvements,
        "sources": sorted({_rel(nba_dir), _rel(SUMMARY_JSON), _rel(SLATE_JSON), _rel(DOCS_SOURCES)}),
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
BUILDERS = {
    "nfl": lambda d: _football_draft("nfl", d),
    "cfb": lambda d: _football_draft("cfb", d),
    "soccer": _soccer_draft,
    "basketball": _basketball_draft,
}


def build(model: str, run_date: str) -> dict:
    return BUILDERS[model](run_date)


def write(doc: dict, out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    d = out_dir / doc["model"]
    d.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n"
    html = render(doc)
    (d / f"{doc['date']}.json").write_text(text, encoding="utf-8")
    (d / f"{doc['date']}.html").write_text(html, encoding="utf-8")
    (d / "latest.json").write_text(text, encoding="utf-8")
    (d / "latest.html").write_text(html, encoding="utf-8")
    return d / f"{doc['date']}.json", d / f"{doc['date']}.html"


def check_latest(model: str, today: str, out_dir: Path = OUT_DIR, max_age_hours: int = 24,
                 log=print) -> int:
    """The review session's first command: is `latest.json` for this model
    today's draft? Exit 0 when its `date` equals `today` and it was
    generated within `max_age_hours`; exit 1 (with a plain reason) when
    the file is missing, carries another date, or is older. A stale
    status is reported but is not a failure here - the review must still
    happen and say what is stale."""
    p = out_dir / model / "latest.json"
    doc = _read_json(p)
    if not doc:
        log(f"DRAFT MISSING: {_rel(p)} does not exist or is not readable. Do not review; report this.")
        return 1
    problems = []
    if doc.get("date") != today:
        problems.append(f"draft date is {doc.get('date')}, today is {today}")
    try:
        gen = datetime.fromisoformat(str(doc.get("generated_at")))
        age_h = (datetime.now(timezone.utc) - gen).total_seconds() / 3600
        if age_h > max_age_hours:
            problems.append(f"generated {age_h:.0f} hours ago ({doc.get('generated_at')}), limit {max_age_hours}")
    except (TypeError, ValueError):
        problems.append(f"generated_at unreadable: {doc.get('generated_at')!r}")
    if problems:
        log(f"DRAFT OLD: {_rel(p)}: " + "; ".join(problems) + ". Do not reuse it; report this.")
        return 1
    state = (doc.get("status") or {}).get("state")
    log(f"DRAFT OK: {_rel(p)} dated {doc.get('date')}, status {state}"
        + (" - review it, but lead with what is stale" if state in ("stale", "incomplete") else ""))
    return 0


def models_for(run_date: str, requested: str | None) -> list[str]:
    if requested and requested != "schedule":
        return list(MODELS) if requested == "all" else [requested]
    return list(SCHEDULE.get(date.fromisoformat(run_date).weekday(), ()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="schedule", choices=list(MODELS) + ["all", "schedule"],
                    help="which draft(s) to build; 'schedule' picks by weekday (default)")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--check", action="store_true",
                    help="do not build; verify that --model's latest.json is today's draft (exit 1 otherwise)")
    args = ap.parse_args(argv)
    if args.check:
        if args.model in ("all", "schedule"):
            todo = list(MODELS) if args.model == "all" else models_for(args.date, None)
        else:
            todo = [args.model]
        return max((check_latest(m, args.date, Path(args.out)) for m in todo), default=0)
    todo = models_for(args.date, args.model)
    if not todo:
        print(f"model_review: nothing scheduled for {args.date} (weekday {date.fromisoformat(args.date).strftime('%A')})")
        return 0
    for m in todo:
        doc = build(m, args.date)
        j, h = write(doc, Path(args.out))
        print(f"model_review: {m:<10} status={doc['status']['state']:<10} -> {_rel(j)}, {_rel(h)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
