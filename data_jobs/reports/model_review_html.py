"""The weekly model-review one-pager: one reusable HTML template for every
model.

The draft generator (`model_review.py`) produces a JSON document with
five fixed sections - current status, key features, interesting
findings, report card, potential improvements - each made of simple
blocks (key/value lists, tables, bullet lists, notes) that carry their
own data source and as-of date. This module turns that document into a
single self-contained page: inline CSS only, no JavaScript, no external
fonts or images, readable at phone width, and small enough to travel as
a message attachment. The Claude review session writes its final
assessment into the same template (`render` with the reviewed document),
so the draft and the final review look alike.

Block shapes (all optional keys):
  {"kind": "kv",      "heading", "rows": [[label, value], ...]}
  {"kind": "table",   "heading", "columns": [...], "rows": [[...], ...], "note"}
  {"kind": "bullets", "heading", "items": ["...", ...]}
  {"kind": "note",    "text": "..."}
Every block may carry "source" and "as_of"; the section may too.
"""

from __future__ import annotations

import html as _html
from typing import Any

SECTION_ORDER = [
    ("status", "Current status"),
    ("features", "Key features"),
    ("findings", "Interesting findings"),
    ("report_card", "Report card"),
    ("improvements", "Potential improvements"),
]

STATE_COLORS = {"live": "#1a7f37", "stale": "#b35900", "incomplete": "#8a6d00",
                "none": "#6b6b6b", "unknown": "#6b6b6b"}

CSS = """
:root{color-scheme:light}
body{margin:0;padding:12px;background:#fff;color:#1f2328;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:15px;line-height:1.45}
main{max-width:760px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px 0;line-height:1.2}
h2{font-size:17px;margin:26px 0 6px 0;padding-bottom:4px;border-bottom:2px solid #d0d7de}
h3{font-size:15px;margin:14px 0 4px 0}
p{margin:6px 0}
.sub{color:#57606a;font-size:13px}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;color:#fff;font-size:12px;font-weight:600;vertical-align:middle}
.grade{display:inline-block;min-width:34px;text-align:center;padding:2px 8px;border:2px solid #1f2328;border-radius:6px;font-weight:700;font-size:18px;margin-left:8px}
.src{color:#6b6b6b;font-size:12px;margin:2px 0 6px 0}
.warn{background:#fff4e5;border:1px solid #f0a000;border-radius:6px;padding:8px 10px;margin:8px 0;font-size:14px}
.ok{background:#eaf7ee;border:1px solid #1a7f37;border-radius:6px;padding:8px 10px;margin:8px 0;font-size:14px}
.kv{width:100%;border-collapse:collapse;font-size:14px}
.kv td{padding:4px 6px;border-bottom:1px solid #eaeef2;vertical-align:top}
.kv td:first-child{color:#57606a;width:38%}
.tbl-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:4px 0}
table.tbl{border-collapse:collapse;font-size:13px;min-width:100%}
table.tbl th{text-align:left;padding:5px 7px;border-bottom:2px solid #333;background:#f6f8fa;white-space:nowrap}
table.tbl td{padding:4px 7px;border-bottom:1px solid #e1e4e8;white-space:nowrap}
table.tbl td.num,table.tbl th.num{text-align:right}
ul{margin:4px 0 4px 18px;padding:0}
li{margin:3px 0}
.note{color:#57606a;font-size:13px}
.footer{color:#8b949e;font-size:12px;margin-top:28px;border-top:1px solid #d0d7de;padding-top:8px}
@media (max-width:480px){body{padding:8px;font-size:14px}h1{font-size:19px}.kv td:first-child{width:44%}}
"""


# TW's reviews carry no em dashes (docs/weekly_model_review_prompt.md), and
# the review session may send the rendered page as it stands, so the
# template never emits one: missing values render as a plain hyphen and any
# em dash in quoted text (the feature docs use them) becomes " - ".
MISSING = "-"


def _no_em_dash(text: str) -> str:
    return text.replace("\u2014", " - ").replace("&mdash;", "-")


def _e(x: Any) -> str:
    """Escape for HTML; numbers and None render plainly."""
    if x is None:
        return MISSING
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return _html.escape(f"{x:g}" if abs(x) >= 1000 or x == int(x) else f"{x:.4g}")
    return _html.escape(str(x))


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _source_line(obj: dict) -> str:
    src, asof = obj.get("source"), obj.get("as_of")
    if not src and not asof:
        return ""
    bits = []
    if src:
        bits.append(f"Source: {_e(src)}")
    if asof:
        bits.append(f"as of {_e(asof)}")
    return f"<p class='src'>{' &middot; '.join(bits)}</p>"


def _block(b: dict) -> str:
    kind = b.get("kind", "note")
    out = []
    if b.get("heading"):
        out.append(f"<h3>{_e(b['heading'])}</h3>")
    if kind == "kv":
        rows = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v) if not b.get('raw') else v}</td></tr>"
                       for k, v in b.get("rows", []))
        out.append(f"<table class='kv'>{rows}</table>")
    elif kind == "table":
        cols = b.get("columns", [])
        numeric = b.get("numeric") or [all(_is_num(r[i]) or r[i] is None for r in b.get("rows", []) if i < len(r))
                                       and bool(b.get("rows")) for i in range(len(cols))]
        head = "".join(f"<th class='{'num' if numeric[i] else ''}'>{_e(c)}</th>" for i, c in enumerate(cols))
        body = []
        for r in b.get("rows", []):
            cells = "".join(f"<td class='{'num' if i < len(numeric) and numeric[i] else ''}'>{_e(c)}</td>"
                            for i, c in enumerate(r))
            body.append(f"<tr>{cells}</tr>")
        if not body:
            body.append(f"<tr><td colspan='{max(len(cols), 1)}' class='note'>No rows.</td></tr>")
        out.append(f"<div class='tbl-wrap'><table class='tbl'><tr>{head}</tr>{''.join(body)}</table></div>")
    elif kind == "bullets":
        items = "".join(f"<li>{_e(i)}</li>" for i in b.get("items", []))
        out.append(f"<ul>{items or '<li class=note>None.</li>'}</ul>")
    elif kind == "warn":
        out.append(f"<div class='warn'>{_e(b.get('text', ''))}</div>")
    elif kind == "ok":
        out.append(f"<div class='ok'>{_e(b.get('text', ''))}</div>")
    else:
        out.append(f"<p class='note'>{_e(b.get('text', ''))}</p>")
    if b.get("note"):
        out.append(f"<p class='note'>{_e(b['note'])}</p>")
    out.append(_source_line(b))
    return "".join(out)


def render(doc: dict) -> str:
    """The whole page for one draft/review document."""
    title = doc.get("title") or doc.get("model", "Model review")
    date = doc.get("date", "")
    state = (doc.get("status") or {}).get("state", "unknown")
    color = STATE_COLORS.get(state, "#6b6b6b")
    kind = doc.get("kind", "draft")
    grade = doc.get("grade")
    grade_html = f"<span class='grade'>{_e(grade)}</span>" if grade else ""
    head = (
        f"<h1>{_e(title)}: weekly model review{grade_html}</h1>"
        f"<p class='sub'>{_e(kind).capitalize()} for <b>{_e(date)}</b> &middot; "
        f"status <span class='badge' style='background:{color}'>{_e(state)}</span>"
        + (f" &middot; generated {_e(doc['generated_at'])}" if doc.get("generated_at") else "")
        + "</p>"
    )
    if doc.get("summary"):
        head += f"<p>{_e(doc['summary'])}</p>"
    if doc.get("grade_reason"):
        head += f"<p><b>Grade {_e(grade)}:</b> {_e(doc['grade_reason'])}</p>"
    sections = []
    for key, label in SECTION_ORDER:
        sec = doc.get(key) or {}
        body = [f"<h2 id='{key}'>{_e(label)}</h2>"]
        if sec.get("summary") and sec["summary"] != doc.get("summary"):
            body.append(f"<p>{_e(sec['summary'])}</p>")
        body.append(_source_line(sec))
        blocks = sec.get("blocks") or []
        if not blocks and not sec.get("summary"):
            body.append("<p class='note'>Nothing to report.</p>")
        for b in blocks:
            body.append(_block(b))
        sections.append("".join(body))
    footer = (
        "<p class='footer'>Can Tre Beat Vegas &middot; weekly model review &middot; "
        + (f"draft generated by the repo at {_e(doc.get('generated_at'))}; " if kind == "draft" else "")
        + "every number above carries the file it came from and its as-of date. "
        "Static page, no scripts.</p>"
    )
    return _no_em_dash(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_e(title)} review {_e(date)}</title><style>{CSS}</style></head>"
        f"<body><main>{head}{''.join(sections)}{footer}</main></body></html>"
    )
