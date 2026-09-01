"""HTML renderer for the twice-weekly soccer update email.

One email, three sections, mirroring the MLB renderers' email-client-safe
style (table markup, inline styles only):

1. Games this week — every fixture in the next EMAIL_FIXTURE_DAYS days
   with W/D/L probabilities, the model pick, and the most likely score.
2. Past week + rolling tracker — graded results from the trailing seven
   days plus 7d/30d/season accuracy and log loss.
3. Final-table forecast — per top flight, the Monte Carlo's expected
   finish with Title / UCL / UEL / Relegation probabilities.
"""

from __future__ import annotations

import pandas as pd

from soccer.clubs.data.leagues import LEAGUES

STYLE_TABLE = (
    "border-collapse:collapse;width:100%;max-width:680px;"
    "font-family:Arial,Helvetica,sans-serif;font-size:13px;"
)
STYLE_TH = (
    "text-align:left;padding:5px 8px;border-bottom:2px solid #333;"
    "background:#f4f4f4;white-space:nowrap;"
)
STYLE_TD = "padding:5px 8px;border-bottom:1px solid #ddd;white-space:nowrap;"
STYLE_H3 = "margin:18px 0 4px 0;"


def _league_name(key: str) -> str:
    lg = LEAGUES.get(key)
    return lg.name if lg else key


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%" if x >= 0.005 else "<1%"


def _wrap(title: str, subtitle: str, body: str) -> str:
    return (
        f'<div style="font-family:Arial,Helvetica,sans-serif;color:#222;">'
        f"<h2 style=\"margin-bottom:2px;\">{title}</h2>"
        f'<p style="margin-top:0;color:#666;font-size:13px;">{subtitle}</p>'
        f"{body}"
        f'<p style="color:#999;font-size:11px;margin-top:24px;">'
        f"Glued club Elo (top-5 leagues + UEFA cross-play) with a "
        f"squad-economics outcome model &middot; Can Tre Beat Vegas &middot; "
        f'<a href="https://whosyurgoat.app/vegas/soccer">whosyurgoat.app/vegas/soccer</a></p>'
        f"</div>"
    )


def _table(headers: list[str], rows: list[str]) -> str:
    head = "".join(f"<th style='{STYLE_TH}'>{h}</th>" for h in headers)
    return (
        f"<table style='{STYLE_TABLE}'><tr>{head}</tr>{''.join(rows)}</table>"
    )


def _fixtures_section(fixtures: pd.DataFrame) -> str:
    if fixtures.empty:
        return "<p>No league fixtures in the coming week.</p>"
    parts = []
    for league, sub in fixtures.groupby("league", sort=False):
        rows = []
        for _, m in sub.iterrows():
            pick_team = {
                "H": m["home_team"], "A": m["away_team"], "D": "Draw",
            }[m["pick"]]
            rows.append(
                f"<tr><td style='{STYLE_TD}'>{m['date']}</td>"
                f"<td style='{STYLE_TD}'><b>{m['home_team']}</b> v {m['away_team']}</td>"
                f"<td style='{STYLE_TD}'>{_pct(m['p_H'])} / {_pct(m['p_D'])} / {_pct(m['p_A'])}</td>"
                f"<td style='{STYLE_TD}'><b>{pick_team}</b></td>"
                f"<td style='{STYLE_TD}'>{int(m['score_home'])}&ndash;{int(m['score_away'])}</td></tr>"
            )
        parts.append(
            f"<h3 style='{STYLE_H3}'>{_league_name(league)}</h3>"
            + _table(["Date", "Match", "H / D / A", "Pick", "Sim score"], rows)
        )
    return "".join(parts)


def _record_row(label: str, stats: dict) -> str:
    if not stats or not stats.get("graded"):
        return (
            f"<tr><td style='{STYLE_TD}'>{label}</td>"
            f"<td style='{STYLE_TD}'>0</td><td style='{STYLE_TD}'>&mdash;</td>"
            f"<td style='{STYLE_TD}'>&mdash;</td></tr>"
        )
    return (
        f"<tr><td style='{STYLE_TD}'>{label}</td>"
        f"<td style='{STYLE_TD}'>{stats['graded']}</td>"
        f"<td style='{STYLE_TD}'>{100 * stats['accuracy']:.1f}%</td>"
        f"<td style='{STYLE_TD}'>{stats['log_loss']:.4f}</td></tr>"
    )


def _performance_section(recent: pd.DataFrame, ledger: dict) -> str:
    parts = []

    rolling = ledger.get("rolling", {})
    tracker_rows = [
        _record_row("Last 7 days", rolling.get("7d", {})),
        _record_row("Last 30 days", rolling.get("30d", {})),
        _record_row("Season to date", ledger),
    ]
    parts.append(
        f"<h3 style='{STYLE_H3}'>Rolling tracker</h3>"
        + _table(["Window", "Graded", "Pick accuracy", "Log loss"], tracker_rows)
    )

    by_league = ledger.get("by_league", {})
    if by_league:
        league_rows = [
            _record_row(_league_name(lg), stats)
            for lg, stats in sorted(by_league.items())
        ]
        parts.append(
            f"<h3 style='{STYLE_H3}'>Season record by league</h3>"
            + _table(["League", "Graded", "Pick accuracy", "Log loss"], league_rows)
        )

    if recent.empty:
        parts.append("<p>No matches graded in the past week.</p>")
    else:
        rows = []
        for _, g in recent.iterrows():
            mark = "&#9989;" if g["pick_correct"] else "&#10060;"
            rows.append(
                f"<tr><td style='{STYLE_TD}'>{mark}</td>"
                f"<td style='{STYLE_TD}'>{g['date']}</td>"
                f"<td style='{STYLE_TD}'>{_league_name(g['league'])}</td>"
                f"<td style='{STYLE_TD}'>{g['home_team']} {int(g['home_score'])}&ndash;"
                f"{int(g['away_score'])} {g['away_team']}</td>"
                f"<td style='{STYLE_TD}'>{g['pick']}</td>"
                f"<td style='{STYLE_TD}'>{g['log_loss']:.2f}</td></tr>"
            )
        correct = int(recent["pick_correct"].sum())
        parts.append(
            f"<h3 style='{STYLE_H3}'>Past week: {correct}/{len(recent)} picks correct</h3>"
            + _table(["", "Date", "League", "Result", "Pick", "Log loss"], rows)
        )
    return "".join(parts)


def _forecast_section(futures: dict) -> str:
    parts = []
    for league, sim in futures.items():
        name = _league_name(league)
        if sim.get("status") == "no_fixtures" or not sim.get("clubs"):
            parts.append(
                f"<h3 style='{STYLE_H3}'>{name}</h3>"
                f"<p style='color:#666;font-size:12px;'>No {sim.get('season', '')} "
                f"fixtures published upstream yet.</p>"
            )
            continue
        rows = []
        for i, c in enumerate(sim["clubs"], start=1):
            rows.append(
                f"<tr><td style='{STYLE_TD}'>{i}</td>"
                f"<td style='{STYLE_TD}'><b>{c['team']}</b></td>"
                f"<td style='{STYLE_TD}'>{c['points']}</td>"
                f"<td style='{STYLE_TD}'>{c['exp_points']:.1f}</td>"
                f"<td style='{STYLE_TD}'>{c.get('exp_position', 0):.1f}</td>"
                f"<td style='{STYLE_TD}'><b>{_pct(c['p_title'])}</b></td>"
                f"<td style='{STYLE_TD}'>{_pct(c['p_top4'])}</td>"
                f"<td style='{STYLE_TD}'>{_pct(c.get('p_uel', 0.0))}</td>"
                f"<td style='{STYLE_TD}'>{_pct(c['p_relegation'])}</td></tr>"
            )
        leader = sim["clubs"][0]
        parts.append(
            f"<h3 style='{STYLE_H3}'>{name} &mdash; "
            f"{leader['team']} {_pct(leader['p_title'])} for the title</h3>"
            + _table(
                ["Proj", "Team", "Pts", "xPts", "xPos",
                 "Title", "UCL", "UEL", "Rel"],
                rows,
            )
        )
    parts.append(
        "<p style='color:#666;font-size:12px;'>Rest-of-season Monte Carlo "
        "(1,000 replays of the remaining fixtures with live in-sim Elo). "
        "Proj = projected finish ordered by expected position; xPts / xPos = "
        "expected points and position; UCL = top 4, UEL = 5th&ndash;6th, "
        "Rel = bottom 3.</p>"
    )
    return "".join(parts)


def update_html(run_date: str, fixtures: pd.DataFrame, recent: pd.DataFrame,
                ledger: dict, futures: dict) -> str:
    body = (
        "<h2 style='font-size:16px;margin:20px 0 0 0;'>&#128197; Games this week</h2>"
        + _fixtures_section(fixtures)
        + "<h2 style='font-size:16px;margin:28px 0 0 0;'>&#128200; Model performance</h2>"
        + _performance_section(recent, ledger)
        + "<h2 style='font-size:16px;margin:28px 0 0 0;'>&#128302; Final-table forecasts</h2>"
        + _forecast_section(futures)
    )
    return _wrap(
        "Soccer Update",
        f"Run date {run_date} &middot; fixtures for the next 7 days, the past "
        f"week's grades, and rest-of-season forecasts for the top 5 leagues.",
        body,
    )
