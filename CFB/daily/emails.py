"""HTML renderer for the twice-weekly college-football update email.

One email, four sections, in the soccer update's email-client-safe style
(table markup, inline styles only):

1. Top 25 — the Elo board with season-to-date record.
2. Games this week — every FBS-involved game in the next
   EMAIL_FIXTURE_DAYS days with win probability, pick and expected score.
3. Model performance — rolling tracker (7d / 30d / season) with the paired
   Δlog-loss vs. always-pick-home, plus the past week's graded games.
4. Season forecasts — expected wins, bowl / undefeated odds and each
   conference's title race from the rest-of-season Monte Carlo.
"""

from __future__ import annotations

import pandas as pd

from CFB.daily.config import BOWL_ELIGIBLE_WINS, TOP_N
from CFB.data.teams import conference_short

STYLE_TABLE = (
    "border-collapse:collapse;width:100%;max-width:720px;"
    "font-family:Arial,Helvetica,sans-serif;font-size:13px;"
)
STYLE_TH = (
    "text-align:left;padding:5px 8px;border-bottom:2px solid #333;"
    "background:#f4f4f4;white-space:nowrap;"
)
STYLE_TD = "padding:5px 8px;border-bottom:1px solid #ddd;white-space:nowrap;"
STYLE_H3 = "margin:18px 0 4px 0;"
STYLE_NOTE = "color:#666;font-size:12px;"


def _pct(x) -> str:
    if x is None or x != x:
        return "&mdash;"
    return f"{100 * x:.0f}%" if x >= 0.005 else "<1%"


def _wrap(title: str, subtitle: str, body: str) -> str:
    return (
        f'<div style="font-family:Arial,Helvetica,sans-serif;color:#222;">'
        f"<h2 style=\"margin-bottom:2px;\">{title}</h2>"
        f'<p style="margin-top:0;color:#666;font-size:13px;">{subtitle}</p>'
        f"{body}"
        f'<p style="color:#999;font-size:11px;margin-top:24px;">'
        f"Betting-blind FBS Elo (conference-aware season regression, pooled "
        f"FCS opponent, margin-weighted updates) &middot; Can Tre Beat Vegas "
        f"&middot; "
        f'<a href="https://whosyurgoat.app/vegas/cfb">whosyurgoat.app/vegas/cfb</a></p>'
        f"</div>"
    )


def _table(headers: list[str], rows: list[str]) -> str:
    head = "".join(f"<th style='{STYLE_TH}'>{h}</th>" for h in headers)
    return f"<table style='{STYLE_TABLE}'><tr>{head}</tr>{''.join(rows)}</table>"


def _matchup(home: str, away: str, neutral: bool) -> str:
    return (f"{away} vs. {home} (N)" if neutral else f"{away} @ <b>{home}</b>")


def _top25_section(ratings: list[dict]) -> str:
    rows = []
    for r in ratings[:TOP_N]:
        rows.append(
            f"<tr><td style='{STYLE_TD}'>{r['rank']}</td>"
            f"<td style='{STYLE_TD}'><b>{r['team']}</b></td>"
            f"<td style='{STYLE_TD}'>{conference_short(r['conference'])}</td>"
            f"<td style='{STYLE_TD}'>{r['wins']}-{r['losses']}</td>"
            f"<td style='{STYLE_TD}'>{r['pts_diff']:+d}</td>"
            f"<td style='{STYLE_TD}'>{r['elo']:.0f}</td></tr>"
        )
    return _table(["#", "Team", "Conf", "W-L", "Pt diff", "Elo"], rows)


def _fixtures_section(fixtures: pd.DataFrame) -> str:
    if fixtures.empty:
        return "<p>No FBS games in the coming week.</p>"
    parts = []
    for day, sub in fixtures.groupby("date", sort=True):
        rows = []
        for m in sub.itertuples(index=False):
            fav_home = m.pick == m.home_team
            score = (f"{m.pred_home_score}&ndash;{m.pred_away_score}" if fav_home
                     else f"{m.pred_away_score}&ndash;{m.pred_home_score}")
            fcs = " <span style='color:#999;'>(FCS)</span>" if (m.home_fcs or m.away_fcs) else ""
            rows.append(
                f"<tr><td style='{STYLE_TD}'>{_matchup(m.home_team, m.away_team, bool(m.neutral))}{fcs}</td>"
                f"<td style='{STYLE_TD}'><b>{m.pick}</b></td>"
                f"<td style='{STYLE_TD}'>{_pct(m.pick_prob)}</td>"
                f"<td style='{STYLE_TD}'>{score}</td>"
                f"<td style='{STYLE_TD}'>{m.pred_total:.1f}</td></tr>"
            )
        parts.append(
            f"<h3 style='{STYLE_H3}'>{day} &middot; {len(sub)} games</h3>"
            + _table(["Matchup (home in bold)", "Pick", "Win prob",
                      "Exp. score (winner first)", "Exp. total"], rows)
        )
    parts.append(
        f"<p style='{STYLE_NOTE}'>Win probability straight from Elo (home "
        f"edge applied except at neutral sites, marked N). Expected score is "
        f"the Elo-implied margin carved out of a matchup-specific total "
        f"(each program's recent points for/against, shrunk to the FBS "
        f"mean); read it as an average, not a literal final. FCS opponents "
        f"are one pooled rating, so those lines are the least informative "
        f"on the slate.</p>"
    )
    return "".join(parts)


def _record_row(label: str, stats: dict) -> str:
    if not stats or not stats.get("graded"):
        return (
            f"<tr><td style='{STYLE_TD}'>{label}</td>"
            f"<td style='{STYLE_TD}'>0</td><td style='{STYLE_TD}'>&mdash;</td>"
            f"<td style='{STYLE_TD}'>&mdash;</td><td style='{STYLE_TD}'>&mdash;</td></tr>"
        )
    d = stats.get("d_ll_mean")
    se = stats.get("d_ll_se")
    delta = ("&mdash;" if d is None else
             f"{d:+.4f}" + (f" &plusmn; {se:.4f}" if se is not None else ""))
    return (
        f"<tr><td style='{STYLE_TD}'>{label}</td>"
        f"<td style='{STYLE_TD}'>{stats['correct']}/{stats['graded']}</td>"
        f"<td style='{STYLE_TD}'>{100 * stats['accuracy']:.1f}%</td>"
        f"<td style='{STYLE_TD}'>{stats['log_loss']:.4f}</td>"
        f"<td style='{STYLE_TD}'>{delta}</td></tr>"
    )


def _performance_section(recent: pd.DataFrame, ledger: dict) -> str:
    parts = []
    rolling = ledger.get("rolling", {})
    tracker_rows = [
        _record_row("Last 7 days", rolling.get("7d", {})),
        _record_row("Last 30 days", rolling.get("30d", {})),
        _record_row("Season to date", ledger),
        _record_row("Season, FBS vs. FBS only", ledger.get("fbs_only", {})),
    ]
    parts.append(
        f"<h3 style='{STYLE_H3}'>Rolling tracker</h3>"
        + _table(["Window", "Correct", "Accuracy", "Log loss",
                  "&Delta;LL vs always-home"], tracker_rows)
        + f"<p style='{STYLE_NOTE}'>&Delta;LL is the paired per-game log-loss "
          f"difference against a fixed always-pick-home forecast on the same "
          f"games (p = 0.632, coin flip at neutral sites), &plusmn; one "
          f"standard error; negative = beating the baseline. Hit rate is "
          f"inflated by FBS-vs-FCS games, which is why the FBS-only row "
          f"exists.</p>"
    )
    if recent.empty:
        parts.append("<p>No games graded in the past week.</p>")
    else:
        rows = []
        for g in recent.itertuples(index=False):
            mark = "&#9989;" if g.pick_correct else "&#10060;"
            rows.append(
                f"<tr><td style='{STYLE_TD}'>{mark}</td>"
                f"<td style='{STYLE_TD}'>{g.date}</td>"
                f"<td style='{STYLE_TD}'>{_matchup(g.home_team, g.away_team, bool(g.neutral))}</td>"
                f"<td style='{STYLE_TD}'>{g.pick} ({_pct(g.pick_prob)})</td>"
                f"<td style='{STYLE_TD}'>{int(g.home_points)}&ndash;{int(g.away_points)}</td>"
                f"<td style='{STYLE_TD}'>{g.log_loss:.2f}</td></tr>"
            )
        correct = int(recent["pick_correct"].astype(bool).sum())
        parts.append(
            f"<h3 style='{STYLE_H3}'>Past week: {correct}/{len(recent)} picks correct</h3>"
            + _table(["", "Date", "Game", "Pick", "Final (home first)", "Log loss"], rows)
        )
    return "".join(parts)


def _forecast_section(futures: dict | None) -> str:
    if not futures or not futures.get("teams"):
        return "<p>No remaining games to simulate.</p>"
    teams = futures["teams"]
    rows = []
    for i, t in enumerate(teams[:TOP_N], start=1):
        rows.append(
            f"<tr><td style='{STYLE_TD}'>{i}</td>"
            f"<td style='{STYLE_TD}'><b>{t['team']}</b></td>"
            f"<td style='{STYLE_TD}'>{conference_short(t['conference'])}</td>"
            f"<td style='{STYLE_TD}'>{t['wins']}-{t['losses']}</td>"
            f"<td style='{STYLE_TD}'>{t['exp_wins']:.1f}-{t['exp_losses']:.1f}</td>"
            f"<td style='{STYLE_TD}'>{_pct(t['p_bowl'])}</td>"
            f"<td style='{STYLE_TD}'>{_pct(t['p_undefeated'])}</td>"
            f"<td style='{STYLE_TD}'>{_pct(t['p_ccg'])}</td>"
            f"<td style='{STYLE_TD}'><b>{_pct(t['p_conf_title'])}</b></td></tr>"
        )
    parts = [
        f"<h3 style='{STYLE_H3}'>Projected wins &mdash; top {TOP_N}</h3>"
        + _table(["#", "Team", "Conf", "W-L", "Proj W-L", "Bowl",
                  "Undefeated", "CCG", "Conf title"], rows)
    ]
    by_conf: dict[str, list[dict]] = {}
    for t in teams:
        if t.get("p_conf_title") is not None:
            by_conf.setdefault(t["conference"], []).append(t)
    conf_rows = []
    for conf in sorted(by_conf):
        contenders = sorted(by_conf[conf], key=lambda t: -t["p_conf_title"])[:4]
        cells = ", ".join(f"<b>{c['team']}</b> {_pct(c['p_conf_title'])}"
                          for c in contenders)
        conf_rows.append(
            f"<tr><td style='{STYLE_TD}'>{conf}</td>"
            f"<td style='{STYLE_TD};white-space:normal;'>{cells}</td></tr>"
        )
    parts.append(
        f"<h3 style='{STYLE_H3}'>Conference title races</h3>"
        + _table(["Conference", "Favorites"], conf_rows)
        + f"<p style='{STYLE_NOTE}'>Rest-of-season Monte Carlo "
          f"({futures['sims']:,} replays of the {futures['remaining_games']} "
          f"remaining regular-season games with live in-sim Elo). Bowl = "
          f"{BOWL_ELIGIBLE_WINS}+ wins; CCG = reaching the conference "
          f"championship game (top two by conference record, ties random); "
          f"the title game is simulated at a neutral site. The 12-team "
          f"playoff is a committee pick and is not modeled.</p>"
    )
    return "".join(parts)


def update_html(run_date: str, ratings: list[dict], fixtures: pd.DataFrame,
                recent: pd.DataFrame, ledger: dict, futures: dict | None,
                week: int | None) -> str:
    week_txt = f" &middot; week {week}" if week else ""
    body = (
        "<h2 style='font-size:16px;margin:20px 0 0 0;'>&#127942; Elo top 25</h2>"
        + _top25_section(ratings)
        + "<h2 style='font-size:16px;margin:28px 0 0 0;'>&#128197; Games this week</h2>"
        + _fixtures_section(fixtures)
        + "<h2 style='font-size:16px;margin:28px 0 0 0;'>&#128200; Model performance</h2>"
        + _performance_section(recent, ledger)
        + "<h2 style='font-size:16px;margin:28px 0 0 0;'>&#128302; Season forecasts</h2>"
        + _forecast_section(futures)
    )
    return _wrap(
        "College Football Update",
        f"Run date {run_date}{week_txt} &middot; the Elo board, the next 7 "
        f"days of games, the past week's grades, and rest-of-season forecasts.",
        body,
    )
