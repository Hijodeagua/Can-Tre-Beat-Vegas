"""HTML renderer for the twice-weekly NFL Elo update email — the NFL
sibling of `CFB/daily/emails.py`, in the same email-client-safe style
(table markup, inline styles only).

One email, four sections:

1. Power ratings — all 32 teams by Elo with season-to-date record.
2. This week's games — win probability, pick, the model's own spread and
   an expected score.
3. Model performance — rolling tracker (7d / 30d / season) with the
   paired Δlog-loss vs. always-pick-home, plus the past week's grades.
4. Season forecasts — expected wins, division / playoff / #1 seed /
   conference / Super Bowl odds and each division race.
"""

from __future__ import annotations

import pandas as pd

from NFL.elo.teams import DIVISIONS, name

STYLE_TABLE = (
    "border-collapse:collapse;width:100%;max-width:760px;"
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


def _record(w: int, l: int, t: int = 0) -> str:
    return f"{w}-{l}" + (f"-{t}" if t else "")


def _spread(x: float) -> str:
    """The model's line from the home side, as a bettor reads it."""
    line = -float(x)
    return f"{line:+.1f}" if line else "PK"


def _wrap(title: str, subtitle: str, body: str) -> str:
    return (
        f'<div style="font-family:Arial,Helvetica,sans-serif;color:#222;">'
        f"<h2 style=\"margin-bottom:2px;\">{title}</h2>"
        f'<p style="margin-top:0;color:#666;font-size:13px;">{subtitle}</p>'
        f"{body}"
        f'<p style="color:#999;font-size:11px;margin-top:24px;">'
        f"Betting-blind NFL Elo (margin-weighted updates, bye-week rest edge, "
        f"off-season regression) &middot; Can Tre Beat Vegas &middot; "
        f'<a href="https://whosyurgoat.app/vegas/nfl">whosyurgoat.app/vegas/nfl</a></p>'
        f"</div>"
    )


def _table(headers: list[str], rows: list[str]) -> str:
    head = "".join(f"<th style='{STYLE_TH}'>{h}</th>" for h in headers)
    return f"<table style='{STYLE_TABLE}'><tr>{head}</tr>{''.join(rows)}</table>"


def _matchup(home: str, away: str, neutral: bool) -> str:
    return (f"{away} vs. {home} (N)" if neutral else f"{away} @ <b>{home}</b>")


def _ratings_section(ratings: list[dict]) -> str:
    rows = []
    for r in ratings:
        delta = ("&mdash;" if r.get("preseason_elo") is None
                 else f"{r['elo'] - r['preseason_elo']:+.0f}")
        rows.append(
            f"<tr><td style='{STYLE_TD}'>{r['rank']}</td>"
            f"<td style='{STYLE_TD}'><b>{r['team']}</b> {name(r['team'])}</td>"
            f"<td style='{STYLE_TD}'>{r['division']}</td>"
            f"<td style='{STYLE_TD}'>{_record(r['wins'], r['losses'], r['ties'])}</td>"
            f"<td style='{STYLE_TD}'>{r['pts_diff']:+d}</td>"
            f"<td style='{STYLE_TD}'>{r['elo']:.0f}</td>"
            f"<td style='{STYLE_TD}'>{delta}</td></tr>"
        )
    return _table(["#", "Team", "Division", "W-L", "Pt diff", "Elo", "&Delta; pre"], rows)


def _divisions_section(divisions: dict) -> str:
    ranked = sorted(divisions.values(), key=lambda d: -(d["avgElo"] or 0))
    rows = []
    for i, d in enumerate(ranked, start=1):
        pre = "&mdash;" if d.get("preseasonAvgElo") is None else f"{d['preseasonAvgElo']:.0f}"
        rows.append(
            f"<tr><td style='{STYLE_TD}'>{i}</td>"
            f"<td style='{STYLE_TD}'><b>{d['name']}</b></td>"
            f"<td style='{STYLE_TD}'><b>{d['avgElo']:.0f}</b></td>"
            f"<td style='{STYLE_TD}'>{pre}</td>"
            f"<td style='{STYLE_TD}'>{d.get('top10', 0)}</td>"
            f"<td style='{STYLE_TD}'>{d['bestTeam']} ({d['bestElo']:.0f})</td>"
            f"<td style='{STYLE_TD}'>{d['worstTeam']} ({d['worstElo']:.0f})</td></tr>"
        )
    return (
        f"<h3 style='{STYLE_H3}'>Divisions by average Elo</h3>"
        + _table(["#", "Division", "Avg", "Preseason", "In top 10", "Best", "Worst"], rows)
    )


def _fixtures_section(fixtures: pd.DataFrame) -> str:
    if fixtures.empty:
        return "<p>No games on the slate.</p>"
    parts = []
    for day, sub in fixtures.groupby("date", sort=True):
        rows = []
        weekday = sub["weekday"].iloc[0]
        for m in sub.itertuples(index=False):
            fav_home = m.pick == m.home_team
            score = (f"{m.pred_home_score}&ndash;{m.pred_away_score}" if fav_home
                     else f"{m.pred_away_score}&ndash;{m.pred_home_score}")
            rows.append(
                f"<tr><td style='{STYLE_TD}'>{m.gametime or ''}</td>"
                f"<td style='{STYLE_TD}'>{_matchup(m.home_team, m.away_team, bool(m.neutral))}</td>"
                f"<td style='{STYLE_TD}'><b>{m.pick}</b></td>"
                f"<td style='{STYLE_TD}'>{_pct(m.pick_prob)}</td>"
                f"<td style='{STYLE_TD}'>{_spread(m.elo_spread)}</td>"
                f"<td style='{STYLE_TD}'>{score}</td>"
                f"<td style='{STYLE_TD}'>{m.pred_total:.1f}</td></tr>"
            )
        parts.append(
            f"<h3 style='{STYLE_H3}'>{weekday} {day} &middot; {len(sub)} games</h3>"
            + _table(["ET", "Matchup (home in bold)", "Pick", "Win prob",
                      "Elo line (home)", "Exp. score (winner first)", "Exp. total"], rows)
        )
    parts.append(
        f"<p style='{STYLE_NOTE}'>Win probability straight from Elo (home "
        f"edge applied except at neutral sites, marked N; a side off its bye "
        f"gets the rest edge). The Elo line is the model's own spread from "
        f"the home side &mdash; it has never seen a market number. Expected "
        f"score is that margin carved out of a matchup-specific total (each "
        f"team's recent points for/against, shrunk to the league mean); read "
        f"it as an average, not a literal final.</p>"
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
    ]
    parts.append(
        f"<h3 style='{STYLE_H3}'>Rolling tracker</h3>"
        + _table(["Window", "Correct", "Accuracy", "Log loss",
                  "&Delta;LL vs always-home"], tracker_rows)
        + f"<p style='{STYLE_NOTE}'>&Delta;LL is the paired per-game log-loss "
          f"difference against a fixed always-pick-home forecast on the same "
          f"games (p = 0.55, coin flip at neutral sites), &plusmn; one "
          f"standard error; negative = beating the baseline. Picks lock the "
          f"first morning a game appears on a slate.</p>"
    )
    if recent.empty:
        parts.append("<p>No games graded in the past week.</p>")
    else:
        rows = []
        for g in recent.itertuples(index=False):
            mark = "&#9989;" if g.pick_correct else ("&#129309;" if g.tie else "&#10060;")
            rows.append(
                f"<tr><td style='{STYLE_TD}'>{mark}</td>"
                f"<td style='{STYLE_TD}'>{g.date}</td>"
                f"<td style='{STYLE_TD}'>{_matchup(g.home_team, g.away_team, bool(g.neutral))}</td>"
                f"<td style='{STYLE_TD}'>{g.pick} ({_pct(g.pick_prob)})</td>"
                f"<td style='{STYLE_TD}'>{_spread(g.elo_spread)}</td>"
                f"<td style='{STYLE_TD}'>{int(g.home_score)}&ndash;{int(g.away_score)}</td>"
                f"<td style='{STYLE_TD}'>{g.log_loss:.2f}</td></tr>"
            )
        correct = int(recent["pick_correct"].astype(bool).sum())
        parts.append(
            f"<h3 style='{STYLE_H3}'>Past week: {correct}/{len(recent)} picks correct</h3>"
            + _table(["", "Date", "Game", "Pick", "Elo line", "Final (home first)",
                      "Log loss"], rows)
        )
    return "".join(parts)


def _forecast_section(futures: dict | None) -> str:
    if not futures or not futures.get("teams"):
        return "<p>No remaining games to simulate.</p>"
    teams = futures["teams"]
    rows = []
    for i, t in enumerate(teams, start=1):
        rows.append(
            f"<tr><td style='{STYLE_TD}'>{i}</td>"
            f"<td style='{STYLE_TD}'><b>{t['team']}</b></td>"
            f"<td style='{STYLE_TD}'>{t['division']}</td>"
            f"<td style='{STYLE_TD}'>{_record(t['wins'], t['losses'], t['ties'])}</td>"
            f"<td style='{STYLE_TD}'>{t['exp_wins']:.1f}-{t['exp_losses']:.1f}</td>"
            f"<td style='{STYLE_TD}'>{_pct(t['p_division'])}</td>"
            f"<td style='{STYLE_TD}'>{_pct(t['p_playoffs'])}</td>"
            f"<td style='{STYLE_TD}'>{_pct(t['p_top_seed'])}</td>"
            f"<td style='{STYLE_TD}'>{_pct(t['p_conf'])}</td>"
            f"<td style='{STYLE_TD}'><b>{_pct(t['p_sb'])}</b></td></tr>"
        )
    parts = [
        f"<h3 style='{STYLE_H3}'>Super Bowl odds &mdash; all 32</h3>"
        + _table(["#", "Team", "Division", "W-L", "Proj W-L", "Division", "Playoffs",
                  "#1 seed", "Conf. title", "Super Bowl"], rows)
    ]
    by_div: dict[str, list[dict]] = {}
    for t in teams:
        by_div.setdefault(t["division"], []).append(t)
    div_rows = []
    for div in DIVISIONS:
        contenders = sorted(by_div.get(div, []), key=lambda t: -t["p_division"])
        cells = ", ".join(f"<b>{c['team']}</b> {_pct(c['p_division'])}" for c in contenders)
        div_rows.append(
            f"<tr><td style='{STYLE_TD}'>{div}</td>"
            f"<td style='{STYLE_TD};white-space:normal;'>{cells}</td></tr>"
        )
    parts.append(
        f"<h3 style='{STYLE_H3}'>Division races</h3>"
        + _table(["Division", "Title odds"], div_rows)
        + f"<p style='{STYLE_NOTE}'>Rest-of-season Monte Carlo "
          f"({futures['sims']:,} replays of the {futures['remaining_games']} "
          f"remaining regular-season games with live in-sim Elo, then the "
          f"full seven-team bracket: Wild Card at the higher seed, the #1 "
          f"seed off its bye, Super Bowl at a neutral site). Ties break on "
          f"division / conference record, then at random &mdash; the real "
          f"tiebreaker ladder is not modeled.</p>"
    )
    return "".join(parts)


def update_html(run_date: str, ratings: list[dict], fixtures: pd.DataFrame,
                recent: pd.DataFrame, ledger: dict, futures: dict | None,
                week_label: str | None, divisions: dict | None = None) -> str:
    week_txt = f" &middot; {week_label}" if week_label else ""
    body = (
        "<h2 style='font-size:16px;margin:20px 0 0 0;'>&#127942; Elo power ratings</h2>"
        + _ratings_section(ratings)
        + (_divisions_section(divisions) if divisions else "")
        + "<h2 style='font-size:16px;margin:28px 0 0 0;'>&#128197; This week's games</h2>"
        + _fixtures_section(fixtures)
        + "<h2 style='font-size:16px;margin:28px 0 0 0;'>&#128200; Model performance</h2>"
        + _performance_section(recent, ledger)
        + "<h2 style='font-size:16px;margin:28px 0 0 0;'>&#128302; Season forecasts</h2>"
        + _forecast_section(futures)
    )
    return _wrap(
        "NFL Elo Update",
        f"Run date {run_date}{week_txt} &middot; the power ratings, this week's "
        f"games, the past week's grades, and rest-of-season forecasts.",
        body,
    )
