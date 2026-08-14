"""HTML renderers for the three daily emails.

All markup is table-based with inline styles (email-client safe). The same
fragments are reused for the site's static history snapshots.
"""

from __future__ import annotations

import pandas as pd

from mlb.daily.config import DIVISIONS, TEAM_LEAGUE, TEAM_NAMES

STYLE_TABLE = (
    "border-collapse:collapse;width:100%;max-width:640px;"
    "font-family:Arial,Helvetica,sans-serif;font-size:14px;"
)
STYLE_TH = (
    "text-align:left;padding:6px 10px;border-bottom:2px solid #333;"
    "background:#f4f4f4;"
)
STYLE_TD = "padding:6px 10px;border-bottom:1px solid #ddd;"


def _name(code: str) -> str:
    return TEAM_NAMES.get(code, code)


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%" if x >= 0.005 else "<1%"


def _wrap(title: str, subtitle: str, body: str) -> str:
    return (
        f'<div style="font-family:Arial,Helvetica,sans-serif;color:#222;">'
        f"<h2 style=\"margin-bottom:2px;\">{title}</h2>"
        f'<p style="margin-top:0;color:#666;font-size:13px;">{subtitle}</p>'
        f"{body}"
        f'<p style="color:#999;font-size:11px;margin-top:24px;">'
        f"Betting-blind Elo model (K=3, +24 home, MOV-weighted, 60% season "
        f"carryover) &middot; Can Tre Beat Vegas &middot; "
        f'<a href="https://whosyurgoat.app/vegas/mlb">whosyurgoat.app/vegas/mlb</a></p>'
        f"</div>"
    )


def futures_html(date: str, futures: pd.DataFrame, standings: pd.DataFrame) -> str:
    st = standings.set_index("team")
    sections = []
    for div, teams in DIVISIONS.items():
        f = (
            futures[futures.team.isin(teams)]
            .sort_values("division_pct", ascending=False)
        )
        rows = ""
        for r in f.itertuples():
            w, l = int(st.loc[r.team, "wins"]), int(st.loc[r.team, "losses"])
            rd = int(st.loc[r.team, "run_diff"])
            rows += (
                f"<tr><td style='{STYLE_TD}'><b>{_name(r.team)}</b></td>"
                f"<td style='{STYLE_TD}'>{w}-{l}</td>"
                f"<td style='{STYLE_TD}'>{rd:+d}</td>"
                f"<td style='{STYLE_TD}'>{r.elo:.0f}</td>"
                f"<td style='{STYLE_TD}'>{r.mean_wins:.1f}</td>"
                f"<td style='{STYLE_TD}'><b>{_pct(r.division_pct)}</b></td>"
                f"<td style='{STYLE_TD}'>{_pct(r.playoff_pct)}</td></tr>"
            )
        leader = f.iloc[0]
        sections.append(
            f"<h3 style='margin-bottom:4px;'>{div} &mdash; "
            f"{_name(leader.team)} {_pct(leader.division_pct)}</h3>"
            f"<table style='{STYLE_TABLE}'><tr>"
            f"<th style='{STYLE_TH}'>Team</th><th style='{STYLE_TH}'>W-L</th>"
            f"<th style='{STYLE_TH}'>Run diff</th><th style='{STYLE_TH}'>Elo</th>"
            f"<th style='{STYLE_TH}'>Proj W</th><th style='{STYLE_TH}'>Div%</th>"
            f"<th style='{STYLE_TH}'>Playoff%</th></tr>{rows}</table>"
        )

    seeds = []
    for lg in ("AL", "NL"):
        f = futures[futures.team.map(TEAM_LEAGUE) == lg]
        top = f.sort_values("top_seed_pct", ascending=False).iloc[0]
        seeds.append(
            f"<li>{lg} #1 seed favorite: <b>{_name(top.team)}</b> "
            f"({_pct(top.top_seed_pct)})</li>"
        )
    body = "".join(sections) + (
        "<h3 style='margin-bottom:4px;'>Top seeds</h3>"
        f"<ul style='font-size:14px;'>{''.join(seeds)}</ul>"
        "<p style='color:#666;font-size:12px;'>Rest-of-season Monte Carlo "
        "from current Elo ratings, live rating updates within each sim. "
        "12-team playoff format; ties broken at random (head-to-head "
        "tiebreakers not modeled). Run differential is season-to-date. "
        "Projected W is the mean simulated final win total.</p>"
    )
    return _wrap(
        f"MLB Futures Forecast &mdash; {date}",
        "Rest-of-season Monte Carlo: division winners, playoff odds, seeds",
        body,
    )


def _slate_table(slate: pd.DataFrame) -> str:
    """One slate table; adds the per-starter Elo adjustment column when the
    frame carries it (the SP-adjusted model)."""
    has_sp_adj = "home_sp_adj" in slate.columns
    rows = ""
    for r in slate.itertuples():
        fav_home = r.pick == r.home
        score = (
            f"{r.pred_home_score}&ndash;{r.pred_away_score}"
            if fav_home
            else f"{r.pred_away_score}&ndash;{r.pred_home_score}"
        )
        dh = f" (G{r.game_num})" if r.game_num > 1 else ""
        away_sp = getattr(r, "away_sp", "") or "TBD"
        home_sp = getattr(r, "home_sp", "") or "TBD"
        total = getattr(r, "pred_total", None)
        total_txt = f"{total:.1f}" if total is not None else "&mdash;"
        sp_adj_cell = ""
        if has_sp_adj:
            sp_adj_cell = (
                f"<td style='{STYLE_TD};font-size:12px;'>"
                f"{r.away_sp_adj:+.1f} / {r.home_sp_adj:+.1f}</td>"
            )
        rows += (
            f"<tr><td style='{STYLE_TD}'>{_name(r.away)} @ "
            f"<b>{_name(r.home)}</b>{dh}</td>"
            f"<td style='{STYLE_TD};font-size:12px;color:#555;'>"
            f"{away_sp} vs. {home_sp}</td>"
            + sp_adj_cell +
            f"<td style='{STYLE_TD}'><b>{_name(r.pick)}</b></td>"
            f"<td style='{STYLE_TD}'>{_pct(r.pick_prob)}</td>"
            f"<td style='{STYLE_TD}'>{score}</td>"
            f"<td style='{STYLE_TD}'>{total_txt}</td></tr>"
        )
    sp_adj_th = (f"<th style='{STYLE_TH}'>SP adj (Elo, away/home)</th>"
                 if has_sp_adj else "")
    return (
        f"<table style='{STYLE_TABLE}'><tr>"
        f"<th style='{STYLE_TH}'>Matchup (home in bold)</th>"
        f"<th style='{STYLE_TH}'>SP (away vs. home)</th>"
        + sp_adj_th +
        f"<th style='{STYLE_TH}'>Pick</th>"
        f"<th style='{STYLE_TH}'>Win prob</th>"
        f"<th style='{STYLE_TH}'>Exp. runs (winner first)</th>"
        f"<th style='{STYLE_TH}'>Exp. total</th></tr>{rows}</table>"
    )


def slate_html(date: str, slate: pd.DataFrame,
               shadow: pd.DataFrame | None = None) -> str:
    if slate.empty:
        body = "<p>No MLB games scheduled today.</p>"
    else:
        sp_in_model = "home_sp_adj" in slate.columns
        sp_note = (
            "Starter identity IS a model input: each probable's rolling "
            "game score (exponentially weighted, 20-start half-life) is "
            "compared to his team's staff average and the difference enters "
            "the team's Elo at 3.0 points per game-score point (the SP adj "
            "column, so the effect is auditable per game). Rookies/thin "
            "history fall back to a shrunk staff rating; TBD uses the staff "
            "rating. Rest and travel add up to a few Elo points each. "
            if sp_in_model else
            "Probable starters from MLB (TBD when unannounced). This "
            "table is the current team-level Elo (starter-blind); the "
            "starter-adjusted model below runs in shadow while its live "
            "record accrues, and takes over after the shadow window. "
        )
        body = (
            _slate_table(slate)
            + "<p style='color:#666;font-size:12px;'>" + sp_note +
            "Win probability from Elo (+24 home advantage). "
            "Expected total is matchup-specific: each club's recent runs "
            "scored/allowed (exponentially weighted, ~20-game half-life, "
            "shrunk to league average) sets the run environment, and the "
            "Elo-implied margin is carved out of it. Expected runs are "
            "shown at one decimal; read them as averages, not a literal "
            "final score.</p>"
        )
        if shadow is not None and not shadow.empty:
            body += (
                "<h3 style='margin-bottom:4px;'>Shadow: starter-adjusted "
                "model</h3>"
                + _slate_table(shadow)
                + "<p style='color:#666;font-size:12px;'>Same team Elo "
                "plus the starting-pitcher, rest, and travel adjustments "
                "(research/SP-BACKTEST.md). Graded in its own ledger; "
                "SP adj is each starter's Elo adjustment (away/home), so "
                "the effect on the pick is visible per game.</p>"
            )
    return _wrap(
        f"MLB Slate &mdash; {date}",
        f"Predictions for every game on {date} (ET)",
        body,
    )


def grade_html(date: str, graded: pd.DataFrame | None,
               ledger_row: pd.Series | None,
               shadow_ledger_row: pd.Series | None = None) -> str:
    if graded is None or ledger_row is None:
        body = (
            "<p>No predictions were on file for yesterday &mdash; grading "
            "begins the day after the first slate email.</p>"
        )
        return _wrap("MLB Grade", "Yesterday's report card", body)

    played = graded[graded.played == True]  # noqa: E712
    rows = ""
    for r in played.itertuples():
        mark = "&#9989;" if r.pick_correct else "&#10060;"
        fav_home = r.pick == r.home
        pred = (
            f"{r.pred_home_score:g}&ndash;{r.pred_away_score:g}"
            if fav_home
            else f"{r.pred_away_score:g}&ndash;{r.pred_home_score:g}"
        )
        act = f"{int(r.home_score)}&ndash;{int(r.away_score)}"
        rows += (
            f"<tr><td style='{STYLE_TD}'>{mark}</td>"
            f"<td style='{STYLE_TD}'>{_name(r.away)} @ <b>{_name(r.home)}</b></td>"
            f"<td style='{STYLE_TD}'>{_name(r.pick)} ({_pct(r.pick_prob)})</td>"
            f"<td style='{STYLE_TD}'>{pred}</td>"
            f"<td style='{STYLE_TD}'>{act} (home first)</td></tr>"
        )
    lr = ledger_row
    d_mean = lr.get("cum_d_ll_mean")
    d_se = lr.get("cum_d_ll_se")
    if pd.notna(d_mean) and pd.notna(d_se):
        beating = d_mean < 0
        headline = (
            f"<p style='font-size:15px;'><b>Cumulative &Delta;log-loss vs "
            f"always-pick-home: {d_mean:+.4f} &plusmn; {d_se:.4f}</b> per "
            f"game over {int(lr['cum_games'])} games (paired; negative = "
            f"beating the baseline{' - currently ahead' if beating else ' - currently behind'})."
            f"</p>"
        )
    else:
        headline = ""
    # Always-home on the same slate: right exactly when the home team won.
    home_row = ""
    if pd.notna(lr.get("home_log_loss")):
        home_row = (
            f"<tr><td style='{STYLE_TD}'>always-pick-home</td>"
            f"<td style='{STYLE_TD}'>{int(lr['home_correct'])}/"
            f"{int(lr['games'])}</td>"
            f"<td style='{STYLE_TD}'>{lr['home_log_loss']:.3f}</td></tr>"
        )
    summary = headline + (
        f"<table style='{STYLE_TABLE};max-width:420px;'><tr>"
        f"<th style='{STYLE_TH}'>Yesterday</th>"
        f"<th style='{STYLE_TH}'>Correct</th>"
        f"<th style='{STYLE_TH}'>Log-loss</th></tr>"
        f"<tr><td style='{STYLE_TD}'>model</td>"
        f"<td style='{STYLE_TD}'>{int(lr['correct'])}/{int(lr['games'])}</td>"
        f"<td style='{STYLE_TD}'>{lr['log_loss']:.3f}</td></tr>"
        + home_row + "</table>"
        f"<p style='font-size:13px;color:#444;'>Daily detail: "
        f"{100 * lr['accuracy']:.0f}% hit rate &middot; "
        f"Brier {lr['brier']:.3f} &middot; avg margin error "
        f"{lr['avg_margin_err']:.1f} &middot; avg total-runs error "
        f"{lr['avg_total_err']:.1f}. Running: "
        f"{int(lr['cum_correct'])}/{int(lr['cum_games'])} "
        f"({100 * lr['cum_accuracy']:.1f}%) &middot; "
        f"cumulative log-loss {lr['cum_log_loss']:.3f} &middot; "
        f"cumulative Brier {lr['cum_brier']:.3f}"
        + (f" &middot; {int(lr['skipped'])} postponed/skipped"
           if lr["skipped"] else "")
        + "</p>"
    )
    body = summary + (
        f"<table style='{STYLE_TABLE}'><tr>"
        f"<th style='{STYLE_TH}'></th><th style='{STYLE_TH}'>Game</th>"
        f"<th style='{STYLE_TH}'>Pick</th><th style='{STYLE_TH}'>Predicted</th>"
        f"<th style='{STYLE_TH}'>Actual</th></tr>{rows}</table>"
        "<p style='color:#666;font-size:12px;'>The headline is the paired "
        "per-game log-loss difference against a fixed always-pick-home "
        "forecast (p=0.534) on the same games, &plusmn; one standard "
        "error; daily hit rate is noisy and demoted on purpose. Reference "
        "points: 0.693 is a coin flip; the model's walk-forward backtest "
        "runs 0.680 (research/SP-BACKTEST.md).</p>"
    )
    if shadow_ledger_row is not None and pd.notna(
            shadow_ledger_row.get("cum_d_ll_mean")):
        sl = shadow_ledger_row
        body += (
            f"<p style='font-size:13px;color:#444;'><b>Shadow model "
            f"(starter-adjusted):</b> cumulative &Delta;log-loss vs "
            f"always-pick-home {sl['cum_d_ll_mean']:+.4f} &plusmn; "
            f"{sl['cum_d_ll_se']:.4f} over {int(sl['cum_games'])} games "
            f"({int(sl['cum_correct'])}/{int(sl['cum_games'])} correct). "
            f"Graded in its own ledger; cutover after the shadow "
            f"window.</p>"
        )
    return _wrap(
        f"MLB Grade &mdash; {date}",
        f"How the {date} predictions scored against actual results",
        body,
    )
