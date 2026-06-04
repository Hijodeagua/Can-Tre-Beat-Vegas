"""
Against-the-spread (ATS) standings and bookmaker accuracy visuals.

This module computes two families of report views from data that is already
committed in the repo (no live API calls required):

  (A) Bookmaker accuracy leaderboard
      - NBA: reuses the moneyline winner-accuracy logic that the NBA daily
        report already produces (passed in as a DataFrame) and adds an
        embeddable bar chart.

  (B) Team ATS (against-the-spread) standings
      - NFL: computed from ``data/2023-2025W3.csv``, which records, per team
        per game, the consensus closing spread (``Spread_vg``) and whether the
        team covered it (``vs._Line_vg`` -> covered / did not cover / push).
        Produces a cover-% table, a cover-% bar chart, and an ATS-margin chart.
      - NBA: the committed NBA odds snapshots store spread *odds* (the juice,
        e.g. -110) but NOT the spread *line* (the point handicap), so a true
        NBA point-spread cover record cannot be computed from available data.
        The report renders a clearly-labelled note instead of fabricated
        numbers.

All chart helpers save PNGs into the existing ``reports/charts`` directory and
return a path relative to ``reports/`` so the HTML report can embed them with a
plain ``<img>`` tag (the same directory layout the report already uses for its
other charts).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CHARTS_DIR = os.path.join(REPORTS_DIR, "charts")

# Path to the bundled NFL per-team game stats (already contains ATS results).
NFL_STATS_CSV = os.path.join(DATA_DIR, "2023-2025W3.csv")

# Brand colors used elsewhere in the report.
NAVY = "#1d428a"
TEAL = "#00788c"
GREEN = "#28a745"
RED = "#dc3545"


# ---------------------------------------------------------------------------
# NFL ATS standings
# ---------------------------------------------------------------------------

def load_nfl_ats_standings(stats_csv: str = NFL_STATS_CSV) -> pd.DataFrame:
    """Compute NFL against-the-spread records per team from the bundled stats.

    Returns a DataFrame sorted by cover % (descending). Empty if the source
    data is missing or lacks the expected columns.

    Columns: Team, ATS Wins, ATS Losses, Pushes, Games, Cover %, Avg ATS Margin
    where ATS Margin = (TeamScore - OppScore) + Spread_vg  (positive => covered).
    """
    if not os.path.exists(stats_csv):
        return pd.DataFrame()

    try:
        df = pd.read_csv(stats_csv)
    except Exception:
        return pd.DataFrame()

    required = {"Team", "vs._Line_vg"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df = df.copy()
    df["vs._Line_vg"] = df["vs._Line_vg"].astype(str).str.strip().str.lower()

    rows: List[Dict] = []
    for team, grp in df.groupby("Team"):
        result_counts = grp["vs._Line_vg"].value_counts()
        wins = int(result_counts.get("covered", 0))
        losses = int(result_counts.get("did not cover", 0))
        pushes = int(result_counts.get("push", 0))
        decided = wins + losses
        if decided == 0:
            continue

        # ATS margin = actual margin relative to the spread line.
        margin = None
        if {"TeamScore", "OppScore", "Spread_vg"}.issubset(grp.columns):
            scored = pd.to_numeric(grp["TeamScore"], errors="coerce")
            allowed = pd.to_numeric(grp["OppScore"], errors="coerce")
            spread = pd.to_numeric(grp["Spread_vg"], errors="coerce")
            ats_margins = (scored - allowed) + spread
            if ats_margins.notna().any():
                margin = round(float(ats_margins.mean()), 1)

        rows.append(
            {
                "Team": team,
                "ATS Wins": wins,
                "ATS Losses": losses,
                "Pushes": pushes,
                "Games": wins + losses + pushes,
                "Cover %": round(wins / decided * 100, 1),
                "Avg ATS Margin": margin,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values("Cover %", ascending=False).reset_index(drop=True)


def create_nfl_ats_chart(ats_df: pd.DataFrame, date_str: str) -> Optional[str]:
    """Bar chart of NFL team cover % plus an ATS-margin panel.

    Saves a PNG into ``reports/charts`` and returns a path relative to
    ``reports/`` (for embedding) or ``None`` if there is nothing to plot.
    """
    if ats_df.empty:
        return None

    os.makedirs(CHARTS_DIR, exist_ok=True)

    has_margin = "Avg ATS Margin" in ats_df.columns and ats_df["Avg ATS Margin"].notna().any()
    fig, axes = plt.subplots(
        2 if has_margin else 1,
        1,
        figsize=(12, 11 if has_margin else 6.5),
    )
    if not has_margin:
        axes = [axes]

    # --- Panel 1: cover % per team ---
    ax = axes[0]
    cover = ats_df.sort_values("Cover %", ascending=True)
    colors = [GREEN if v >= 52.4 else (RED if v < 47.6 else "#6c757d") for v in cover["Cover %"]]
    ax.barh(cover["Team"], cover["Cover %"], color=colors)
    ax.axvline(50, color="#333", linestyle="--", linewidth=1, label="50% (coin flip)")
    ax.axvline(52.4, color=NAVY, linestyle=":", linewidth=1, label="52.4% (break-even @ -110)")
    ax.set_xlabel("Cover % (against the spread)")
    ax.set_title("NFL Team ATS Cover %", fontsize=14, fontweight="bold", color=NAVY)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, max(70, cover["Cover %"].max() + 5))
    for i, (v, w, l) in enumerate(zip(cover["Cover %"], cover["ATS Wins"], cover["ATS Losses"])):
        ax.text(v + 0.5, i, f"{v:.0f}% ({w}-{l})", va="center", fontsize=8)

    # --- Panel 2: average ATS margin per team ---
    if has_margin:
        ax2 = axes[1]
        m = ats_df.dropna(subset=["Avg ATS Margin"]).sort_values("Avg ATS Margin", ascending=True)
        mcolors = [GREEN if v >= 0 else RED for v in m["Avg ATS Margin"]]
        ax2.barh(m["Team"], m["Avg ATS Margin"], color=mcolors)
        ax2.axvline(0, color="#333", linewidth=1)
        ax2.set_xlabel("Average ATS margin (points beyond the spread)")
        ax2.set_title(
            "NFL Team Average ATS Margin", fontsize=14, fontweight="bold", color=NAVY
        )
        for i, v in enumerate(m["Avg ATS Margin"]):
            ax2.text(
                v + (0.1 if v >= 0 else -0.1),
                i,
                f"{v:+.1f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8,
            )

    plt.suptitle(
        "NFL Against-the-Spread Standings (2023-2025)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.98))

    out_path = os.path.join(CHARTS_DIR, f"nfl_ats_{date_str}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.relpath(out_path, REPORTS_DIR)


# ---------------------------------------------------------------------------
# Bookmaker accuracy chart (NBA leaderboard already computed upstream)
# ---------------------------------------------------------------------------

def create_bookmaker_accuracy_chart(
    bookie_df: pd.DataFrame, date_str: str, league: str = "NBA"
) -> Optional[str]:
    """Horizontal bar chart of bookmaker winner-prediction accuracy.

    Expects the columns produced by ``calculate_bookie_accuracy``:
    ``Bookmaker``, ``Accuracy %``, ``Correct Predictions``, ``Games Analyzed``.
    Returns a path relative to ``reports/`` or ``None``.
    """
    if bookie_df is None or bookie_df.empty or "Accuracy %" not in bookie_df.columns:
        return None

    os.makedirs(CHARTS_DIR, exist_ok=True)

    data = bookie_df.sort_values("Accuracy %", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.55 * len(data) + 1.5)))

    best = data["Accuracy %"].max()
    colors = [GREEN if v == best else TEAL for v in data["Accuracy %"]]
    ax.barh(data["Bookmaker"], data["Accuracy %"], color=colors)

    avg = data["Accuracy %"].mean()
    ax.axvline(avg, color=NAVY, linestyle="--", linewidth=1, label=f"Field avg {avg:.1f}%")

    ax.set_xlabel("Winner-prediction accuracy (%)")
    ax.set_title(
        f"{league} Bookmaker Accuracy Leaderboard",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
    )
    ax.set_xlim(0, max(75, best + 5))
    ax.legend(fontsize=9, loc="lower right")

    has_record = {"Correct Predictions", "Games Analyzed"}.issubset(data.columns)
    for i in range(len(data)):
        acc = data["Accuracy %"].iloc[i]
        rec = ""
        if has_record:
            rec = f" ({int(data['Correct Predictions'].iloc[i])}/{int(data['Games Analyzed'].iloc[i])})"
        ax.text(acc + 0.4, i, f"{acc:.1f}%{rec}", va="center", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, f"bookmaker_accuracy_{league.lower()}_{date_str}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.relpath(out_path, REPORTS_DIR)


# ---------------------------------------------------------------------------
# HTML section builders
# ---------------------------------------------------------------------------

def build_bookmaker_section_html(
    bookie_df: pd.DataFrame, chart_rel_path: Optional[str], league: str = "NBA"
) -> str:
    """Leaderboard table + embedded accuracy chart as an HTML card."""
    if bookie_df is None or bookie_df.empty:
        return (
            '<div class="card"><p class="data-note">No bookmaker accuracy data '
            "available for this period.</p></div>"
        )

    rows = ""
    for idx, row in bookie_df.iterrows():
        hl = ' class="highlight"' if idx == 0 else ""
        rows += (
            f"<tr{hl}><td>{idx + 1}</td><td>{row['Bookmaker']}</td>"
            f"<td><strong>{row['Accuracy %']}%</strong></td>"
            f"<td>{int(row['Correct Predictions'])}/{int(row['Games Analyzed'])}</td></tr>"
        )

    img_html = ""
    if chart_rel_path:
        img_html = (
            f'<img src="{chart_rel_path}" alt="{league} bookmaker accuracy chart" '
            'style="width:100%;height:auto;border-radius:8px;margin-top:12px;">'
        )

    return f"""
        <div class="card">
            <table>
                <tr><th>Rank</th><th>Bookmaker</th><th>Accuracy</th><th>Record</th></tr>
                {rows}
            </table>
            <p style="font-size:11px;color:#666;margin-top:8px;">
                Accuracy = share of games where the book's moneyline favorite won.
            </p>
            {img_html}
        </div>"""


def build_nfl_ats_section_html(
    ats_df: pd.DataFrame, chart_rel_path: Optional[str], top_n: int = 12
) -> str:
    """NFL ATS standings table (top & bottom) + embedded chart."""
    if ats_df.empty:
        return (
            '<div class="card"><p class="data-note">NFL ATS standings unavailable '
            "(source stats file missing or empty).</p></div>"
        )

    def _rows(frame: pd.DataFrame, start_rank: int) -> str:
        out = ""
        for offset, (_, row) in enumerate(frame.iterrows()):
            margin = row["Avg ATS Margin"]
            margin_str = f"{margin:+.1f}" if pd.notna(margin) else "—"
            push_str = f" ({int(row['Pushes'])}P)" if row["Pushes"] else ""
            out += (
                f"<tr><td>{start_rank + offset}</td><td>{row['Team']}</td>"
                f"<td><strong>{row['Cover %']}%</strong></td>"
                f"<td>{int(row['ATS Wins'])}-{int(row['ATS Losses'])}{push_str}</td>"
                f"<td>{margin_str}</td></tr>"
            )
        return out

    n = len(ats_df)
    half = min(top_n, n)
    top_rows = _rows(ats_df.head(half), 1)
    table_html = f"""
            <h3>Best ATS Teams</h3>
            <table>
                <tr><th>Rank</th><th>Team</th><th>Cover %</th><th>ATS Record</th><th>Avg Margin</th></tr>
                {top_rows}
            </table>"""

    if n > top_n:
        bottom = ats_df.tail(min(top_n, n - half))
        bottom_rows = _rows(bottom, n - len(bottom) + 1)
        table_html += f"""
            <h3>Worst ATS Teams</h3>
            <table>
                <tr><th>Rank</th><th>Team</th><th>Cover %</th><th>ATS Record</th><th>Avg Margin</th></tr>
                {bottom_rows}
            </table>"""

    img_html = ""
    if chart_rel_path:
        img_html = (
            f'<img src="{chart_rel_path}" alt="NFL ATS cover % chart" '
            'style="width:100%;height:auto;border-radius:8px;margin-top:12px;">'
        )

    return f"""
        <div class="card">
            {table_html}
            <p style="font-size:11px;color:#666;margin-top:8px;">
                Cover % = ATS wins / (wins + losses), pushes excluded.
                Avg Margin = average of (final margin + closing spread); positive means
                covering with room to spare. Source: bundled NFL game stats (2023-2025).
            </p>
            {img_html}
        </div>"""


def build_nba_ats_note_html() -> str:
    """Labeled placeholder: NBA point-spread cover records are not computable."""
    return """
        <div class="card">
            <p class="data-note">
                <strong>NBA team ATS records are not shown.</strong> The committed NBA
                odds snapshots record spread <em>odds</em> (the juice, e.g. -110) but not
                the spread <em>line</em> (the point handicap), so a true against-the-spread
                cover record cannot be computed from available data without fabricating the
                missing lines. The NBA bookmaker accuracy leaderboard above is computed from
                real moneyline outcomes.
            </p>
        </div>"""
