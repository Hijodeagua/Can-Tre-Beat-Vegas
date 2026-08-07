"""Charts for the extended analysis: additional stats, prior-season
effects, and the rank-vs-value question."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
AN = REPO / "data" / "mlb" / "analysis"
CH = REPO / "reports" / "charts"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
C = {"hitting": "#2a78d6", "pitching": "#eb6834",
     "advanced": "#1baf7a", "combined": "#eda100", "elo": "#4a3aa7"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "Segoe UI",
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9,
})


def _style_barh(ax):
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=BASE, lw=0.8)


def extra_stats_chart():
    ex = pd.read_csv(AN / "extra_stat_correlations.csv")
    d = ex[ex.outcome == "wl"].copy()
    d = d.reindex(d.pearson_r.abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(d.stat, d.pearson_r, color=d.group.map(C), height=0.62)
    ax.set_title("Previously-untested stats vs win % (2009–2026)",
                 loc="left", color=INK)
    ax.set_xlabel("Pearson r (within-season z-scores)")
    ax.set_xlim(-1, 1)
    _style_barh(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C[g])
               for g in ["hitting", "pitching", "advanced"]]
    ax.legend(handles, ["Hitting", "Pitching", "Advanced"],
              loc="lower right", frameon=False, fontsize=8,
              labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(CH / "mlb_extra_stats.png", dpi=150)
    plt.close(fig)


def prior_season_chart():
    ps = pd.read_csv(AN / "prior_season.csv")
    elo_r = float(ps.loc[ps.stat == "elo_start", "r_prevstat_vs_curwl"]
                  .iloc[0])
    d = ps[ps.stat != "elo_start"].copy()
    d = d.reindex(d.r_prevstat_vs_curwl.abs()
                 .sort_values(ascending=False).index).head(14).iloc[::-1]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.barh(y + 0.19, d.r_prevstat_vs_curwl, height=0.36,
            color="#2a78d6", label="raw r, prior-season stat vs this year's win %")
    ax.barh(y - 0.19, d.r_given_elo_start, height=0.36,
            color="#eb6834",
            label="partial r, controlling for Elo entering the season")
    ax.axvline(elo_r, color=C["elo"], lw=2, ls=(0, (4, 2)))
    ax.text(elo_r + 0.01, len(d) - 0.4, f"Elo entering season\nr = {elo_r:.2f}",
            color=C["elo"], fontsize=8, fontweight="bold")
    ax.set_yticks(y, d.stat)
    ax.set_xlabel("Correlation with THIS season's win % "
                  "(2020-adjacent pairs excluded)")
    ax.set_title("Does last season's stat level predict this season?",
                 loc="left", color=INK)
    ax.set_xlim(-0.65, 0.65)
    _style_barh(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5,
              labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(CH / "mlb_prior_season.png", dpi=150)
    plt.close(fig)


def rank_vs_value_chart():
    rv = pd.read_csv(AN / "rank_vs_value.csv")
    d = rv[rv.outcome == "wl"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5),
                             gridspec_kw={"width_ratios": [1, 1.05]})

    ax = axes[0]
    lim = (-0.95, 0.95)
    ax.plot(lim, lim, color=BASE, lw=1, ls=(0, (4, 2)), zorder=1)
    ax.scatter(d.pearson_value_r, d.pearson_rank_r, s=32,
               color="#2a78d6", edgecolors=SURFACE, linewidths=0.6,
               zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Pearson r on raw value (z-score)")
    ax.set_ylabel("Pearson r on within-season rank (= Spearman)")
    ax.set_title("Rank vs. raw value: same predictive power",
                 loc="left", color=INK)
    ax.text(-0.9, 0.8, "dots on the line = rank adds nothing\n"
            "beyond the raw number", color=MUTED, fontsize=8)

    ax2 = axes[1]
    d2 = d.reindex(d.gap.abs().sort_values(ascending=False).index).head(12)
    d2 = d2.iloc[::-1]
    ax2.barh(d2.stat, d2.gap, color="#eda100", height=0.6)
    ax2.set_title("Largest rank−value gaps (still tiny)", loc="left",
                  color=INK)
    ax2.set_xlabel("Spearman r − Pearson r")
    ax2.set_xlim(-0.08, 0.08)
    _style_barh(ax2)

    fig.suptitle("Does a team's ORDINAL RANK on a stat matter more than "
                 "its raw value?", x=0.02, ha="left", fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(CH / "mlb_rank_vs_value.png", dpi=150)
    plt.close(fig)


def quartile_chart():
    qb = pd.read_csv(AN / "quartile_buckets.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    colors = {"RunDiff/G": "#eda100", "OPS+": "#2a78d6", "ERA+": "#eb6834"}
    for ax, stat in zip(axes, ["RunDiff/G", "OPS+", "ERA+"]):
        d = qb[qb.stat == stat]
        ax.bar(d.quartile.astype(str), d.wl, color=colors[stat], width=0.6)
        for x, y, n in zip(d.quartile.astype(str), d.wl, d.n):
            ax.text(x, y + 0.008, f"{y:.3f}", ha="center", fontsize=8,
                    color=INK2)
        ax.set_title(stat, loc="left", color=INK)
        ax.set_xlabel("Within-season quartile (1=worst, 4=best)")
        ax.set_ylim(0.35, 0.65)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Average win %")
    fig.suptitle("Linearity check: win % by stat quartile "
                 "(near-equal steps = no rank premium)", x=0.02, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(CH / "mlb_quartile_linearity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    extra_stats_chart()
    prior_season_chart()
    rank_vs_value_chart()
    quartile_chart()
    print("extended charts written to", CH)
