"""
Render report charts for the MLB stat <-> Elo association study.

Static PNGs (repo convention: reports/charts/). Colors follow the validated
reference palette (light mode) from the dataviz skill: categorical slots
1-4 = blue/orange/aqua/yellow, chart chrome in muted ink.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
AN = REPO / "data" / "mlb" / "analysis"
CH = REPO / "reports" / "charts"
CH.mkdir(parents=True, exist_ok=True)

# reference palette (light mode)
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
    "savefig.facecolor": SURFACE,
    "font.family": "Segoe UI",
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "font.size": 9,
})


def _style_barh(ax):
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=BASE, lw=0.8)


def corr_bars():
    ct = pd.read_csv(AN / "correlations.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.5), sharey=False)
    for ax, outcome, title in [
        (axes[0], "wl", "vs same-season win %"),
        (axes[1], "elo_delta", "vs same-season Elo change"),
    ]:
        d = ct[ct.outcome == outcome].copy()
        d = d.reindex(d.pearson_r.abs().sort_values().index)
        colors = d.group.map(C)
        ax.barh(d.stat, d.pearson_r, color=colors, height=0.62)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Pearson r (within-season z-scores)")
        ax.set_xlim(-1, 1)
        _style_barh(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C[g])
               for g in ["hitting", "pitching", "advanced", "combined"]]
    axes[0].legend(handles, ["Hitting", "Pitching", "Advanced (WPA/RE24)",
                             "Run differential"],
                   loc="lower right", frameon=False, fontsize=8,
                   labelcolor=INK2)
    fig.suptitle("Which team stats track outcomes? (2009–2026, n=540 "
                 "team-seasons)", x=0.02, ha="left", fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(CH / "mlb_corr_outcomes.png", dpi=150)
    plt.close(fig)


def key_scatters():
    m = pd.read_csv(AN / "merged_panel.csv")
    z = m.copy()
    for c in ["OPS+", "ERA+", "RunDiff/G", "WPA/G", "wl", "elo_delta"]:
        grp = z.groupby("Season")[c]
        z[c] = (z[c] - grp.transform("mean")) / grp.transform("std")
    pairs = [("OPS+", "wl", "OPS+ vs win %", C["hitting"]),
             ("ERA+", "wl", "ERA+ vs win %", C["pitching"]),
             ("RunDiff/G", "elo_delta", "Run diff/G vs Elo change",
              C["combined"]),
             ("WPA/G", "wl", "WPA/G vs win % (circular by construction)",
              C["advanced"])]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))
    for ax, (x, y, title, color) in zip(axes.flat, pairs):
        sub = z[[x, y]].dropna()
        ax.scatter(sub[x], sub[y], s=14, color=color, alpha=0.55,
                   edgecolors=SURFACE, linewidths=0.5)
        b = np.polyfit(sub[x], sub[y], 1)
        xs = np.linspace(sub[x].min(), sub[x].max(), 50)
        ax.plot(xs, np.polyval(b, xs), color=INK2, lw=2)
        r = np.corrcoef(sub[x], sub[y])[0, 1]
        ax.set_title(f"{title}  (r = {r:.2f})", loc="left", color=INK)
        ax.set_xlabel(f"{x} (z)")
        ax.set_ylabel(f"{y} (z)")
    fig.suptitle("Key stat–outcome relationships, all z-scored within "
                 "season", x=0.02, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(CH / "mlb_key_scatters.png", dpi=150)
    plt.close(fig)


def yoy_chart():
    yt = pd.read_csv(AN / "yoy_deltas.csv")
    m = pd.read_csv(AN / "merged_panel.csv").sort_values(
        ["franchise", "Season"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 6),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    d = yt.reindex(yt.r_delta_elo.abs().sort_values().index).tail(16)
    axes[0].barh(d.stat, d.r_delta_elo, color=d.group.map(C), height=0.62)
    axes[0].set_title("Year-over-year: Δstat vs ΔElo (end of "
                      "season)", loc="left", color=INK)
    axes[0].set_xlabel("Pearson r on YoY deltas")
    axes[0].set_xlim(-1, 1)
    _style_barh(axes[0])

    dd = m.copy()
    for c in ["RunDiff/G", "elo_end"]:
        dd[f"d_{c}"] = dd.groupby("franchise")[c].diff()
    dd["prev"] = dd.groupby("franchise")["Season"].shift()
    dd = dd[(dd.prev == dd.Season - 1) & (dd.Season != 2020)
            & (dd.prev != 2020)].dropna(subset=["d_RunDiff/G", "d_elo_end"])
    axes[1].scatter(dd["d_RunDiff/G"], dd["d_elo_end"], s=14,
                    color=C["combined"], alpha=0.6,
                    edgecolors=SURFACE, linewidths=0.5)
    b = np.polyfit(dd["d_RunDiff/G"], dd["d_elo_end"], 1)
    xs = np.linspace(dd["d_RunDiff/G"].min(), dd["d_RunDiff/G"].max(), 50)
    axes[1].plot(xs, np.polyval(b, xs), color=INK2, lw=2)
    r = np.corrcoef(dd["d_RunDiff/G"], dd["d_elo_end"])[0, 1]
    axes[1].set_title(f"Δ run diff/G vs Δ Elo  (r = {r:.2f})",
                      loc="left", color=INK)
    axes[1].set_xlabel("Change in run differential per game")
    axes[1].set_ylabel("Change in end-of-season Elo")
    fig.suptitle("Do year-to-year stat changes move Elo?", x=0.02,
                 ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(CH / "mlb_yoy_deltas.png", dpi=150)
    plt.close(fig)


def persistence_chart():
    pt = pd.read_csv(AN / "persistence.csv")
    elo_r = float(pt.loc[pt.stat == "elo_end", "r_next_wl"].iloc[0])
    d = pt[pt.stat != "elo_end"].copy()
    d = d.reindex(d.r_next_wl.abs().sort_values(ascending=False).index)
    d = d.head(14).iloc[::-1]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(y + 0.19, d.r_next_wl, height=0.36, color="#2a78d6",
            label="raw r with next-season win %")
    ax.barh(y - 0.19, d.r_next_wl_given_elo, height=0.36, color="#eb6834",
            label="partial r, controlling for current Elo")
    ax.axvline(elo_r, color=C["elo"], lw=2, ls=(0, (4, 2)))
    ax.text(elo_r + 0.01, len(d) - 0.4, f"Elo itself\nr = {elo_r:.2f}",
            color=C["elo"], fontsize=8, fontweight="bold")
    ax.set_yticks(y, d.stat)
    ax.set_xlabel("Correlation with next season's win % "
                  "(z-scores, 2020 pairs excluded)")
    ax.set_title("Do stats predict NEXT season beyond Elo?", loc="left",
                 color=INK)
    ax.set_xlim(-0.65, 0.65)
    _style_barh(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=8,
              labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(CH / "mlb_persistence.png", dpi=150)
    plt.close(fig)


def calibration_chart():
    cal = pd.read_csv(AN / "calibration.csv")
    with open(REPO / "data" / "mlb" / "elo_params.json") as fh:
        best = json.load(fh)["best"]
    fig, ax = plt.subplots(figsize=(6.4, 6))
    lim = (0.28, 0.75)
    ax.plot(lim, lim, color=BASE, lw=1, ls=(0, (4, 2)))
    ax.plot(cal.p_pred, cal.p_actual, color="#2a78d6", lw=2, zorder=3)
    ax.scatter(cal.p_pred, cal.p_actual,
               s=np.clip(cal.n / 60, 12, 90), color="#2a78d6",
               edgecolors=SURFACE, linewidths=0.8, zorder=4)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Elo predicted home win probability")
    ax.set_ylabel("Actual home win rate")
    ax.set_title(
        "Elo calibration, 2012–2026  "
        f"(log loss {best['log_loss']} vs {best['baseline_log_loss']} "
        "home-rate baseline)", loc="left", color=INK)
    ax.text(0.30, 0.72, "dot size = games in bin", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(CH / "mlb_calibration.png", dpi=150)
    plt.close(fig)


def elo_history_chart():
    hist = pd.read_csv(REPO / "data" / "mlb" / "elo_game_history.csv")
    teams = ["LAD", "NYY", "HOU", "COL"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    slot = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for t, color in zip(teams, slot):
        rows = []
        for side, col in [("home", "elo_home_pre"), ("away", "elo_away_pre")]:
            sub = hist[hist[side] == t][["date", col]].rename(
                columns={col: "elo"})
            rows.append(sub)
        d = (pd.concat(rows).sort_values("date").reset_index(drop=True))
        d["date"] = pd.to_datetime(d["date"])
        d["elo_s"] = d.elo.rolling(30, min_periods=1).mean()
        ax.plot(d.date, d.elo_s, color=color, lw=2)
        ax.text(d.date.iloc[-1] + pd.Timedelta(days=25),
                d.elo_s.iloc[-1], t, color=color, fontweight="bold",
                fontsize=9, va="center")
    ax.axhline(1500, color=BASE, lw=1, ls=(0, (4, 2)))
    ax.set_title("Betting-blind Elo, 30-game rolling mean "
                 "(fresh 1500 start in 2009)", loc="left", color=INK)
    ax.set_ylabel("Elo rating")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(CH / "mlb_elo_history.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    corr_bars()
    key_scatters()
    yoy_chart()
    persistence_chart()
    calibration_chart()
    elo_history_chart()
    print("charts written to", CH)
