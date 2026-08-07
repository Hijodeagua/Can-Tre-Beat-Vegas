"""Charts for the game-level (head-to-head) prior-vs-same-season analysis."""

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
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GOLD = "#eda100"
GREEN = "#1baf7a"

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


def oracle_vs_prior_chart():
    ovp = pd.read_csv(AN / "game_level_oracle_vs_prior.csv")
    order = (ovp[ovp.kind == "same"].sort_values("point_biserial_r")
             ["stat"].tolist())
    same = ovp[ovp.kind == "same"].set_index("stat").loc[order]
    prior = ovp[ovp.kind == "prior"].set_index("stat").loc[order]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.5))
    y = np.arange(len(order))

    ax = axes[0]
    ax.barh(y + 0.19, same.point_biserial_r, height=0.36, color=BLUE,
            label="same-season FINAL (oracle / look-ahead)")
    ax.barh(y - 0.19, prior.point_biserial_r, height=0.36, color=ORANGE,
            label="prior-season FINAL (legitimate, zero look-ahead)")
    ax.set_yticks(y, order)
    ax.set_xlabel("Point-biserial r vs. actual home win")
    ax.set_title("Head-to-head game prediction, per stat", loc="left",
                 color=INK)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=BASE, lw=0.8)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5,
              labelcolor=INK2)

    ax2 = axes[1]
    ax2.barh(y + 0.19, same.accuracy, height=0.36, color=BLUE)
    ax2.barh(y - 0.19, prior.accuracy, height=0.36, color=ORANGE)
    ax2.set_yticks(y, [])
    ax2.set_xlim(0.5, 0.60)
    ax2.axvline(0.5, color=BASE, lw=1, ls=(0, (4, 2)))
    ax2.set_xlabel("Single-feature logistic regression accuracy")
    ax2.set_title("Same axis, as game-win accuracy", loc="left", color=INK)
    ax2.grid(axis="y", visible=False)

    fig.suptitle("Does LAST season's stat predict a game better than THIS "
                 "season's final stat? (it doesn't — same-season is an "
                 "oracle)", x=0.02, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(CH / "mlb_h2h_oracle_vs_prior.png", dpi=150)
    plt.close(fig)


def crossover_chart():
    rvp = pd.read_csv(AN / "game_level_rolling_vs_prior.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))

    ax = axes[0]
    ax.plot(rvp.games_played_cutoff, rvp.r_prior, color=ORANGE, lw=2,
            marker="o", ms=5, label="prior-season FINAL RunDiff/G")
    ax.plot(rvp.games_played_cutoff, rvp.r_roll, color=BLUE, lw=2,
            marker="o", ms=5,
            label="THIS season, rolling to date (no look-ahead)")
    ax.set_xlabel("Team games played so far this season (both teams ≥ cutoff)")
    ax.set_ylabel("Point-biserial r vs. home win")
    ax.set_title("When does in-season form overtake last year's record?",
                 loc="left", color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK2)
    ax.set_ylim(0.08, 0.19)

    ax2 = axes[1]
    ax2.plot(rvp.games_played_cutoff, rvp.combo_beta_prior, color=ORANGE,
             lw=2, marker="o", ms=5, label="weight on prior-season prior")
    ax2.plot(rvp.games_played_cutoff, rvp.combo_beta_roll, color=BLUE,
             lw=2, marker="o", ms=5, label="weight on in-season rolling")
    ax2.set_xlabel("Team games played so far this season")
    ax2.set_ylabel("Logistic regression coefficient (combined model)")
    ax2.set_title("Combined model: the prior never quite dies out",
                  loc="left", color=INK)
    ax2.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK2)
    ax2.axhline(0, color=BASE, lw=0.8)

    fig.suptitle("Prior-season record vs. this season's actual results, "
                 "run differential", x=0.02, ha="left", fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(CH / "mlb_h2h_crossover.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    oracle_vs_prior_chart()
    crossover_chart()
    print("game-level charts written to", CH)
