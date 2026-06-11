"""
Monte Carlo simulation of the 2026 World Cup group stage.

Builds match win/draw/loss probabilities from the trained outcome model
(falling back to pure Elo if no trained model exists), then simulates the
72 group-stage fixtures N times to estimate, per team:

- P(win group), P(top 2), P(advance) — top 2 of each of the 12 groups plus
  the 8 best third-placed teams advance in the 48-team format
- expected points

Groups are derived from the fixture graph (each group of 4 is a closed
cluster of 6 matches), so no hand-maintained group table is needed.

Tiebreak approximation: within a group, ranking is by points, then a
simulated goal-difference proxy (margins sampled from a Poisson scaled by
Elo gap), then a small random jitter standing in for goals-scored /
fair-play tiebreakers. Third-place comparison uses points then the same
GD proxy.

Usage:
    python -m soccer.model.simulate [--sims 10000] [--seed 42]
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from soccer.model.elo import HOME_ADVANTAGE, load_results, run_history
from soccer.model.squad import attach_squad_features
from soccer.model.train import FEATURES, KNOCKOUT_TOURNAMENTS

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

GROUP_NAMES = "ABCDEFGHIJKL"


def group_fixtures() -> pd.DataFrame:
    df = load_results()
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["home_score"].isna())]
    return wc.reset_index(drop=True)


def derive_groups(fixtures: pd.DataFrame) -> dict:
    """Connected components of the who-plays-whom graph -> groups of 4."""
    adj = defaultdict(set)
    for _, row in fixtures.iterrows():
        adj[row["home_team"]].add(row["away_team"])
        adj[row["away_team"]].add(row["home_team"])
    seen, groups = set(), []
    for team in adj:
        if team in seen:
            continue
        comp, stack = set(), [team]
        while stack:
            t = stack.pop()
            if t in comp:
                continue
            comp.add(t)
            stack.extend(adj[t] - comp)
        seen |= comp
        groups.append(sorted(comp))
    if any(len(g) != 4 for g in groups) or len(groups) != 12:
        raise SystemExit(f"Expected 12 groups of 4, got sizes {[len(g) for g in groups]}")
    # Stable naming: groups ordered by their earliest fixture date
    first_date = {}
    for _, row in fixtures.iterrows():
        for g_idx, g in enumerate(groups):
            if row["home_team"] in g:
                first_date.setdefault(g_idx, row["date"])
    order = sorted(range(12), key=lambda i: (first_date.get(i, "9999"), groups[i][0]))
    return {GROUP_NAMES[rank]: groups[i] for rank, i in enumerate(order)}


def match_probabilities(fixtures: pd.DataFrame) -> pd.DataFrame:
    """W/D/L probabilities per fixture from the trained model, or pure Elo."""
    engine, _ = run_history()
    rows = []
    for _, row in fixtures.iterrows():
        adv = 0.0 if bool(row["neutral"]) else HOME_ADVANTAGE
        rows.append(
            {
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "elo_home": engine.get(row["home_team"]),
                "elo_away": engine.get(row["away_team"]),
                "elo_gap": (engine.get(row["home_team"]) + adv) - engine.get(row["away_team"]),
                "host_home": int(row["country"] == row["home_team"]),
                "knockout": int(row["tournament"] in KNOCKOUT_TOURNAMENTS),
            }
        )
    table = attach_squad_features(pd.DataFrame(rows))

    model_path = ARTIFACTS / "outcome_model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)["model"]
        probs = model.predict_proba(table[FEATURES])
        classes = list(model.classes_)
        table["p_home"] = probs[:, classes.index("H")]
        table["p_draw"] = probs[:, classes.index("D")]
        table["p_away"] = probs[:, classes.index("A")]
        print("Using trained outcome model (Elo + squad + venue features)")
    else:
        # Pure-Elo fallback: expected score split into W/D/L with a fixed
        # draw share that shrinks as the matchup gets lopsided.
        exp = 1 / (1 + 10 ** (-table["elo_gap"] / 400))
        draw = 0.27 * (1 - (exp - 0.5).abs() * 2 * 0.6)
        table["p_draw"] = draw
        table["p_home"] = exp * (1 - draw)
        table["p_away"] = (1 - exp) * (1 - draw)
        print("No trained model found — using pure Elo probabilities")
    return table


def simulate(table: pd.DataFrame, groups: dict, n_sims: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    team_group = {t: g for g, teams in groups.items() for t in teams}
    teams = sorted(team_group)
    idx = {t: i for i, t in enumerate(teams)}

    p = table[["p_home", "p_draw", "p_away"]].to_numpy()
    home_i = table["home_team"].map(idx).to_numpy()
    away_i = table["away_team"].map(idx).to_numpy()
    # Poisson margin scale per match (for the GD tiebreak proxy)
    margin_lam = np.maximum(0.35, np.abs(table["elo_gap"].to_numpy()) / 350)

    n_teams = len(teams)
    win_group = np.zeros(n_teams)
    top_two = np.zeros(n_teams)
    advance = np.zeros(n_teams)
    total_points = np.zeros(n_teams)

    group_members = {g: np.array([idx[t] for t in ts]) for g, ts in groups.items()}

    for _ in range(n_sims):
        pts = np.zeros(n_teams)
        gd = np.zeros(n_teams)
        u = rng.random(len(p))
        outcome = (u > p[:, 0]).astype(int) + (u > p[:, 0] + p[:, 1]).astype(int)
        margins = 1 + rng.poisson(margin_lam)
        for m in range(len(p)):
            h, a = home_i[m], away_i[m]
            if outcome[m] == 0:      # home win
                pts[h] += 3
                gd[h] += margins[m]
                gd[a] -= margins[m]
            elif outcome[m] == 1:    # draw
                pts[h] += 1
                pts[a] += 1
            else:                    # away win
                pts[a] += 3
                gd[a] += margins[m]
                gd[h] -= margins[m]

        thirds = []
        for g, members in group_members.items():
            key = pts[members] * 1e6 + gd[members] * 1e2 + rng.random(4)
            order = members[np.argsort(-key)]
            win_group[order[0]] += 1
            top_two[order[:2]] += 1
            advance[order[:2]] += 1
            thirds.append(order[2])
        thirds = np.array(thirds)
        tkey = pts[thirds] * 1e6 + gd[thirds] * 1e2 + rng.random(12)
        advance[thirds[np.argsort(-tkey)][:8]] += 1
        total_points += pts

    out = pd.DataFrame(
        {
            "team": teams,
            "group": [team_group[t] for t in teams],
            "p_win_group": win_group / n_sims,
            "p_top_two": top_two / n_sims,
            "p_advance": advance / n_sims,
            "exp_points": total_points / n_sims,
        }
    )
    return out.sort_values(["group", "p_advance"], ascending=[True, False]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fixtures = group_fixtures()
    groups = derive_groups(fixtures)
    table = match_probabilities(fixtures)
    results = simulate(table, groups, args.sims, args.seed)

    ARTIFACTS.mkdir(exist_ok=True)
    out_path = ARTIFACTS / "group_stage_sim.csv"
    results.round(4).to_csv(out_path, index=False)
    print(f"\n{args.sims:,} simulations -> {out_path}\n")
    for g in GROUP_NAMES:
        sub = results[results["group"] == g]
        print(f"Group {g}")
        for _, r in sub.iterrows():
            print(
                f"  {r['team']:<24} win grp {r['p_win_group']*100:5.1f}%  "
                f"top2 {r['p_top_two']*100:5.1f}%  advance {r['p_advance']*100:5.1f}%  "
                f"xPts {r['exp_points']:.2f}"
            )
        print()


if __name__ == "__main__":
    main()
