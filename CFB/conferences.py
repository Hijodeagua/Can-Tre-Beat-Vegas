"""Per-season conference clusters recovered from schedule structure.

The standings pull (DATA_PULL_PLAN.md §3.1) is Cloudflare-blocked for
automated access, but the thing it unlocks — a per-season conference map for
conference-mean regression — is recoverable from the schedule alone:
conference teams play 8-9 games against each other and only 3-4 outside, so
conferences are exactly the dense communities in each season's game graph.

Modularity-based local moving (one-level Louvain), deterministic order.
Plain label propagation was tried first and merged the P5 conferences into
two mega-clusters (its known resolution problem); modularity recovers the
individual conferences. Independents (Notre Dame, Army, BYU pre-2023) get
absorbed into whichever community they play most — acceptable noise for a
regression target, worth remembering when reading the clusters.
Realignment is handled for free because every season is clustered
separately.

Usage
    python3 -m CFB.conferences          # writes agg/cfb_conference_clusters.csv
    python3 -m CFB.conferences --show 2025
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
AGG_DIR = REPO_ROOT / "data" / "college_football" / "agg"

MAX_SWEEPS = 50


def _cluster_graph(edges: list[tuple[str, str]]) -> dict[str, int]:
    """One-level Louvain: greedy modularity local moving, deterministic."""
    weight: dict[str, Counter] = defaultdict(Counter)
    for a, b in edges:
        weight[a][b] += 1
        weight[b][a] += 1

    teams = sorted(weight)
    degree = {t: sum(weight[t].values()) for t in teams}
    two_m = float(sum(degree.values()))

    comm = {t: i for i, t in enumerate(teams)}
    comm_degree = {comm[t]: float(degree[t]) for t in teams}

    for _ in range(MAX_SWEEPS):
        changed = False
        for t in teams:
            old = comm[t]
            # Weight from t into each neighbouring community.
            links: Counter = Counter()
            for n, w in weight[t].items():
                links[comm[n]] += w
            comm_degree[old] -= degree[t]

            # Modularity gain of joining community c:
            #   k_{t,in}(c) - k_t * sum_deg(c) / 2m
            best_c, best_gain = old, links.get(old, 0) - degree[t] * comm_degree.get(old, 0.0) / two_m
            for c in sorted(links):
                gain = links[c] - degree[t] * comm_degree.get(c, 0.0) / two_m
                if gain > best_gain + 1e-12:
                    best_c, best_gain = c, gain

            comm_degree[best_c] = comm_degree.get(best_c, 0.0) + degree[t]
            if best_c != old:
                comm[t] = best_c
                changed = True
        if not changed:
            break
    return comm


def cluster_season(games: pd.DataFrame, season: int) -> dict[str, int]:
    """Cluster one season's FBS-vs-FBS regular-season games."""
    g = games[
        (games["season"] == season)
        & (games["game_type"] == "REG")
        & (games["home_team"] != "FCS")
        & (games["away_team"] != "FCS")
    ]
    return _cluster_graph(list(zip(g["home_team"], g["away_team"])))


def build_clusters(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in sorted(games["season"].unique()):
        labels = cluster_season(games, int(season))
        # Renumber labels 0..n per season, largest cluster first.
        sizes = Counter(labels.values())
        order = {
            old: new
            for new, (old, _) in enumerate(
                sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))
            )
        }
        rows += [
            {"season": int(season), "team": t, "cluster": order[lbl]}
            for t, lbl in sorted(labels.items())
        ]
    return pd.DataFrame(rows)


def load_clusters() -> dict[tuple[int, str], int]:
    """(season, team) -> cluster id, for the Elo season roll."""
    path = AGG_DIR / "cfb_conference_clusters.csv"
    df = pd.read_csv(path)
    return {
        (int(r.season), r.team): int(r.cluster) for r in df.itertuples(index=False)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, help="print one season's clusters")
    args = ap.parse_args()

    from CFB import elo

    games = elo.load_games(pool=True)
    clusters = build_clusters(games)
    out = AGG_DIR / "cfb_conference_clusters.csv"
    clusters.to_csv(out, index=False)
    n = clusters.groupby("season")["cluster"].nunique()
    print(f"wrote {out}")
    print(f"clusters per season: min {n.min()}, max {n.max()} (expect ~9-13)")

    if args.show is not None:
        season = clusters[clusters["season"] == args.show]
        for cid, grp in season.groupby("cluster"):
            print(f"\ncluster {cid} ({len(grp)}):")
            print("  " + ", ".join(grp["team"]))


if __name__ == "__main__":
    main()
