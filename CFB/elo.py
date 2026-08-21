"""Base head-to-head Elo for college football.

Reuses the NFL engine (``NFL/model/v2/elo.py``) — same MOV-adjusted update,
same walk-forward discipline — with CFB constants **fitted by
``python3 -m CFB.fit``** (grid search on walk-forward log loss, scored on
2005+ with 2000-2004 as burn-in; see ``data/college_football/agg/fit_grid.csv``).

How the fitted values landed vs the plan's §8 guesses:

- **K = 35** — top of the guessed 20-40 range; CFB ratings really do move.
- **HFA = 50 Elo** — *lower* than the NFL's 55 in Elo, but at the fitted
  18.6 Elo/point it is **~2.7 points**, more than the NFL's ~2.2, matching
  the plan's expectation in the units that matter.
- **Season regression = 0.35 toward the conference-cluster mean** (see
  ``CFB.conferences`` — clusters recovered from schedule structure, no
  standings pull needed). The plan guessed 0.35-0.50 and was right *given
  a meaningful target*: under flat-1500 regression the optimum collapses
  to 0.20, because heavy regression toward a wrong target destroys
  information. Cluster mode is the default whenever the cluster map
  exists; ``--flat`` restores flat regression.
- **No margin cap** — the engine's log MOV damping already discounts
  blowouts; every capped config graded worse.
- **FCS opponents pooled** into one synthetic ``FCS`` team (plan §3.4);
  see ``pool_fcs`` for how FBS membership is decided.
- **Postseason** (bowls + conference championships) gets the playoff K
  multiplier, and bowls are treated as neutral-site (they are).

Usage
    python3 -m CFB.ingest                # build the normalised spine first
    python3 -m CFB.elo                   # ratings board to stdout + CSV
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from NFL.model.v2.elo import BASE_RATING as _BASE, EloEngine

REPO_ROOT = Path(__file__).resolve().parents[1]
AGG_DIR = REPO_ROOT / "data" / "college_football" / "agg"

# Fitted by CFB.fit on walk-forward log loss, 2005+ (grid in agg/fit_grid.csv),
# under the default conference-cluster regression.
BASE_RATING = 1500.0
K_FACTOR = 35.0
POSTSEASON_K_MULT = 1.2
HFA_ELO = 50.0
SEASON_REGRESSION = 0.35  # toward cluster mean; flat-1500 mode optimises at 0.20
ELO_PER_POINT = 16.5  # margin regression through the origin, 2005+
MARGIN_CAP = 999.0  # grid preferred uncapped; log MOV damping suffices

POSTSEASON_TYPES = {"BOWL", "CCG"}

FCS_TEAM = "FCS"
FBS_MIN_GAMES = 5  # a team with this many schedule games in a season is FBS
BURN_IN_THROUGH = 2004  # plan §2.1: evaluate on 2005+

EVAL_FROM = BURN_IN_THROUGH + 1


def pool_fcs(
    games: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """Replace non-FBS opponents with one synthetic ``FCS`` team (plan §3.4).

    A team is FBS-in-a-season if **either** holds:

    - the offense scoring aggregate lists it for that season, or
    - it appears in ``FBS_MIN_GAMES``+ games that season.

    The second clause exists because CFR's stat pages abbreviate some
    schools (``LSU``, ``USC``, ``UCF``, ``BYU``) while the schedule exports
    spell them out (``Louisiana State``, ``Southern California``, …). A
    name-mismatched FBS team plays a full schedule; a genuine FCS opponent
    appears in at most a handful of games (these exports contain FBS games
    only), so the game count separates them cleanly — including 2020's
    shortened seasons. ``verbose`` prints the mismatch suspects, which is
    the raw material for the plan §3.3 crosswalk.

    Known accepted flaw (plan §3.4): the pool treats North Dakota State the
    same as an FCS bottom-feeder.
    """
    df = games.copy()

    fbs: dict[int, set[str]] = {}
    off_path = AGG_DIR / "cfb_offense_team_season.csv"
    if off_path.exists():
        off = pd.read_csv(off_path)
        fbs = off.groupby("season")["team"].agg(set).to_dict()

    counts: dict[int, pd.Series] = {
        int(season): pd.concat([grp["home_team"], grp["away_team"]]).value_counts()
        for season, grp in df.groupby("season")
    }

    empty: set[str] = set()
    suspects: set[str] = set()
    for col in ("home_team", "away_team"):
        is_fcs = []
        for s, t in zip(df["season"], df[col]):
            s = int(s)
            in_stats = t in fbs.get(s, empty)
            plays_full = int(counts[s].get(t, 0)) >= FBS_MIN_GAMES
            if plays_full and not in_stats:
                suspects.add(t)
            is_fcs.append(not in_stats and not plays_full)
        df[col] = np.where(is_fcs, FCS_TEAM, df[col])

    if verbose and suspects:
        print(
            f"stat-page name mismatches kept as FBS via game count "
            f"({len(suspects)}): {', '.join(sorted(suspects))}"
        )

    # An FCS-vs-FCS pairing would have the synthetic team play itself;
    # there should be none (CFR season schedules are FBS games), but drop
    # defensively rather than corrupt the engine.
    both = (df["home_team"] == FCS_TEAM) & (df["away_team"] == FCS_TEAM)
    return df[~both].reset_index(drop=True)


class ClusterRegressEngine(EloEngine):
    """Season roll regresses toward the team's *conference-cluster* mean.

    ``clusters`` maps ``(season, team) -> cluster id`` (see
    ``CFB.conferences``); membership of the season being entered is used,
    which is known before kickoff. The target for a cluster is the mean of
    its incoming members' carried-over ratings; teams without a cluster
    that season (the pooled FCS side, programs entering FBS) regress toward
    1500 exactly as the flat engine does.
    """

    def __init__(self, clusters: dict[tuple[int, str], int], **kwargs):
        super().__init__(**kwargs)
        self.clusters = clusters

    def _roll_season(self, season: int) -> None:
        if self._last_season is not None and season != self._last_season:
            groups: dict[int, list[float]] = defaultdict(list)
            for team, r in self.ratings.items():
                c = self.clusters.get((season, team))
                if c is not None:
                    groups[c].append(r)
            means = {c: sum(v) / len(v) for c, v in groups.items()}
            for team, r in self.ratings.items():
                c = self.clusters.get((season, team))
                target = means[c] if c in means else _BASE
                self.ratings[team] = target + (1 - self.regression) * (r - target)
        self._last_season = season


def walk_forward(
    games: pd.DataFrame,
    *,
    k: float = K_FACTOR,
    hfa: float = HFA_ELO,
    regression: float = SEASON_REGRESSION,
    margin_cap: float = MARGIN_CAP,
    clusters: dict[tuple[int, str], int] | None = None,
) -> tuple[pd.DataFrame, EloEngine]:
    """One chronological pass: attach pregame Elo columns, update on results.

    The single source of truth for the update loop — the board, the
    evaluation, and the ``CFB.fit`` grid search all run through here so a
    tweak to the update rule cannot fork behaviour.
    """
    df = games.sort_values(["gameday", "home_team"]).reset_index(drop=True)
    if clusters is None:
        engine = EloEngine(k=k, hfa=hfa, regression=regression)
    else:
        engine = ClusterRegressEngine(clusters, k=k, hfa=hfa, regression=regression)

    home_elo, away_elo, probs = [], [], []
    for row in df.itertuples(index=False):
        neutral = str(row.location) != "Home"
        h, a, p = engine.pregame(int(row.season), row.home_team, row.away_team, neutral)
        home_elo.append(h)
        away_elo.append(a)
        probs.append(p)
        if pd.notna(row.home_score) and pd.notna(row.away_score):
            margin = float(row.home_score) - float(row.away_score)
            capped = float(np.clip(margin, -margin_cap, margin_cap))
            # Feed the engine capped scores so its MOV multiplier sees the
            # capped margin; the win/loss outcome is unchanged by the cap.
            engine.update(
                row.home_team,
                row.away_team,
                capped if capped > 0 else 0.0,
                0.0 if capped > 0 else -capped,
                neutral=neutral,
                playoff=str(row.game_type) in POSTSEASON_TYPES,
            )

    df["home_elo"] = home_elo
    df["away_elo"] = away_elo
    df["elo_diff"] = (
        df["home_elo"] - df["away_elo"]
        + np.where(df["location"].eq("Home"), hfa, 0.0)
    )
    df["elo_home_prob"] = probs
    df["elo_spread"] = df["elo_diff"] / ELO_PER_POINT
    return df, engine


def compute_elo(games: pd.DataFrame) -> pd.DataFrame:
    """Attach pregame Elo columns; update on played games, capped margins."""
    df, _ = walk_forward(games)
    return df


def ratings_board(
    games: pd.DataFrame, clusters: dict[tuple[int, str], int] | None = None
) -> pd.DataFrame:
    """Final ratings plus each team's season line, sorted."""
    played = games.dropna(subset=["home_score", "away_score"])
    _, engine = walk_forward(played, clusters=clusters)

    recs: dict[str, list[int]] = {}
    for row in played.itertuples(index=False):
        hw = row.home_score > row.away_score
        recs.setdefault(row.home_team, [0, 0])[0 if hw else 1] += 1
        recs.setdefault(row.away_team, [0, 0])[1 if hw else 0] += 1

    board = pd.DataFrame(
        {
            "team": list(engine.ratings),
            "elo": [round(r, 1) for r in engine.ratings.values()],
        }
    )
    board["w"] = board["team"].map(lambda t: recs.get(t, [0, 0])[0])
    board["l"] = board["team"].map(lambda t: recs.get(t, [0, 0])[1])
    board["pts_vs_avg"] = ((board["elo"] - BASE_RATING) / ELO_PER_POINT).round(1)
    # With FCS pooled, remaining tiny-sample teams are new FBS programs
    # passing through. Flag rather than hide.
    board["games"] = board["w"] + board["l"]
    board["small_sample"] = board["games"] < 4
    return board.sort_values("elo", ascending=False).reset_index(drop=True)


def evaluate(
    games: pd.DataFrame,
    eval_from: int | None = EVAL_FROM,
    **constants,
) -> dict[str, float]:
    """Walk-forward log loss and accuracy of pregame probabilities.

    Ratings warm up on the full history, but scoring starts at
    ``eval_from`` (default 2005 — the plan treats 2000-2004 as burn-in).
    """
    df, _ = walk_forward(games, **constants)
    df = df.dropna(subset=["home_score", "away_score"])
    if eval_from is not None:
        df = df[df["season"] >= eval_from]
    y = (df["home_score"] > df["away_score"]).astype(float)
    p = df["elo_home_prob"].clip(1e-6, 1 - 1e-6)
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    acc = float(((p > 0.5) == y.astype(bool)).mean())
    return {"games": len(df), "log_loss": round(ll, 4), "accuracy": round(acc, 4)}


def load_games(pool: bool = True, verbose: bool = False) -> pd.DataFrame:
    games = pd.read_csv(AGG_DIR / "cfb_games.csv")
    return pool_fcs(games, verbose=verbose) if pool else games


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument(
        "--no-pool", action="store_true", help="rate FCS opponents individually"
    )
    ap.add_argument(
        "--flat", action="store_true", help="flat-1500 season regression"
    )
    args = ap.parse_args()

    # Conference-cluster regression is the default whenever the cluster map
    # exists (python3 -m CFB.conferences builds it).
    clusters = None
    if not args.flat and (AGG_DIR / "cfb_conference_clusters.csv").exists():
        from CFB.conferences import load_clusters

        clusters = load_clusters()
        print("season regression: conference-cluster means")

    games = load_games(pool=not args.no_pool, verbose=True)
    board = ratings_board(games, clusters=clusters)
    board.to_csv(AGG_DIR / "cfb_elo_ratings.csv", index=False)

    fbs = board[~board["small_sample"]]
    print(fbs.head(args.top).to_string(index=False))
    fcs_row = board[board["team"] == FCS_TEAM]
    if not fcs_row.empty:
        r = fcs_row.iloc[0]
        print(f"\npooled FCS team: elo={r['elo']}  ({r['w']}-{r['l']} vs FBS)")
    print()
    print("eval 2005+ :", evaluate(games, clusters=clusters))
    print("eval all   :", evaluate(games, eval_from=None, clusters=clusters))


if __name__ == "__main__":
    main()
