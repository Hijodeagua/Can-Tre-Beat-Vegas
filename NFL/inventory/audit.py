"""End-of-season inventory: what we actually captured, and how Vegas did.

Answers four questions about a completed season before any modelling starts:

1. **Results** — is the season complete in the schedule file?
2. **Odds capture** — how many of those games did the twice-daily snapshot
   job actually see, at what depth, and under which schema generation?
3. **Vegas' report card** — how good was the closing line? Favourites SU,
   ATS splits, over/under, spread error, moneyline calibration. This is the
   number any model has to beat.
4. **Our assets** — which training files are stale, which tracking files are
   empty, and how the model scored out of sample.

Usage
    python3 -m NFL.inventory.audit --season 2025
    python3 -m NFL.inventory.audit --season 2025 --write
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from NFL.model.line_movement import (
    aggregate_movement,
    load_all_snapshots,
    movement_by_nflverse_game,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GAMES_PATH = REPO_ROOT / "data" / "schedules" / "nflverse_games.csv"
SNAPSHOT_GLOB = str(REPO_ROOT / "data" / "odds_api_data_*.csv")
OUT_DIR = REPO_ROOT / "NFL" / "inventory"
V2_ARTIFACTS = REPO_ROOT / "NFL" / "model" / "v2" / "artifacts"


# --------------------------------------------------------------------------
# 1. results
# --------------------------------------------------------------------------

def season_results(season: int) -> pd.DataFrame:
    g = pd.read_csv(GAMES_PATH)
    g = g[g["season"] == season].copy()
    g["gameday"] = pd.to_datetime(g["gameday"])
    g["margin"] = g["home_score"] - g["away_score"]
    g["total_points"] = g["home_score"] + g["away_score"]
    return g.sort_values(["gameday", "game_type"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. odds capture
# --------------------------------------------------------------------------

SCHEMA_MARKERS = {
    "Game ID": "game_id",
    "Avg Home Spread Points": "spread_points",
    "Avg Total Points": "total_points",
    "League": "league_col",
}


def snapshot_file_inventory(pattern: str = SNAPSHOT_GLOB) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        try:
            header = pd.read_csv(path, nrows=0).columns
        except Exception:
            rows.append({"file": name, "readable": False})
            continue
        cols = set(header)
        rows.append({
            "file": name,
            "readable": True,
            "n_cols": len(header),
            "bytes": os.path.getsize(path),
            **{label: (marker in cols) for marker, label in SCHEMA_MARKERS.items()},
        })
    df = pd.DataFrame(rows)
    stamp = df["file"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0]
    df["pulled_on"] = pd.to_datetime(stamp, errors="coerce")
    return df


def schema_generations(files: pd.DataFrame) -> pd.DataFrame:
    ok = files[files["readable"] & files["pulled_on"].notna()]
    grp = ok.groupby(["game_id", "spread_points", "league_col"], dropna=False)
    out = grp.agg(
        files=("file", "size"),
        first_pull=("pulled_on", "min"),
        last_pull=("pulled_on", "max"),
        col_counts=("n_cols", lambda s: ", ".join(str(c) for c in sorted(set(s)))),
    ).reset_index()
    out["generation"] = np.where(out["game_id"], "current", "legacy")
    return out.sort_values("first_pull")


def odds_coverage(results: pd.DataFrame, movement: pd.DataFrame) -> pd.DataFrame:
    """One row per scheduled game: did the snapshot job capture it, how deep?"""
    joined = movement_by_nflverse_game(results, movement)
    cov = results[["game_id", "season", "week", "game_type", "gameday",
                   "home_team", "away_team"]].merge(
        joined[["game_id", "num_snapshots", "first_seen", "last_seen",
                "open_novig_home", "close_novig_home", "novig_move_home",
                "open_spread_home", "close_spread_home", "spread_move",
                "open_total", "close_total", "total_move"]],
        on="game_id", how="left",
    )
    cov["captured"] = cov["num_snapshots"].notna()
    cov["has_spread_points"] = cov["open_spread_home"].notna()
    return cov


# --------------------------------------------------------------------------
# 3. Vegas report card
# --------------------------------------------------------------------------

def _record(mask_win: pd.Series, mask_push: pd.Series | None = None) -> str:
    w = int(mask_win.sum())
    p = int(mask_push.sum()) if mask_push is not None else 0
    l = int(len(mask_win) - w - p)
    pct = w / (w + l) if (w + l) else float("nan")
    tail = f"-{p}" if p else ""
    return f"{w}-{l}{tail} ({pct:.1%})"


def vegas_report_card(results: pd.DataFrame) -> pd.DataFrame:
    g = results.dropna(subset=["home_score", "away_score"]).copy()
    rows: list[dict] = []

    def add(metric: str, value, note: str = "") -> None:
        rows.append({"metric": metric, "value": value, "note": note})

    add("games graded", len(g))

    # Straight up
    home_win = g["margin"] > 0
    add("home teams SU", _record(home_win, g["margin"] == 0))

    spread = g["spread_line"]
    fav_home = spread > 0
    pickem = spread == 0
    fav = g[~pickem]
    fav_won = np.where(fav["spread_line"] > 0, fav["margin"] > 0, fav["margin"] < 0)
    add("closing favourites SU", _record(pd.Series(fav_won), pd.Series(fav["margin"].values == 0)))

    # Against the spread
    ats = g["margin"] - spread
    add("home teams ATS", _record(ats > 0, ats == 0))
    add("closing favourites ATS", _record(
        pd.Series(np.where(fav_home[~pickem], ats[~pickem] > 0, ats[~pickem] < 0)),
        pd.Series((ats[~pickem] == 0).values)))
    big = g[spread.abs() >= 7]
    big_ats = big["margin"] - big["spread_line"]
    add("dogs of 7+ ATS", _record(
        pd.Series(np.where(big["spread_line"] > 0, big_ats < 0, big_ats > 0)),
        pd.Series((big_ats == 0).values)), f"n={len(big)}")

    # Totals
    tot = g["total_points"] - g["total_line"]
    add("overs", _record(tot > 0, tot == 0))

    # Accuracy of the number itself
    add("spread MAE (pts)", round(float((g["margin"] - spread).abs().mean()), 2),
        "average miss of the closing spread vs actual margin")
    add("spread bias (pts)", round(float((g["margin"] - spread).mean()), 2),
        "positive = home teams beat the number on average")
    add("total MAE (pts)", round(float(tot.abs().mean()), 2))
    add("total bias (pts)", round(float(tot.mean()), 2))
    add("median |spread|", float(spread.abs().median()))

    # Moneyline calibration
    ml = g.dropna(subset=["home_moneyline", "away_moneyline"]).copy()
    if len(ml):
        hp = np.where(ml["home_moneyline"] < 0,
                      -ml["home_moneyline"] / (-ml["home_moneyline"] + 100),
                      100 / (ml["home_moneyline"] + 100))
        ap = np.where(ml["away_moneyline"] < 0,
                      -ml["away_moneyline"] / (-ml["away_moneyline"] + 100),
                      100 / (ml["away_moneyline"] + 100))
        book = hp + ap
        novig = hp / book
        actual = (ml["margin"] > 0).astype(float)
        add("games with moneyline", len(ml))
        add("average hold (vig)", f"{np.mean(book - 1):.2%}",
            "book's built-in margin on the two-way moneyline")
        add("closing ML Brier", round(float(np.mean((novig - actual) ** 2)), 4))
        add("closing ML predicted home win rate", f"{novig.mean():.1%}")
        add("closing ML actual home win rate", f"{actual.mean():.1%}")

    return pd.DataFrame(rows)


def calibration_table(results: pd.DataFrame, bins: int = 6) -> pd.DataFrame:
    """Is the closing moneyline honest at every price?"""
    g = results.dropna(subset=["home_score", "home_moneyline", "away_moneyline"]).copy()
    hp = np.where(g["home_moneyline"] < 0, -g["home_moneyline"] / (-g["home_moneyline"] + 100),
                  100 / (g["home_moneyline"] + 100))
    ap = np.where(g["away_moneyline"] < 0, -g["away_moneyline"] / (-g["away_moneyline"] + 100),
                  100 / (g["away_moneyline"] + 100))
    g["novig_home"] = hp / (hp + ap)
    g["home_won"] = (g["margin"] > 0).astype(int)
    g["bucket"] = pd.cut(g["novig_home"], np.linspace(0, 1, bins + 1))
    out = g.groupby("bucket", observed=True).agg(
        n=("home_won", "size"),
        predicted=("novig_home", "mean"),
        actual=("home_won", "mean"),
    ).reset_index()
    out["predicted"] = out["predicted"].round(3)
    out["actual"] = out["actual"].round(3)
    out["gap"] = (out["actual"] - out["predicted"]).round(3)
    return out


def line_movement_study(coverage: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Does the direction the money moved predict the result?"""
    m = coverage.dropna(subset=["novig_move_home"]).merge(
        results[["game_id", "margin", "spread_line"]], on="game_id", how="inner")
    m = m.dropna(subset=["margin"])
    if m.empty:
        return pd.DataFrame()

    m["home_won"] = (m["margin"] > 0).astype(int)
    m["home_covered"] = np.where(m["margin"] - m["spread_line"] > 0, 1,
                                 np.where(m["margin"] - m["spread_line"] < 0, 0, np.nan))
    m["direction"] = np.where(m["novig_move_home"] > 0.01, "toward home",
                              np.where(m["novig_move_home"] < -0.01, "toward away", "flat"))

    out = m.groupby("direction").agg(
        games=("home_won", "size"),
        avg_move=("novig_move_home", "mean"),
        closing_implied=("close_novig_home", "mean"),
        home_win_rate=("home_won", "mean"),
        home_cover_rate=("home_covered", "mean"),
    ).reset_index()
    # The bar isn't 50% — it's what the *closing* price already said. Movement
    # only carries information if the actual rate beats that.
    out["beat_closing_by"] = out["home_win_rate"] - out["closing_implied"]
    for c in ["avg_move", "closing_implied", "home_win_rate", "home_cover_rate",
              "beat_closing_by"]:
        out[c] = out[c].round(3)
    return out


# --------------------------------------------------------------------------
# 4. our assets
# --------------------------------------------------------------------------

def asset_audit(season: int) -> pd.DataFrame:
    rows = []

    def add(asset: str, status: str, detail: str) -> None:
        rows.append({"asset": asset, "status": status, "detail": detail})

    legacy_train = REPO_ROOT / "data" / "2023-2025W3.csv"
    if legacy_train.exists():
        d = pd.read_csv(legacy_train, usecols=["Date"])
        last = pd.to_datetime(d["Date"]).max().date()
        add("data/2023-2025W3.csv", "STALE",
            f"per-team box scores end {last} — never saw the {season} season "
            "(v2 no longer depends on it)")

    adv = REPO_ROOT / "data" / "advanced_stats_25-26.csv"
    if adv.exists():
        head = pd.read_csv(adv, nrows=3)
        looks_nba = head.astype(str).apply(
            lambda s: s.str.contains("Thunder|Celtics|Lakers", na=False)).any().any()
        add("data/advanced_stats_25-26.csv", "MISFILED" if looks_nba else "OK",
            "contains NBA team ratings, not NFL — not used by any NFL model"
            if looks_nba else "NFL advanced stats")

    games = pd.read_csv(GAMES_PATH, usecols=["season", "home_score"])
    done = games[(games["season"] == season) & games["home_score"].notna()]
    add("data/schedules/nflverse_games.csv", "CURRENT",
        f"{len(done)} completed {season} games; through {games['season'].max()} schedule")

    empties = []
    for p in sorted(REPO_ROOT.glob("NFL/Week*/*.md")) + [REPO_ROOT / "NFL" / "Weeks.md"]:
        if p.exists() and p.stat().st_size <= 2:
            empties.append(p.relative_to(REPO_ROOT).as_posix())
    if empties:
        add("NFL weekly pick logs", "EMPTY",
            f"{len(empties)} placeholder files with no content: {', '.join(empties)}")

    old_art = REPO_ROOT / "NFL" / "model" / "artifacts" / "metrics_win.csv"
    if old_art.exists():
        m = pd.read_csv(old_art)
        n_test = int(m.loc[m["model"] == "lgbm_test", "n"].iloc[0]) if (m["model"] == "lgbm_test").any() else 0
        add("NFL/model/artifacts (v1)", "SUPERSEDED",
            f"single split, {n_test}-game test set, trained on the stale box-score file")

    for target in ("win", "ats", "total"):
        p = V2_ARTIFACTS / f"metrics_{target}.csv"
        if p.exists():
            m = pd.read_csv(p)
            n = int(m["n"].iloc[0])
            add(f"NFL/model/v2 ({target})", "CURRENT",
                f"walk-forward, {n} out-of-sample games")

    return pd.DataFrame(rows)


def v2_season_scorecard(season: int) -> pd.DataFrame:
    """How the v2 model did on this season, out of sample."""
    rows = []
    for target, label in [("win", "straight up"), ("ats", "against the spread"),
                          ("total", "over/under")]:
        p = V2_ARTIFACTS / f"oos_predictions_{target}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d[d["season"] == season]
        if d.empty:
            continue
        pick_home = d["prob"] >= 0.5
        correct = np.where(pick_home, d["y"] == 1, d["y"] == 0)
        rows.append({
            "target": f"{target} ({label})",
            "games": len(d),
            "accuracy": round(float(correct.mean()), 4),
            "brier": round(float(np.mean((d["prob"] - d["y"]) ** 2)), 4),
            "break_even_at_-110": 0.5238,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _md(df: pd.DataFrame) -> str:
    """Markdown table without pulling in `tabulate` as a dependency."""
    if not len(df):
        return "_(none)_"
    cols = [str(c) for c in df.columns]
    body = [[("" if pd.isna(v) else str(v)) for v in row] for row in df.itertuples(index=False)]
    widths = [max(len(cols[i]), *(len(r[i]) for r in body)) for i in range(len(cols))]
    fmt = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    return "\n".join([fmt(cols), "|" + "|".join("-" * (w + 2) for w in widths) + "|",
                      *(fmt(r) for r in body)])


def build_report(season: int) -> tuple[str, dict[str, pd.DataFrame]]:
    results = season_results(season)
    files = snapshot_file_inventory()
    gens = schema_generations(files)
    snaps = load_all_snapshots()
    movement = aggregate_movement(snaps)
    coverage = odds_coverage(results, movement)
    card = vegas_report_card(results)
    calib = calibration_table(results)
    moves = line_movement_study(coverage, results)
    assets = asset_audit(season)
    scorecard = v2_season_scorecard(season)

    played = results.dropna(subset=["home_score"])
    captured = coverage[coverage["captured"]]
    by_week = coverage.groupby("week").agg(
        games=("game_id", "size"),
        captured=("captured", "sum"),
        median_snapshots=("num_snapshots", "median"),
    ).reset_index()
    by_week["captured_pct"] = (by_week["captured"] / by_week["games"]).round(2)
    missed_weeks = ", ".join(str(w) for w in by_week.loc[by_week["captured"] == 0, "week"])

    gen_tbl = gens[["generation", "files", "first_pull", "last_pull",
                    "col_counts", "game_id", "spread_points"]].copy()
    gen_tbl["first_pull"] = gen_tbl["first_pull"].dt.date
    gen_tbl["last_pull"] = gen_tbl["last_pull"].dt.date

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# NFL {season} season inventory",
        "",
        f"_Generated {generated} by `python3 -m NFL.inventory.audit --season {season} --write`._",
        "",
        "## 1. Results",
        "",
        f"- **{len(played)} of {len(results)}** scheduled {season} games have final scores.",
        f"- Regular season: {int((results['game_type'] == 'REG').sum())} games; "
        f"postseason: {int((results['game_type'] != 'REG').sum())}.",
        f"- Season ran {played['gameday'].min().date()} to {played['gameday'].max().date()}.",
        "",
        "## 2. Odds capture",
        "",
        f"- **{len(files)}** snapshot files on disk, "
        f"{int(files['readable'].sum())} readable, {len(snaps):,} total rows.",
        f"- Snapshots span {snaps['ts'].min():%Y-%m-%d} to {snaps['ts'].max():%Y-%m-%d}, "
        f"covering **{movement['game_key'].nunique()}** distinct games across all seasons.",
        f"- Of the {len(results)} {season} games, **{int(coverage['captured'].sum())} "
        f"({coverage['captured'].mean():.0%})** were captured, at a median of "
        f"**{captured['num_snapshots'].median():.0f}** snapshots each.",
        f"- Spread *points* are present for only **{int(coverage['has_spread_points'].sum())}** "
        f"{season} game(s): the legacy schema stored the juice on each side but never the "
        "number itself, so this season's line movement is **moneyline-only**.",
        f"- Weeks with nothing captured: **{missed_weeks or 'none'}**. Weeks 1-4 predate the "
        "first snapshot; the later gaps are missed runs of the fetch job.",
        f"- {int((~coverage['captured']).sum())} games have no odds record at all.",
        "",
        "### Schema generations",
        "",
        _md(gen_tbl),
        "",
        "### Capture by week",
        "",
        _md(by_week),
        "",
        "## 3. Vegas' report card",
        "",
        "This is the bar. Every number below is the closing line's own performance.",
        "",
        _md(card),
        "",
        "### Closing moneyline calibration",
        "",
        _md(calib),
        "",
        "### Did the line movement mean anything?",
        "",
        f"Opening vs closing no-vig moneyline for the {int(coverage['captured'].sum())} "
        "captured games:",
        "",
        _md(moves),
        "",
        "## 4. Our assets",
        "",
        _md(assets),
        "",
        f"### v2 model on {season}, out of sample",
        "",
        _md(scorecard),
        "",
    ]
    tables = {
        "coverage": coverage,
        "vegas_report_card": card,
        "calibration": calib,
        "line_movement": moves,
        "assets": assets,
        "scorecard": scorecard,
        "schema_generations": gen_tbl,
        "capture_by_week": by_week,
    }
    return "\n".join(lines), tables


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--write", action="store_true", help="write markdown + CSVs")
    args = ap.parse_args()

    report, tables = build_report(args.season)
    print(report)

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"INVENTORY_{args.season}.md").write_text(report)
        tables["coverage"].to_csv(OUT_DIR / f"coverage_{args.season}.csv", index=False)
        tables["vegas_report_card"].to_csv(
            OUT_DIR / f"vegas_report_card_{args.season}.csv", index=False)
        print(f"\nwrote INVENTORY_{args.season}.md + CSVs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
