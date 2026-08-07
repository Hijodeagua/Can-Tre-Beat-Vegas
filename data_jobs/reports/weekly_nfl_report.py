"""Twice-weekly NFL report: the week's slate, the model's picks, the books' form.

Designed to run Tuesday and Friday nights (see
``.github/workflows/weekly-nfl-report.yml``):

- **Tuesday** — the previous week's games have just finished: grade last
  week's logged picks, grade the books, preview the coming week.
- **Friday** — lines have settled for the weekend: refreshed picks with
  current market numbers.

Sections
    1. This week's slate — every game with model win probability, market win
       probability, and the model-vs-market edge.
    2. Top 3 confidence picks — the product: "Chiefs over Browns" x3.
    3. Pick ledger — every previously logged pick that now has a result,
       graded, plus the season running tally.
    4. Bookmaker leaderboard — sharpest and cheapest closing prices, last
       graded week and season to date.

The picks model is the bake-off winner (`compare_models.py`): calibrated
logistic regression over the 45 v2 features, recency-weighted with a 6-season
half-life. Trained on every played game outside the calibration season
(2002 onward), calibrated on the most recent completed season.

Usage
    python3 -m data_jobs.reports.weekly_nfl_report                  # infer week
    python3 -m data_jobs.reports.weekly_nfl_report --season 2026 --week 1 --write
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from NFL.inventory.book_performance import book_performance
from NFL.model.v2.compare_models import make_model, recency_weights
from NFL.model.v2.dataset import build_dataset, feature_matrix
from NFL.model.v2.train import _apply_platt, _platt

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "weekly"
PICKS_LOG = OUT_DIR / "picks_log.csv"

PICKS_MODEL = "logistic"
HALF_LIFE = 6.0
TOP_K = 3

# The top 10 of NFL/model/v2/artifacts/importance/uniform_ranking.csv. Pinned
# rather than read at runtime so a regenerated ranking can never silently
# change what the report ships. The ablation study found log loss flat across
# 5-15 features (0.6140-0.6146) and clearly worse at all 45 (0.6178), so 10 is
# chosen for being in the middle of a plateau, not for being the argmin.
PICKS_FEATURES = [
    "spread_line", "market_home_prob", "elo_diff", "roll_margin_diff",
    "elo_vs_spread", "elo_spread", "away_roll_pf", "home_roll_margin",
    "elo_home_prob", "away_ppg_diff_std",
]

# nflverse abbrev -> short display name for the "X over Y" line.
NICKNAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills",
    "CAR": "Panthers", "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns",
    "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs",
    "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
    "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks",
    "SF": "49ers", "TB": "Buccaneers", "TEN": "Titans", "WAS": "Commanders",
}


def _nick(team: str) -> str:
    return NICKNAMES.get(team, team)


def _best_prices() -> pd.DataFrame:
    """Best available moneyline per side for upcoming games, or empty."""
    try:
        from NFL.inventory.line_shopping import best_prices_upcoming
        return best_prices_upcoming()
    except Exception:
        return pd.DataFrame()


def _shop_line(prices: pd.DataFrame, row) -> str | None:
    """"+164 at DraftKings" for the picked side, if we have a price for it.

    Shopping is worth ~2.9 points of ROI per bet versus taking the consensus
    (see NFL/inventory/line_shopping.py), so the report names the book rather
    than leaving the reader to take whatever their default app shows.
    """
    if prices.empty:
        return None
    m = prices[(prices["home_team"] == row.home_team)
               & (prices["away_team"] == row.away_team)]
    if m.empty:
        return None
    r = m.iloc[0]
    picked_home = row.pick_team == row.home_team
    odds = r["best_home_odds"] if picked_home else r["best_away_odds"]
    book = r["best_home_book"] if picked_home else r["best_away_book"]
    if pd.isna(odds):
        return None
    return f"{int(odds):+d} at {book}"


def infer_week(df: pd.DataFrame, today: pd.Timestamp) -> tuple[int, int]:
    """The (season, week) of the next unplayed game on or after today.

    Falls back to the last played week (e.g. running right after the season
    ends) so the report always has something to say.
    """
    upcoming = df[df["home_score"].isna() & (df["gameday"] >= today.normalize())]
    if len(upcoming):
        nxt = upcoming.sort_values("gameday").iloc[0]
        return int(nxt["season"]), int(nxt["week"])
    played = df.dropna(subset=["home_score"]).sort_values("gameday")
    last = played.iloc[-1]
    return int(last["season"]), int(last["week"])


def fit_picks_model(df: pd.DataFrame, target_season: int,
                    cutoff: pd.Timestamp | None = None):
    """Fit on all played games outside the calibration season; Platt on it.

    ``cutoff`` (the target week's first kickoff) keeps backdated runs honest:
    live runs never have the scored games in the fit set, but a regenerated
    past week would without it.
    """
    played = df.dropna(subset=["home_win"]).copy()
    if cutoff is not None:
        played = played[played["gameday"] < cutoff]
    cal_season = int(played.loc[played["season"] < target_season, "season"].max())
    cal = played[played["season"] == cal_season]
    fit = played[played["season"] != cal_season]

    model = make_model(PICKS_MODEL)
    w = recency_weights(fit["season"], target_season, HALF_LIFE)
    model.fit(feature_matrix(fit, PICKS_FEATURES), fit["home_win"].astype(int),
              clf__sample_weight=w)
    raw_cal = model.predict_proba(feature_matrix(cal, PICKS_FEATURES))[:, 1]
    platt = _platt(raw_cal, cal["home_win"].astype(int).to_numpy())
    return model, platt, cal_season, len(fit)


def score_week(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    games = df[(df["season"] == season) & (df["week"] == week)].copy()
    if games.empty:
        raise SystemExit(f"no games found for season {season} week {week}")

    model, platt, cal_season, n_fit = fit_picks_model(
        df, season, cutoff=games["gameday"].min())
    raw = model.predict_proba(feature_matrix(games, PICKS_FEATURES))[:, 1]
    games["prob_home"] = _apply_platt(platt, raw)
    games["pick_team"] = np.where(games["prob_home"] >= 0.5,
                                  games["home_team"], games["away_team"])
    games["opp_team"] = np.where(games["prob_home"] >= 0.5,
                                 games["away_team"], games["home_team"])
    games["confidence"] = np.where(games["prob_home"] >= 0.5,
                                   games["prob_home"], 1 - games["prob_home"])
    games["market_conf"] = np.where(games["prob_home"] >= 0.5,
                                    games["market_home_prob"],
                                    1 - games["market_home_prob"])
    games["edge"] = games["confidence"] - games["market_conf"]
    games.attrs.update(cal_season=cal_season, n_fit=n_fit)
    return games.sort_values("confidence", ascending=False)


# --------------------------------------------------------------------------
# pick ledger
# --------------------------------------------------------------------------

LOG_COLS = ["logged_at", "season", "week", "game_id", "gameday",
            "pick_team", "opp_team", "prob", "market_prob", "model"]


def log_picks(scored: pd.DataFrame, season: int, week: int, top_k: int = TOP_K) -> int:
    """Append this run's top-k picks to the ledger (idempotent per game)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(PICKS_LOG) if PICKS_LOG.exists() else pd.DataFrame(columns=LOG_COLS)

    top = scored.head(top_k)
    rows = pd.DataFrame({
        "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "season": season, "week": week,
        "game_id": top["game_id"], "gameday": top["gameday"].dt.date,
        "pick_team": top["pick_team"], "opp_team": top["opp_team"],
        "prob": top["confidence"].round(4),
        "market_prob": top["market_conf"].round(4),
        "model": f"{PICKS_MODEL}_hl{HALF_LIFE:g}",
    })
    # A game already logged keeps its first (earlier) pick — Friday reruns
    # refresh the report but don't rewrite history.
    fresh = rows[~rows["game_id"].isin(existing["game_id"])]
    out = pd.concat([existing, fresh], ignore_index=True)
    out.to_csv(PICKS_LOG, index=False)
    return len(fresh)


def grade_ledger(df: pd.DataFrame) -> pd.DataFrame:
    if not PICKS_LOG.exists():
        return pd.DataFrame()
    log = pd.read_csv(PICKS_LOG)
    results = df.dropna(subset=["home_score"])[
        ["game_id", "home_team", "home_score", "away_score"]]
    g = log.merge(results, on="game_id", how="left")
    winner = np.where(g["home_score"] > g["away_score"], g["home_team"],
                      np.where(g["home_score"] < g["away_score"],
                               g["game_id"].map(
                                   df.set_index("game_id")["away_team"]), "TIE"))
    g["result"] = np.where(g["home_score"].isna(), "pending",
                           np.where(g["pick_team"] == winner, "W", "L"))
    return g


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _md(df: pd.DataFrame) -> str:
    if not len(df):
        return "_(none)_"
    cols = [str(c) for c in df.columns]
    body = [[("" if pd.isna(v) else str(v)) for v in row] for row in df.itertuples(index=False)]
    widths = [max(len(cols[i]), *(len(r[i]) for r in body)) for i in range(len(cols))]
    fmt = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    return "\n".join([fmt(cols), "|" + "|".join("-" * (w + 2) for w in widths) + "|",
                      *(fmt(r) for r in body)])


def render(scored: pd.DataFrame, df: pd.DataFrame, season: int, week: int) -> str:
    now = datetime.now(timezone.utc)
    top = scored.head(TOP_K)

    slate = pd.DataFrame({
        "game": scored["away_team"].map(_nick) + " @ " + scored["home_team"].map(_nick),
        "date": scored["gameday"].dt.date,
        "spread": scored["spread_line"],
        "total": scored["total_line"],
        "model home %": (scored["prob_home"] * 100).round(1),
        "market home %": (scored["market_home_prob"] * 100).round(1),
        "model pick": scored["pick_team"].map(_nick),
        "edge": (scored["edge"] * 100).round(1),
    }).sort_values("date")

    prices = _best_prices()
    picks_lines = []
    for i, r in enumerate(top.itertuples(index=False), 1):
        line = (f"{i}. **{_nick(r.pick_team)} over {_nick(r.opp_team)}** — "
                f"model {r.confidence:.0%}, market {r.market_conf:.0%}")
        shop = _shop_line(prices, r)
        picks_lines.append(line + (f" — best price **{shop}**" if shop else ""))

    lines = [
        f"# NFL weekly report — {season} week {week}",
        "",
        f"_Generated {now:%Y-%m-%d %H:%M UTC} ({now:%A})_",
        f"_Model: calibrated {PICKS_MODEL} regression on {len(PICKS_FEATURES)} features, "
        f"recency half-life {HALF_LIFE:g} seasons, trained on "
        f"{scored.attrs.get('n_fit', '?')} games (2002-present), calibrated on "
        f"{scored.attrs.get('cal_season', '?')}._",
        "",
        f"## Top {TOP_K} confidence picks",
        "",
        *picks_lines,
        "",
        "The market column is the no-vig closing consensus — when it is higher than "
        "the model's number, the books are *more* sure than we are and the pick "
        "carries no betting edge. These are confidence picks, not value picks.",
        "",
        "## The slate",
        "",
        _md(slate),
        "",
        "_Spread is the home line (positive = home favoured). Edge = model "
        "confidence minus market confidence on the picked side, in points._",
        "",
    ]

    ledger = grade_ledger(df)
    if len(ledger):
        graded = ledger[ledger["result"].isin(["W", "L"])]
        lines += ["## Pick ledger", ""]
        show = ledger.tail(12)[["season", "week", "gameday", "pick_team",
                                "opp_team", "prob", "market_prob", "result"]].copy()
        show["pick_team"] = show["pick_team"].map(_nick)
        show["opp_team"] = show["opp_team"].map(_nick)
        lines += [_md(show), ""]
        if len(graded):
            w = int((graded["result"] == "W").sum())
            l = int((graded["result"] == "L").sum())
            lines += [f"Season tally on graded picks: **{w}-{l}** "
                      f"({w / max(w + l, 1):.0%}).", ""]

    # Grade the books against the most recent season that actually has results
    # (in-season that's the current one; in the offseason, last season).
    graded_season = season
    if _last_graded_week(df, graded_season) is None:
        prior = df.dropna(subset=["home_score"])["season"]
        graded_season = int(prior.max()) if len(prior) else None

    if graded_season is not None:
        last_wk = _last_graded_week(df, graded_season)
        for scope, kwargs in [
            (f"{graded_season} week {last_wk}",
             {"season": graded_season, "week": last_wk}),
            (f"season {graded_season} to date", {"season": graded_season}),
        ]:
            lb = book_performance(**kwargs)
            if len(lb):
                lines += [f"## Bookmaker leaderboard — {scope}", "",
                          "Lower Brier = the book's closing moneyline was closer to what "
                          "actually happened. Lower hold = the same bet costs less.", "",
                          _md(lb), ""]

    lines += [
        "---",
        "_Every number here is out of sample: the model never trains on the games "
        "it grades. Historical honesty check: over 2015-2025 the market's own top-3 "
        "hit rate (79.6%) beat every model tested (see "
        "`NFL/model/v2/artifacts/bakeoff_win.csv`)._",
        "",
    ]
    return "\n".join(lines)


def _last_graded_week(df: pd.DataFrame, season: int) -> int | None:
    played = df[(df["season"] == season)].dropna(subset=["home_score"])
    return int(played["week"].max()) if len(played) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--no-log", action="store_true",
                    help="don't append picks to the ledger (dry run)")
    args = ap.parse_args()

    df = build_dataset()
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    season, week = (args.season, args.week) if args.season and args.week \
        else infer_week(df, today)

    scored = score_week(df, season, week)
    if not args.no_log:
        n = log_picks(scored, season, week)
        print(f"[picks log] {n} new picks appended")
    report = render(scored, df, season, week)
    print(report)

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{season}_week{week:02d}.md"
        path.write_text(report)
        print(f"\nwrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
