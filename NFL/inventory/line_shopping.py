"""What is line shopping worth, and which book should you take each side at?

The snapshots already carry 8-10 books per game, but every model result in this
repo is measured against the *consensus* price. That understates what a bettor
actually faces, because nobody has to bet the consensus — you bet the best
number available, and the gap between them is free money that requires no
predictive skill at all.

Method: per bet, ROI = p_true * decimal_odds - 1. Take ``p_true`` to be the
no-vig consensus probability (the best available estimate of truth), so the
only thing that changes between shopping and not shopping is the price. Note
that consensus is computed by averaging implied *probabilities* and converting
back — averaging decimal odds is invalid, since longshots dominate the mean.

Usage
    python3 -m NFL.inventory.line_shopping --season 2025
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from NFL.inventory.book_performance import join_results, load_book_closes

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "NFL" / "inventory"


def american_to_decimal(o) -> np.ndarray:
    o = np.asarray(o, dtype=float)
    return np.where(o < 0, 1 + 100 / (-o), 1 + o / 100)


def per_game_prices(season: int | None = None) -> pd.DataFrame:
    g = join_results(load_book_closes())
    if season is not None:
        g = g[g["season"] == season]
    if g.empty:
        return pd.DataFrame()
    g = g.assign(dec_home=american_to_decimal(g["h2h_home"]),
                 dec_away=american_to_decimal(g["h2h_away"]))

    best = g.groupby("game_id").agg(
        best_home=("dec_home", "max"), best_away=("dec_away", "max"),
        n_books=("book", "size"), home_won=("home_won", "first")).reset_index()
    cons = g.groupby("game_id").agg(
        ph=("dec_home", lambda s: (1 / s).mean()),
        pa=("dec_away", lambda s: (1 / s).mean())).reset_index()

    per = best.merge(cons, on="game_id")
    per["cons_home"], per["cons_away"] = 1 / per["ph"], 1 / per["pa"]
    per["hold_consensus"] = per["ph"] + per["pa"] - 1
    per["hold_best"] = 1 / per["best_home"] + 1 / per["best_away"] - 1
    per["true_home"] = per["ph"] / (per["ph"] + per["pa"])
    per["true_away"] = 1 - per["true_home"]
    return per


def shopping_value(per: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side, dc, db, p in (("home", "cons_home", "best_home", "true_home"),
                            ("away", "cons_away", "best_away", "true_away")):
        roi_c = float((per[p] * per[dc] - 1).mean())
        roi_b = float((per[p] * per[db] - 1).mean())
        rows.append({"side": side, "roi_consensus": round(roi_c, 4),
                     "roi_best_price": round(roi_b, 4), "gain": round(roi_b - roi_c, 4)})
    both_c = pd.concat([per["true_home"] * per["cons_home"] - 1,
                        per["true_away"] * per["cons_away"] - 1]).mean()
    both_b = pd.concat([per["true_home"] * per["best_home"] - 1,
                        per["true_away"] * per["best_away"] - 1]).mean()
    rows.append({"side": "both", "roi_consensus": round(float(both_c), 4),
                 "roi_best_price": round(float(both_b), 4),
                 "gain": round(float(both_b - both_c), 4)})
    return pd.DataFrame(rows)


def best_prices_upcoming() -> pd.DataFrame:
    """Best available moneyline per side for games that have not kicked off.

    Unlike the graded helpers above this needs no results, so it can price the
    coming week. One row per game with the best price and which book posts it.
    """
    from NFL.model.line_movement import NAME_TO_NFLVERSE

    closes = load_book_closes()
    if closes.empty:
        return pd.DataFrame()
    now = pd.Timestamp.now()
    upcoming = closes[closes["game_date"] >= now].copy()
    if upcoming.empty:
        return pd.DataFrame()

    upcoming["dec_home"] = american_to_decimal(upcoming["h2h_home"])
    upcoming["dec_away"] = american_to_decimal(upcoming["h2h_away"])
    upcoming["home_nfl"] = upcoming["home_team"].map(NAME_TO_NFLVERSE)
    upcoming["away_nfl"] = upcoming["away_team"].map(NAME_TO_NFLVERSE)

    rows = []
    for key, g in upcoming.groupby("game_key"):
        h = g.loc[g["dec_home"].idxmax()]
        a = g.loc[g["dec_away"].idxmax()]
        rows.append({
            "game_key": key,
            "gameday": g["game_date"].iloc[0].normalize(),
            "home_team": g["home_nfl"].iloc[0], "away_team": g["away_nfl"].iloc[0],
            "books": int(len(g)),
            "best_home_odds": float(h["h2h_home"]), "best_home_book": h["book"],
            "best_away_odds": float(a["h2h_away"]), "best_away_book": a["book"],
            "consensus_home_prob": float((1 / g["dec_home"]).mean()),
            "consensus_away_prob": float((1 / g["dec_away"]).mean()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["hold_consensus"] = out["consensus_home_prob"] + out["consensus_away_prob"] - 1
    out["hold_best"] = 1 / american_to_decimal(out["best_home_odds"]) + \
                       1 / american_to_decimal(out["best_away_odds"]) - 1
    return out


def best_price_by_book(season: int | None = None) -> pd.DataFrame:
    """How often each book posts the best number, split by side.

    The split matters: a book can be systematically generous on home favourites
    and stingy on away dogs, which tells you where to send each bet.
    """
    g = join_results(load_book_closes())
    if season is not None:
        g = g[g["season"] == season]
    g = g.assign(dec_home=american_to_decimal(g["h2h_home"]),
                 dec_away=american_to_decimal(g["h2h_away"]))
    g["is_best_home"] = g.groupby("game_id")["dec_home"].transform(lambda s: s == s.max())
    g["is_best_away"] = g.groupby("game_id")["dec_away"].transform(lambda s: s == s.max())
    out = g.groupby("book").agg(
        games=("game_id", "nunique"),
        best_home_pct=("is_best_home", "mean"),
        best_away_pct=("is_best_away", "mean")).reset_index()
    out["best_either"] = (out["best_home_pct"] + out["best_away_pct"]) / 2
    for c in ("best_home_pct", "best_away_pct", "best_either"):
        out[c] = out[c].round(3)
    return out.sort_values("best_either", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    per = per_game_prices(args.season)
    if per.empty:
        raise SystemExit(f"no per-book prices captured for {args.season}")

    print(f"Season {args.season}: {len(per)} games with per-book closing prices, "
          f"median {per['n_books'].median():.0f} books each\n")
    print(f"hold at consensus prices : {per['hold_consensus'].mean():.2%}")
    print(f"hold taking the best     : {per['hold_best'].mean():.2%}")
    print(f"break-even win rate      : {0.5 * (1 + per['hold_consensus'].mean()):.2%}"
          f"  ->  {0.5 * (1 + per['hold_best'].mean()):.2%}\n")

    val = shopping_value(per)
    print("Expected ROI for a bettor with NO predictive edge:")
    print(val.assign(**{c: val[c].map("{:+.2%}".format)
                        for c in ("roi_consensus", "roi_best_price", "gain")})
             .to_string(index=False))

    books = best_price_by_book(args.season)
    print("\nHow often each book posts the best price:")
    print(books.to_string(index=False))

    if args.write:
        val.to_csv(OUT_DIR / f"line_shopping_{args.season}.csv", index=False)
        books.to_csv(OUT_DIR / f"best_price_by_book_{args.season}.csv", index=False)
        print(f"\nwrote -> {OUT_DIR}")


if __name__ == "__main__":
    main()
