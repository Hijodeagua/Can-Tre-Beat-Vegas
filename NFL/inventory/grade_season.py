"""Grade a season of model picks, week by week, and write the markdown ledger.

Reads the walk-forward out-of-sample predictions written by
``NFL.model.v2.train --save`` — for season S those came from a model that only
ever saw seasons < S, so every pick below is a genuine prediction rather than a
fit. Writes ``NFL/Week_<n>/Week_<n>.md`` per week and a season tally in
``NFL/Weeks.md``.

Usage
    python3 -m NFL.inventory.grade_season --season 2025 --write
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "NFL" / "model" / "v2" / "artifacts"
NFL_DIR = REPO_ROOT / "NFL"

# Only count a spread/total pick when the model is at least this far off a coin
# flip. Below that the pick is noise and betting it just pays vig.
EDGE_THRESHOLD = 0.03
BREAK_EVEN = 110 / 210


def load_predictions(season: int) -> dict[str, pd.DataFrame]:
    out = {}
    for target in ("win", "ats", "total"):
        path = ARTIFACTS / f"oos_predictions_{target}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        d = d[d["season"] == season].copy()
        if len(d):
            out[target] = d
    return out


def _pick_frame(d: pd.DataFrame, target: str) -> pd.DataFrame:
    """Normalise a prediction frame into pick / confidence / hit."""
    d = d.copy()
    d["pick_home"] = d["prob"] >= 0.5
    d["confidence"] = np.where(d["pick_home"], d["prob"], 1 - d["prob"])
    d["hit"] = np.where(d["pick_home"], d["y"] == 1, d["y"] == 0)

    if target == "win":
        d["pick"] = np.where(d["pick_home"], d["home_team"], d["away_team"])
        d["vegas_pick"] = np.where(d["market_home_prob"] >= 0.5, d["home_team"], d["away_team"])
        d["vegas_hit"] = np.where(d["market_home_prob"] >= 0.5, d["y"] == 1, d["y"] == 0)
    elif target == "ats":
        d["pick"] = np.where(
            d["pick_home"],
            d["home_team"] + " " + np.where(d["spread_line"] > 0, "-", "+")
            + d["spread_line"].abs().astype(str),
            d["away_team"] + " " + np.where(d["spread_line"] > 0, "+", "-")
            + d["spread_line"].abs().astype(str),
        )
    else:
        d["pick"] = np.where(d["pick_home"], "Over " + d["total_line"].astype(str),
                             "Under " + d["total_line"].astype(str))

    d["edge"] = (d["prob"] - 0.5).abs()
    d["qualified"] = d["edge"] >= EDGE_THRESHOLD
    d["score"] = d["home_team"] + " " + d["home_score"].astype("Int64").astype(str) + \
        " - " + d["away_score"].astype("Int64").astype(str) + " " + d["away_team"]
    return d


def _rec(hits: pd.Series) -> str:
    w, l = int(hits.sum()), int((~hits.astype(bool)).sum())
    pct = w / (w + l) if (w + l) else float("nan")
    return f"{w}-{l} ({pct:.1%})"


def _md(df: pd.DataFrame) -> str:
    if not len(df):
        return "_(no games)_"
    cols = [str(c) for c in df.columns]
    body = [[("" if pd.isna(v) else str(v)) for v in row] for row in df.itertuples(index=False)]
    widths = [max(len(cols[i]), *(len(r[i]) for r in body)) for i in range(len(cols))]
    row = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    return "\n".join([row(cols), "|" + "|".join("-" * (w + 2) for w in widths) + "|",
                      *(row(r) for r in body)])


def week_report(season: int, week: int, picks: dict[str, pd.DataFrame]) -> str:
    win = picks.get("win", pd.DataFrame())
    ats = picks.get("ats", pd.DataFrame())
    tot = picks.get("total", pd.DataFrame())
    w_win = win[win["week"] == week] if len(win) else win
    w_ats = ats[ats["week"] == week] if len(ats) else ats
    w_tot = tot[tot["week"] == week] if len(tot) else tot

    lines = [f"# {season} Week {week}", ""]
    if len(w_win):
        lines += [
            "## Straight up — model vs. Vegas",
            "",
            _md(pd.DataFrame({
                "matchup": w_win["away_team"] + " @ " + w_win["home_team"],
                "final": w_win["score"],
                "model pick": w_win["pick"],
                "conf": w_win["confidence"].round(3),
                "vegas pick": w_win["vegas_pick"],
                "model": np.where(w_win["hit"], "W", "L"),
                "vegas": np.where(w_win["vegas_hit"], "W", "L"),
            })),
            "",
            f"- Model: **{_rec(w_win['hit'])}**  |  Vegas: **{_rec(w_win['vegas_hit'])}**",
            "",
        ]
    if len(w_ats):
        q = w_ats[w_ats["qualified"]]
        lines += [
            "## Against the spread",
            "",
            _md(pd.DataFrame({
                "matchup": w_ats["away_team"] + " @ " + w_ats["home_team"],
                "pick": w_ats["pick"],
                "conf": w_ats["prob"].round(3),
                "cover margin": w_ats["ats_margin_home"].round(1),
                "result": np.where(w_ats["hit"], "W", "L"),
                "qualified": np.where(w_ats["qualified"], "yes", ""),
            })),
            "",
            f"- All games: **{_rec(w_ats['hit'])}**  |  "
            f"Qualified picks (edge ≥ {EDGE_THRESHOLD:.0%}): "
            f"**{_rec(q['hit']) if len(q) else 'no plays'}**",
            "",
        ]
    if len(w_tot):
        q = w_tot[w_tot["qualified"]]
        lines += [
            "## Totals",
            "",
            f"- All games: **{_rec(w_tot['hit'])}**  |  "
            f"Qualified picks: **{_rec(q['hit']) if len(q) else 'no plays'}**",
            "",
        ]
    return "\n".join(lines)


def season_tally(season: int, picks: dict[str, pd.DataFrame]) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# NFL {season} — model vs. Vegas, graded",
        "",
        f"_Generated {generated} by `python3 -m NFL.inventory.grade_season "
        f"--season {season} --write`._",
        "",
        "Every pick here is **out of sample**: the model that made week N's picks was "
        f"trained only on seasons before {season} and calibrated on {season - 1}. "
        "Spread and total picks are graded two ways — all games, and only the ones where "
        f"the model was at least {EDGE_THRESHOLD:.0%} off a coin flip.",
        "",
        f"Break-even at standard -110 juice is **{BREAK_EVEN:.1%}**.",
        "",
    ]

    win = picks.get("win")
    if win is not None:
        rows = []
        for week, g in win.groupby("week"):
            rows.append({
                "week": int(week), "games": len(g),
                "model SU": _rec(g["hit"]), "vegas SU": _rec(g["vegas_hit"]),
            })
        tbl = pd.DataFrame(rows)
        tbl.loc[len(tbl)] = {
            "week": "TOTAL", "games": len(win),
            "model SU": _rec(win["hit"]), "vegas SU": _rec(win["vegas_hit"]),
        }
        lines += ["## Straight up, week by week", "", _md(tbl), ""]

        agree = (win["pick"] == win["vegas_pick"])
        disagree = win[~agree]
        lines += [
            "### Where the model disagreed with the favourite",
            "",
            f"- The model took the underdog **{len(disagree)}** times "
            f"({len(disagree) / len(win):.0%} of games).",
            f"- On those games the model went **{_rec(disagree['hit'])}** and Vegas went "
            f"**{_rec(disagree['vegas_hit'])}**." if len(disagree) else "",
            "",
        ]

    for target, label in (("ats", "Against the spread"), ("total", "Totals")):
        d = picks.get(target)
        if d is None:
            continue
        rows = []
        for week, g in d.groupby("week"):
            q = g[g["qualified"]]
            rows.append({
                "week": int(week), "games": len(g),
                "all": _rec(g["hit"]),
                "qualified": _rec(q["hit"]) if len(q) else "—",
            })
        tbl = pd.DataFrame(rows)
        qa = d[d["qualified"]]
        tbl.loc[len(tbl)] = {
            "week": "TOTAL", "games": len(d),
            "all": _rec(d["hit"]),
            "qualified": _rec(qa["hit"]) if len(qa) else "—",
        }
        lines += [f"## {label}", "", _md(tbl), ""]

    lines += [
        "## Verdict",
        "",
    ]
    if win is not None:
        model_pct = win["hit"].mean()
        vegas_pct = win["vegas_hit"].mean()
        verdict = "beat" if model_pct > vegas_pct else ("matched" if model_pct == vegas_pct else "trailed")
        lines.append(
            f"Straight up the model {verdict} the closing moneyline "
            f"({model_pct:.1%} vs {vegas_pct:.1%} over {len(win)} games)."
        )
    for target, label in (("ats", "spread"), ("total", "total")):
        d = picks.get(target)
        if d is None:
            continue
        qa = d[d["qualified"]]
        pct = qa["hit"].mean() if len(qa) else float("nan")
        if len(qa):
            call = "cleared" if pct > BREAK_EVEN else "did not clear"
            lines.append(
                f"On the {label}, qualified picks went {_rec(qa['hit'])} — "
                f"{call} the {BREAK_EVEN:.1%} needed to beat the juice."
            )
        else:
            lines.append(f"On the {label}, no picks cleared the edge threshold.")

    long_run = long_run_context(picks)
    if long_run:
        lines += ["", "### One season is not evidence", "",
                  "The same walk-forward backtest across every season since 2010:", "",
                  long_run, "",
                  "A single season is ~280 games; the standard error on a 50% hit rate over "
                  "that sample is about 3 points, which is wider than the 2.4-point edge you "
                  "need to beat the vig. Read the multi-season row, not the season above."]
    lines.append("")
    return "\n".join(lines)


def long_run_context(picks: dict[str, pd.DataFrame]) -> str:
    rows = []
    for target, label in (("win", "straight up"), ("ats", "spread"), ("total", "total")):
        path = ARTIFACTS / f"backtest_{target}.csv"
        if target not in picks or not path.exists():
            continue
        bt = pd.read_csv(path)
        allrow = bt[bt["season"].astype(str) == "ALL"]
        if allrow.empty:
            continue
        r = allrow.iloc[0]
        rows.append({
            "market": label,
            "bets": int(r["bets"]),
            "win %": f"{float(r['win_pct']):.1%}",
            "units": float(r["units"]),
            "ROI": f"{float(r['roi']):+.1%}",
        })
    return _md(pd.DataFrame(rows)) if rows else ""


def write_files(season: int, picks: dict[str, pd.DataFrame]) -> list[Path]:
    written = []
    weeks = sorted(set().union(*[set(d["week"]) for d in picks.values()]))
    for week in weeks:
        wdir = NFL_DIR / f"Week_{int(week)}"
        wdir.mkdir(parents=True, exist_ok=True)
        # Reuse an existing (empty) placeholder filename if there is one.
        existing = sorted(wdir.glob("*.md"))
        path = existing[0] if existing else wdir / f"Week_{int(week)}.md"
        path.write_text(week_report(season, int(week), picks))
        written.append(path)

    tally = NFL_DIR / "Weeks.md"
    tally.write_text(season_tally(season, picks))
    written.append(tally)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    raw = load_predictions(args.season)
    if not raw:
        raise SystemExit(
            f"no out-of-sample predictions for {args.season} — run "
            "`python3 -m NFL.model.v2.train --target all --save` first"
        )
    picks = {t: _pick_frame(d, t) for t, d in raw.items()}

    print(season_tally(args.season, picks))
    if args.write:
        written = write_files(args.season, picks)
        print(f"wrote {len(written)} files:")
        for p in written:
            print(f"  {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
