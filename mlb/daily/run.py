"""Daily pipeline orchestrator.

    python -m mlb.daily.run [--date YYYY-MM-DD] [--skip-fetch] [--skip-db]
                            [--season-sims N] [--game-sims N]

For a run dated D (default: today in US/Eastern):

1. Pull final scores through D-1 and the remaining schedule (statsapi).
2. Replay Elo over the updated game file -> current ratings + standings.
3. Grade the slate predicted on D-1 against its actual results.
4. Predict today's slate (games dated D) and persist it for tomorrow's grade.
5. Monte Carlo the rest of the season (futures).
6. Render the three emails, write the site JSON + history snapshot.
7. Optionally upsert everything to Postgres (DATABASE_URL).

A manifest at reports/mlb_daily/manifest_latest.json tells the workflow which
emails exist and their subjects.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from mlb.daily import export_site, grade, scoring, simulate, update_games
from mlb.daily.config import (
    GAME_SIMS, PREDICTIONS_DIR, REPORTS_DIR, SEASON_SIMS,
)
from mlb.daily.emails import futures_html, grade_html, slate_html
from mlb.daily.ratings import build_state


def eastern_today() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=eastern_today(),
                    help="Run date D (ET). Slate covers D; grades cover D-1.")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="Skip the statsapi pull (offline/dev runs).")
    ap.add_argument("--skip-db", action="store_true")
    ap.add_argument("--season-sims", type=int, default=SEASON_SIMS)
    ap.add_argument("--game-sims", type=int, default=GAME_SIMS)
    args = ap.parse_args(argv)

    run_date = args.date
    yesterday = (date.fromisoformat(run_date) - timedelta(days=1)).isoformat()

    if args.skip_fetch:
        print("1/7 fetch skipped (--skip-fetch)")
    else:
        summary = update_games.update(yesterday)
        print(f"1/7 data updated: {summary}")

    state = build_state()
    print(f"2/7 Elo replayed over {len(state.history)} games")

    graded = grade.grade_day(yesterday, state.games)
    ledger_row = None
    if graded is not None:
        ledger_row = grade.update_ledger(yesterday, graded)
        played = int(ledger_row["games"])
        print(f"3/7 graded {yesterday}: {int(ledger_row['correct'])}/{played}")
    else:
        print(f"3/7 no slate on file for {yesterday} - nothing to grade")

    schedule = update_games.read_schedule()
    todays = [g for g in schedule if g["date"] == run_date]
    params = simulate.calibrate(state.history, state.games)
    rates = scoring.rates_from_games(state.games)
    slate = simulate.slate_predictions(
        state.ratings, todays, params, n=args.game_sims, rates=rates
    )
    if not slate.empty:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        slate.to_csv(grade.slate_path(run_date), index=False)
    print(f"4/7 slate for {run_date}: {len(slate)} games "
          f"(margin fit: {params.margin_slope:.2f}x + "
          f"{params.margin_intercept:.2f}, total {params.total_mean:.2f}, "
          f"NB r={params.dispersion:.1f})")

    remaining = [g for g in schedule if g["date"] >= run_date]
    futures = simulate.simulate_season(
        state.ratings, remaining, state.standings, n_sims=args.season_sims
    )
    print(f"5/7 futures: {len(remaining)} remaining games x "
          f"{args.season_sims} sims")

    out = REPORTS_DIR / run_date
    out.mkdir(parents=True, exist_ok=True)
    emails = {}

    (out / "futures.html").write_text(
        futures_html(run_date, futures, state.standings), encoding="utf-8"
    )
    emails["futures"] = {
        "path": str((out / "futures.html").relative_to(REPORTS_DIR.parent.parent)),
        "subject": f"⚾ MLB Futures — {run_date}",
    }
    if not slate.empty:
        (out / "slate.html").write_text(
            slate_html(run_date, slate), encoding="utf-8"
        )
        emails["slate"] = {
            "path": str((out / "slate.html").relative_to(REPORTS_DIR.parent.parent)),
            "subject": f"⚾ MLB Slate — {run_date} ({len(slate)} games)",
        }
    if graded is not None and ledger_row is not None and ledger_row["games"]:
        (out / "grade.html").write_text(
            grade_html(yesterday, graded, ledger_row), encoding="utf-8"
        )
        emails["grade"] = {
            "path": str((out / "grade.html").relative_to(REPORTS_DIR.parent.parent)),
            "subject": (
                f"⚾ MLB Grade — {yesterday}: "
                f"{int(ledger_row['correct'])}/{int(ledger_row['games'])} "
                f"({100 * ledger_row['accuracy']:.0f}%)"
            ),
        }

    # Rebuild yesterday's slate frame for the snapshot page footer.
    y_slate = None
    if grade.slate_path(yesterday).exists():
        y_slate = pd.read_csv(grade.slate_path(yesterday))
    export_site.write_history_snapshot(yesterday, graded, ledger_row, y_slate)
    export_site.export_latest(
        run_date, slate, futures, state.standings, graded,
        yesterday if graded is not None else None,
    )
    print("6/7 emails + site JSON written")

    if args.skip_db:
        print("7/7 db skipped (--skip-db)")
    else:
        from mlb.daily.db import write_postgres
        write_postgres(run_date, slate, futures, ledger_row)
        print("7/7 db step done")

    manifest = {"date": run_date, "graded_date": yesterday, "emails": emails}
    (REPORTS_DIR / "manifest_latest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    print(f"manifest: {json.dumps(manifest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
