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

from mlb.daily import export_site, grade, scoring, simulate, sp_state, update_games
from mlb.daily.config import (
    ACTIVE_MODEL, GAME_SIMS, MODEL_V1, MODEL_V2, REPORTS_DIR, SEASON_SIMS,
    SHADOW_MODEL, predictions_dir,
)
from mlb.daily.emails import futures_html, grade_html, slate_html
from mlb.daily.ratings import build_state


def eastern_today() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def build_slate_for_version(model_version: str, ratings: dict,
                            todays: list, params, rates,
                            adjustments: dict | None, n: int) -> pd.DataFrame:
    """One model version's slate. Whether `adjustments` actually reaches the
    win-probability calculation depends only on `model_version` (MODEL_V2
    uses them, MODEL_V1 doesn't) - never on whether that version happens to
    be playing the active or shadow role this run. That indifference is
    deliberate: on a cutover day the same version can be either, and a
    version-keyed check (not a role-keyed one) is what makes the adjustment
    follow the model rather than silently staying pinned to whichever role
    originally computed it."""
    adj = adjustments if model_version == MODEL_V2 else None
    return simulate.slate_predictions(
        ratings, todays, params, n=n, rates=rates,
        adjustments=adj, model_version=model_version,
    )


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

    # Grade every version's bucket that has a slate for yesterday, not just
    # whichever is active today - on a cutover day (ACTIVE_MODEL just
    # changed), yesterday's slate was produced by the OLD active model and
    # lives in ITS bucket, not the new one. Grading by version rather than
    # by role means that boundary day still gets graded instead of silently
    # dropped, and behaves identically to before on any non-cutover day
    # (only the actually-active/shadow versions ever have a file to find).
    graded_frames: dict[str, pd.DataFrame] = {}
    ledger_rows: dict[str, pd.Series] = {}
    for v in (MODEL_V1, MODEL_V2):
        d = predictions_dir(v)
        g = grade.grade_day(yesterday, state.games, d)
        if g is not None:
            graded_frames[v] = g
            ledger_rows[v] = grade.update_ledger(yesterday, g, d)
            played = int(ledger_rows[v]["games"])
            print(f"3/7 graded {yesterday} ({v}): "
                  f"{int(ledger_rows[v]['correct'])}/{played}")
    graded = graded_frames.get(ACTIVE_MODEL)
    ledger_row = ledger_rows.get(ACTIVE_MODEL)
    shadow_ledger_row = ledger_rows.get(SHADOW_MODEL) if SHADOW_MODEL else None
    if graded is None:
        print(f"3/7 no slate on file for {yesterday} under "
              f"{ACTIVE_MODEL} - nothing to grade")

    schedule = update_games.read_schedule()
    todays = [g for g in schedule if g["date"] == run_date]
    params = simulate.calibrate(state.history, state.games)
    rates = scoring.rates_from_games(state.games)

    # Pitcher/rest-travel adjustments are shared by whichever role (active
    # or shadow) currently holds MODEL_V2 - computed once regardless, so
    # cutover (flipping which role that is) doesn't need its own branch.
    adjustments = None
    if todays and MODEL_V2 in (ACTIVE_MODEL, SHADOW_MODEL):
        pbook, rtbook = sp_state.build_books(state.games, run_date)
        adjustments = sp_state.slate_adjustments(pbook, rtbook, todays)

    def predict(model_version: str) -> pd.DataFrame:
        return build_slate_for_version(
            model_version, state.ratings, todays, params, rates,
            adjustments, args.game_sims)

    active_dir = predictions_dir(ACTIVE_MODEL)
    slate = predict(ACTIVE_MODEL)
    if not slate.empty:
        active_dir.mkdir(parents=True, exist_ok=True)
        slate.to_csv(grade.slate_path(run_date, active_dir), index=False)

    shadow_slate = pd.DataFrame()
    if SHADOW_MODEL and todays:
        shadow_slate = predict(SHADOW_MODEL)
        if not shadow_slate.empty:
            shadow_dir = predictions_dir(SHADOW_MODEL)
            shadow_dir.mkdir(parents=True, exist_ok=True)
            shadow_slate.to_csv(
                grade.slate_path(run_date, shadow_dir), index=False)
    print(f"4/7 slate for {run_date}: {len(slate)} games ({ACTIVE_MODEL}) "
          f"(+{len(shadow_slate)} shadow: {SHADOW_MODEL}) "
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
            slate_html(run_date, slate, shadow=shadow_slate), encoding="utf-8"
        )
        emails["slate"] = {
            "path": str((out / "slate.html").relative_to(REPORTS_DIR.parent.parent)),
            "subject": f"⚾ MLB Slate — {run_date} ({len(slate)} games)",
        }
    if graded is not None and ledger_row is not None and ledger_row["games"]:
        (out / "grade.html").write_text(
            grade_html(yesterday, graded, ledger_row,
                       shadow_ledger_row=shadow_ledger_row),
            encoding="utf-8"
        )
        d_mean = ledger_row.get("cum_d_ll_mean")
        subject_tail = (
            f"Δll {d_mean:+.4f}±{ledger_row.get('cum_d_ll_se'):.4f} vs home"
            if pd.notna(d_mean) else
            f"{int(ledger_row['correct'])}/{int(ledger_row['games'])}"
        )
        emails["grade"] = {
            "path": str((out / "grade.html").relative_to(REPORTS_DIR.parent.parent)),
            "subject": f"⚾ MLB Grade — {yesterday}: {subject_tail}",
        }

    # Rebuild yesterday's slate frame for the snapshot page footer - from
    # the active model's own bucket, same reasoning as the grading loop
    # above (yesterday's slate may predate a same-day cutover).
    y_slate = None
    y_slate_path = grade.slate_path(yesterday, active_dir)
    if y_slate_path.exists():
        y_slate = pd.read_csv(y_slate_path)
    export_site.write_history_snapshot(yesterday, graded, ledger_row, y_slate)
    export_site.export_latest(
        run_date, slate, futures, state.standings, graded,
        yesterday if graded is not None else None,
        shadow_slate=shadow_slate,
    )
    print("6/7 emails + site JSON written")

    if args.skip_db:
        print("7/7 db skipped (--skip-db)")
    else:
        from mlb.daily.db import write_postgres
        write_postgres(run_date, slate, futures, ledger_row)
        print("7/7 db step done")

    # Idempotency: consult the send ledger so a rerun re-sends nothing
    # unless the content actually changed (then it goes out as "(updated)").
    from mlb.daily.send_ledger import plan
    emails = plan(emails, run_date, yesterday)

    manifest = {"date": run_date, "graded_date": yesterday, "emails": emails}
    (REPORTS_DIR / "manifest_latest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    print(f"manifest: {json.dumps(manifest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
