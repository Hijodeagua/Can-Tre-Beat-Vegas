"""College-football daily pipeline orchestrator — the CFB sibling of
`mlb/daily/run.py` and `soccer/clubs/daily/run.py`.

    python -m CFB.daily.run [--date YYYY-MM-DD] [--skip-fetch]
                            [--season-sims N] [--force-email]

For a run dated D (default: today in US/Eastern — a 10:00 UTC run sees
the prior evening's games final):

1. Refresh the current season's rows in data/college_football/games.csv
   from cfbfastR-data (+ the ESPN scoreboard fill-in) — best-effort; the
   committed spine carries a failed fetch.
2. Replay the tuned Elo over the full 2001-present spine and refit the
   score model from that history (deterministic, no incremental state).
   Every result dated on or after D is masked first, so a backdated run
   reproduces that morning's outputs with no hindsight.
3. Grade every previously persisted slate row whose final has now landed;
   append to the running ledger (idempotent by game id).
4. Predict the slate for [D, D+2) and persist it for future grading.
5. Monte Carlo the rest of the regular season (expected wins, bowl /
   undefeated odds, CCG berth and conference title).
6. Write the site JSON (latest + history snapshot).
7. Render the update email HTML + manifest. The email is rendered every
   run but only marked sendable on EMAIL_WEEKDAYS (Mon/Thu), and the
   shared send ledger keeps reruns from double-sending.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

from CFB.daily import emails, export_site, grade, predict, simulate
from CFB.daily.config import (
    EMAIL_FIXTURE_DAYS, EMAIL_REPORTS_DIR, EMAIL_WEEKDAYS, SEASON_SIMS,
)
from CFB.daily.state import build_state


def eastern_today() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def refresh_data() -> None:
    """Run the fetcher as a subprocess so its exit code can't kill the run."""
    proc = subprocess.run([sys.executable, "-m", "CFB.data.fetch_schedule"], check=False)
    if proc.returncode != 0:
        print(f"! CFB.data.fetch_schedule failed (exit {proc.returncode}); using committed data")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=eastern_today(),
                        help="Run date D (ET). Slate covers [D, D+2).")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--season-sims", type=int, default=SEASON_SIMS)
    parser.add_argument("--force-email", action="store_true",
                        help="mark the update email sendable regardless of weekday")
    args = parser.parse_args(argv)
    run_date = args.date

    if not args.skip_fetch:
        print("== Refreshing upstream data")
        refresh_data()

    print("== Building Elo state (full replay + score fits)")
    state = build_state(run_date=run_date)
    print(f"   {len(state.history)} games replayed; season {state.season}; "
          f"{len(state.fbs_teams())} FBS programs; "
          f"{state.score_params.elo_per_point:.1f} Elo/point")

    print("== Grading")
    graded = grade.grade_all(state.games, run_date)
    ledger = grade.ledger_summary(run_date)
    if len(graded):
        print(f"   graded {len(graded)} games "
              f"({int(graded['pick_correct'].astype(bool).sum())} picks correct)")
    print(f"   ledger: {ledger.get('graded', 0)} games, "
          f"accuracy {ledger.get('accuracy', '—')}")

    print("== Predicting slate")
    slate = predict.build_slate(state, run_date)
    predict.persist_slate(slate, run_date)
    print(f"   {len(slate)} games in window")
    for m in slate.itertuples(index=False):
        print(f"   {m.date}  {m.away_team} @ {m.home_team}"
              f"  p_home {m.p_home:.2f}  pick {m.pick}"
              f"  ({m.pred_home_score}-{m.pred_away_score})")

    print("== Season Monte Carlo")
    futures = simulate.simulate_season(state, n_sims=args.season_sims)
    if futures:
        top = futures["teams"][0]
        print(f"   {futures['remaining_games']} games left; "
              f"most projected wins {top['team']} {top['exp_wins']:.1f}")

    print("== Exporting site JSON")
    recent = grade.recent_grades(run_date, days=7)
    export_site.export(state, run_date, slate, futures, ledger, graded, recent)

    print("== Rendering update email")
    week_slate = predict.build_slate(state, run_date, window_days=EMAIL_FIXTURE_DAYS)
    ratings = export_site.ratings_payload(state)
    week = export_site.current_week(state, run_date)
    html = emails.update_html(run_date, ratings, week_slate, recent, ledger,
                              futures, week)
    out = EMAIL_REPORTS_DIR / run_date
    out.mkdir(parents=True, exist_ok=True)
    (out / "update.html").write_text(html, encoding="utf-8")

    from data_jobs.email_ledger import plan
    email_entries = {
        "update": {
            "path": str((out / "update.html").relative_to(EMAIL_REPORTS_DIR.parent.parent)),
            "subject": f"🎓 College Football Update — {run_date} "
                       f"({len(week_slate)} games this week)",
            "date": run_date,
        }
    }
    email_entries = plan(email_entries, EMAIL_REPORTS_DIR / "sent.json")
    email_day = date.fromisoformat(run_date).weekday() in EMAIL_WEEKDAYS
    if not (email_day or args.force_email):
        email_entries["update"]["send"] = False
    manifest = {"date": run_date, "emails": email_entries}
    (EMAIL_REPORTS_DIR / "manifest_latest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"manifest: {json.dumps(manifest)}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
