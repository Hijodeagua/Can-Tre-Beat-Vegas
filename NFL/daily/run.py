"""NFL daily pipeline orchestrator — the NFL sibling of `CFB/daily/run.py`.

    python -m NFL.daily.run [--date YYYY-MM-DD] [--skip-fetch]
                            [--season-sims N] [--force-email]

For a run dated D (default: today in US/Eastern — a 10:00 UTC run sees
the prior evening's games final):

1. Refresh data/schedules/nflverse_games.csv (`NFL/model/schedule.py
   --refresh`) — best-effort; the committed spine carries a failed fetch.
2. Replay the tuned Elo over the full 1999-present spine and refit the
   score model from that history (deterministic, no incremental state).
   Every result dated on or after D is masked first, so a backdated run
   reproduces that morning's outputs with no hindsight.
3. Grade every previously persisted slate row whose final has now landed;
   append to the running ledger (idempotent by game id).
4. Predict the next NFL week's unplayed games and persist the slate.
5. Monte Carlo the rest of the season and the playoff bracket.
6. Write the site JSON (latest + history snapshot).
7. Render the update email HTML + manifest. Rendered every run, marked
   sendable only on EMAIL_WEEKDAYS (Tue/Thu); the shared send ledger
   keeps reruns from double-sending.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

from NFL.daily import emails, export_site, grade, predict, simulate
from NFL.daily.config import EMAIL_REPORTS_DIR, EMAIL_WEEKDAYS, REPO_ROOT, SEASON_SIMS
from NFL.daily.state import build_state


def eastern_today() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def refresh_data() -> None:
    """Run the schedule refresh as a subprocess so its exit code can't kill the run."""
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "NFL" / "model" / "schedule.py"),
                           "--refresh"], check=False)
    if proc.returncode != 0:
        print(f"! schedule refresh failed (exit {proc.returncode}); using committed data")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=eastern_today(),
                        help="Run date D (ET). Slate covers the next NFL week.")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--season-sims", type=int, default=SEASON_SIMS)
    parser.add_argument("--force-email", action="store_true",
                        help="mark the update email sendable regardless of weekday")
    args = parser.parse_args(argv)
    run_date = args.date

    if not args.skip_fetch:
        print("== Refreshing nflverse schedule")
        refresh_data()

    print("== Building Elo state (full replay + score fits)")
    state = build_state(run_date=run_date)
    print(f"   {len(state.history)} games replayed; season {state.season}; "
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
    week, label = export_site.current_week(state, run_date)
    print(f"   {len(slate)} games ({label})")
    for m in slate.itertuples(index=False):
        print(f"   {m.date}  {m.away_team} @ {m.home_team}"
              f"  p_home {m.p_home:.2f}  pick {m.pick}"
              f"  line {m.elo_spread:+.1f}  ({m.pred_home_score}-{m.pred_away_score})")

    print("== Season Monte Carlo")
    futures = simulate.simulate_season(state, n_sims=args.season_sims)
    if futures:
        top = futures["teams"][0]
        print(f"   {futures['remaining_games']} games left; "
              f"Super Bowl favourite {top['team']} {100 * top['p_sb']:.0f}%")

    print("== Exporting site JSON")
    recent = grade.recent_grades(run_date, days=7)
    payload = export_site.export(state, run_date, slate, futures, ledger, graded, recent)

    print("== Rendering update email")
    html = emails.update_html(run_date, payload["ratings"], slate, recent, ledger,
                              futures, label, payload["divisions"])
    out = EMAIL_REPORTS_DIR / run_date
    out.mkdir(parents=True, exist_ok=True)
    (out / "update.html").write_text(html, encoding="utf-8")

    from data_jobs.email_ledger import plan
    email_entries = {
        "update": {
            "path": str((out / "update.html").relative_to(REPO_ROOT)),
            "subject": f"🏈 NFL Elo Update — {run_date}"
                       + (f" ({label}: {len(slate)} games)" if label else ""),
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
