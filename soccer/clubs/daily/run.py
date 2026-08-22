"""Club-soccer daily pipeline orchestrator.

    python -m soccer.clubs.daily.run [--date YYYY-MM-DD] [--skip-fetch]
                                     [--season-sims N]

For a run dated D (default: today, UTC — a 10:00 UTC run grades the prior
evening's European slate in full):

1. Refresh results.csv (league fixtures + finals) and uefa_results.csv
   from openfootball — best-effort; committed data carries a failed fetch.
2. Rebuild the glued Elo state (leagues + UEFA cross-league matches) and
   refit the outcome + score models from it.
3. Grade every previously persisted slate row whose result has now landed;
   append to the running ledger.
4. Predict the slate for [D, D+2) and persist it for future grading.
5. Monte Carlo the rest of each league season (title / top-4 / relegation).
6. Write the site JSON (latest + history snapshot) and refresh the
   portable ratings artifact.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date

from soccer.clubs.daily import export_site, grade, predict, simulate
from soccer.clubs.daily.config import SEASON_SIMS
from soccer.clubs.daily.state import build_state
from soccer.clubs.data.leagues import TIER1, current_season_for


def refresh_data() -> None:
    """Run the fetchers as subprocesses so their exit codes can't kill the
    run — the committed CSVs are the fallback. fetch_mls MUST run after
    fetch_results: the latter owns results.csv outright and rewrites it
    whole from the openfootball leagues alone, which would drop MLS's rows
    if it ran second."""
    for mod in (
        "soccer.clubs.data.fetch_results",
        "soccer.clubs.data.fetch_uefa",
        "soccer.clubs.data.fetch_mls",
    ):
        proc = subprocess.run([sys.executable, "-m", mod], check=False)
        if proc.returncode != 0:
            print(f"! {mod} failed (exit {proc.returncode}); using committed data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--season-sims", type=int, default=SEASON_SIMS)
    args = parser.parse_args()
    run_date = args.date

    if not args.skip_fetch:
        print("== Refreshing upstream data")
        refresh_data()

    print("== Building Elo state (glued replay + outcome/score fits)")
    state = build_state()

    print("== Grading")
    graded = grade.grade_all(state.results, run_date)
    ledger = grade.ledger_summary()
    if len(graded):
        correct = int(graded["pick_correct"].sum())
        print(f"   graded {len(graded)} matches ({correct} picks correct)")
    print(f"   ledger: {ledger.get('graded', 0)} matches, "
          f"accuracy {ledger.get('accuracy', '—')}")

    print("== Predicting slate")
    slate = predict.build_slate(state, run_date)
    predict.persist_slate(slate, run_date)
    print(f"   {len(slate)} fixtures in window")
    for _, m in slate.iterrows():
        print(f"   {m['date']} {m['league']:>10}  {m['home_team']} v {m['away_team']}"
              f"  H {m['p_H']:.2f} D {m['p_D']:.2f} A {m['p_A']:.2f}"
              f"  pick {m['pick']}  ({m['score_home']}-{m['score_away']})")

    # Futures cover the top flights; second divisions are ratings + slate
    # only for now (Championship promotion odds are one config flip away).
    # MLS naturally opts itself out here rather than needing a special case:
    # its source is a completed-match log with no upcoming-fixture rows, so
    # simulate_league() always finds nothing to simulate and reports the
    # same "no fixtures" skip a season that hasn't published yet would.
    print("== Futures Monte Carlo")
    futures = {}
    for league in TIER1:
        season = current_season_for(league, run_date)
        sim = simulate.simulate_league(state, league, season, n_sims=args.season_sims)
        if sim is None:
            print(f"   {league}: no {season} fixtures upstream yet — skipped")
            futures[league] = {"season": season, "status": "no_fixtures"}
            continue
        futures[league] = sim
        top = sim["clubs"][0]
        print(f"   {league}: {sim['remaining_matches']} matches left; "
              f"title favorite {top['team']} {top['p_title']:.0%}")

    print("== Exporting site JSON + ratings artifact")
    export_site.export(state, run_date, slate, futures, ledger, graded)
    from soccer.clubs.model import export_ratings
    export_ratings.export()
    print("Done.")


if __name__ == "__main__":
    main()
