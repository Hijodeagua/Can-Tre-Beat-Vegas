# SP-AUDIT: starting-pitcher work, probables plumbing, and grading idempotency

Phase 0 audit for wiring starting pitchers into the MLB Elo model.
Audited at `4821314` (2026-08-14), branch `claude/mlb-starting-pitcher-elo-ct0nae`.

## 1. Prior starting-pitcher work

**No pitcher adjustment was ever written into the rating path and later
removed or stubbed.** The live rating code (`mlb/elo.py`,
`mlb/daily/simulate.py`) has never referenced starter identity in any
reachable commit. What exists is a deliberate *data-and-analysis* layer that
stopped short of the model, plus one abandoned unmerged effort:

| Commit | Date | What it did |
|---|---|---|
| `3ea14c7` | 2026-08-07 | Pitcher data layer: `mlb/build_pitchers.py` (BRef season CSVs → `data/mlb/pitcher_seasons.csv`, IP≥125 seasons only), `mlb/build_crosswalk.py` (Chadwick register → `data/mlb/pitcher_id_crosswalk.csv`: bbref/retro/MLBAM IDs), `mlb/build_starters.py` (Retrosheet game-log starters, run on a GitHub runner because retrosheet.org is unreachable from the dev sandbox), probable-pitcher hydration in `update_games.py`, SP name columns in slate email/DB/site. Comments state explicitly: "Display only - the Elo model does not use starter identity." |
| `3b675b2` | 2026-08-07 | `data/mlb/starting_pitchers_2009_2025.csv` — 39,773 rows, one per game, both starters' Retrosheet IDs + names. Identity only; **no pitching lines** (no IP/H/ER/BB/SO). |
| `1709d1a` | 2026-08-07 | `mlb/backtest_starters.py` + `reports/mlb_starter_quality_backtest.md`: with Elo controlled, prior-completed-season ERA+/FIP deltas are highly significant (p < 1e-4); OOS 2021-2025 log-loss −0.002 on the matched subset. Conclusion: signal is real, **coverage is the binding constraint** — the IP≥125 season table matches both starters in only 25–33% of games. Analysis only; never wired into predictions. |
| `a24a74b` | 2026-08-07 | Review fixes (Postgres SP-column migration, crosswalk fails loudly on gaps). |

Abandoned effort: GitHub Actions history shows pushes to a deleted, never-merged
branch `claude/mlb-elo-data-pipeline-5wb78q` (2026-08-05) titled *"MLB:
per-start pitcher lines + reliever usage from retrosplits, 2005-…"*. The branch
is gone from all refs, so that per-start ingest was discarded, not merged. It is
the closest thing to "prior work later removed" — and it validates retrosplits
as the per-start-line source (see §5).

Takeaway: the prior conclusion ("real signal, coverage-limited") argues *for*
the rolling-game-score design in this task — a rGS built from every start has
no IP≥125 coverage cliff, and its fallback ladder never drops a game.

## 2. Probables: where fetched, how persisted, what schema

**Fetch.** `mlb/daily/update_games.py` pulls
`statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R&hydrate=probablePitcher`
once per daily run (10:00 UTC). `_probable`/`_probable_id` extract each side's
probable pitcher full name and MLBAM id; empty string when unannounced.

**Persistence chain** (each step overwritten/upserted per run):

1. `data/mlb/schedule_2026_remaining.csv` — full rewrite each run; columns
   `away_sp, home_sp, away_sp_id, home_sp_id` (names + MLBAM ids). Only the
   *latest* snapshot survives; no history of probables-as-of-slate-time beyond
   step 2.
2. `data/mlb/predictions/slate_{date}.csv` — written by
   `simulate.slate_predictions` at slate time; carries `away_sp, home_sp`
   **names only. The MLBAM ids are dropped here** — a gap the pitcher
   adjustment must fix, since name-joins are exactly what the crosswalk work
   was built to avoid.
3. Neon Postgres `mlb_slate_predictions` (`mlb/daily/db.py`) — `away_sp`,
   `home_sp` TEXT, upsert keyed `(date, away, home, game_num)`. No id columns.
4. Slate email/site — display strings, with the footer disclaimer ("shown for
   context only; the model ... does not adjust for starter identity").

**Model use: none.** `slate_predictions` computes `p_home` purely from team
Elo + 24 home advantage; `g.get("away_sp", "")` is passed through for display.

## 3. Grading idempotency and the 2026-08-10 duplicate runs

**The computation layer is idempotent; the email layer is not.**

- `grade.update_ledger` drops any existing ledger row for the date
  (`ledger[ledger.date != d]`) and recomputes every cumulative column from the
  full ledger. Re-grading a day replaces, never appends.
- `grade.grade_day` and the graded/slate CSVs are whole-file writes keyed by
  date. `update_games.update` replaces the refetched tail of the games file
  rather than appending. Postgres writes are `ON CONFLICT ... DO UPDATE`
  upserts. A rerun therefore converges to the same state.
- Email sending has **no** idempotency: three `action-send-mail` steps fire on
  every successful pipeline run, so every extra run re-sends that day's
  reports.

**Forensics on 8/10.** The workflow ran twice: 06:08 UTC (a `push` event —
PR #22's merge touched `mlb/daily/**`, matching the workflow's push-path
trigger, which exists to send first emails on merge) and 11:00 UTC (the
schedule). Each produced a commit ("MLB daily: 2026-08-10" — `f114cd5`,
`f16cf48`); the diff between them is **one line** in
`web/public/data/mlb/latest.json` (a generated timestamp). Slate, graded
files, and the ledger were byte-identical. The same mechanism sent the 8/07
reports three times on launch day (three "MLB daily: 2026-08-07" commits).

**Verdict on the headline: 51/94 is correct.** `data/mlb/predictions/grades.csv`
has exactly one row per date 8/07–8/13; games sum to 94, correct to 51,
cumulative log-loss 0.6874, Brier 0.2470 — matching the reported record. The
duplicate runs re-sent emails but did not double-count anything.

Fix direction (Phase 6): an idempotency key on `(report_type, date)` that
turns a rerun into an update/no-send unless content changed.

## 4. The "dead" daily-report.yml — already fixed, root cause documented

The failing workflow was the *pre-rewrite* `daily-report.yml` ("Daily Odds
Report"): its trigger block contained an **empty `schedule:` key**, making the
file invalid, so from late July GitHub logged a failed, job-less run
("No jobs were run") for every push. Actions history shows these push-event
failures from 2026-07-31 through 2026-08-07 on all branches.

Commit `0b61ad3` (merged 2026-08-07, PR #21) replaced the file wholesale with
the current, working MLB daily pipeline; every run since 2026-08-07 06:27 UTC
has succeeded (schedule and push events alike). **There is no dead workflow to
delete today** — the Phase 6 item reduces to the duplicate-send fix above,
plus pruning the now-stale push trigger if we want reruns to be schedule-only.

## 5. Baseline to reproduce, and environment constraints for Phase 1

**The 0.680 claim** is `data/mlb/elo_params.json` → `best.log_loss = 0.67961`
(accuracy 0.5665, Brier 0.24332, n = 34,205), produced by `mlb/tune_elo.py`:
full 2009→present Elo replay, **evaluated on seasons ≥ 2012** so the
fresh-start burn-in doesn't dominate. Phase 4's variant (a) must reproduce
this number with the same eval window before anything else is trusted.
Note the tuning grid was scored on all seasons ≥2012 with no train/holdout
split — the new backtest's 2010-2021 tune / 2022-2025 holdout discipline is
stricter by design.

**Network constraints (verified from this sandbox):**

- `statsapi.mlb.com` and `retrosheet.org`: blocked (proxy CONNECT 403). Same
  constraint that forced `fetch-starters.yml` to run on a GitHub runner.
- GitHub (raw/api/git): reachable. `chadwickbureau/retrosplits` has per-game,
  per-player day-by-day files **through 2025**, including pitching lines
  (outs, H, R, ER, BB, SO) keyed by Retrosheet game and person ids — exactly
  the Phase 1 row shape for 2010–2025, joinable to MLBAM ids via the existing
  crosswalk builder.
- `DATABASE_URL` is not present in this sandbox; Neon writes only happen in CI
  (the established `db.py` pattern). The ingest therefore writes a committed
  CSV artifact as source of truth plus an optional Postgres upsert that runs
  where credentials exist.

**Consequent Phase 1 shape:** 2010–2025 per-start lines from retrosplits
(runs here, now); 2026-to-date and the daily forward fill from statsapi
boxscores with on-disk checkpointing, executed runner-side like
`fetch-starters.yml`. The task's "Retrosheet game logs via pybaseball" route
cannot supply per-start pitching lines (game logs carry starter identity
only) and retrosheet.org is unreachable anyway; retrosplits is the same
Retrosheet data, already game-level, and reachable.
