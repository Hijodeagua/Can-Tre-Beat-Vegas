# mlb/daily — the daily prediction pipeline

Runs once per day (GitHub Actions, `.github/workflows/daily-report.yml`) and
produces three emails plus the site data for the Can Tre Beat Vegas model
card at `whosyurgoat.app/vegas/mlb`.

## Module map

| Module | Job |
|---|---|
| `run.py` | Orchestrator CLI — `python -m mlb.daily.run [--date D] [--skip-fetch] [--skip-db]` |
| `config.py` | Paths, divisions/leagues, team names, sim defaults |
| `update_games.py` | Pull final scores + remaining schedule from the MLB Stats API (same source/parsing as `mlb/build_games.py`); idempotent tail-replace |
| `ratings.py` | Replay the tuned Elo (`mlb/elo.py`) over the full game file → current ratings + season-to-date standings/run diff |
| `simulate.py` | Score-model calibration, per-game score sim, rest-of-season Monte Carlo with live in-sim Elo updates |
| `grade.py` | Grade the prior day's persisted slate vs actuals; maintain the running ledger `data/mlb/predictions/grades.csv` |
| `emails.py` | HTML renderers for the futures / slate / grade emails |
| `export_site.py` | `web/public/data/mlb/latest.json` (all four tabs) + static per-day history snapshots |
| `db.py` | Optional Postgres (Neon) upserts, gated on `DATABASE_URL` |

## Data flow for a run dated D (ET)

1. Finals through D−1 appended to `data/mlb/games_2009_2026.csv`; remaining
   schedule (incl. D's slate) written to `data/mlb/schedule_2026_remaining.csv`.
2. Elo replayed from 2009 — deterministic, no incremental state to corrupt.
3. `slate_{D−1}.csv` (written yesterday, before those games were played) is
   graded against actuals → `graded_{D−1}.csv` + a ledger row.
4. Today's slate predicted and persisted as `slate_{D}.csv` — tomorrow's
   grading input. Predictions only ever use games completed before D.
5. Futures Monte Carlo over the remaining schedule.
6. Emails rendered to `reports/mlb_daily/{D}/`, manifest to
   `reports/mlb_daily/manifest_latest.json`; site JSON + history snapshot
   into `web/public/data/mlb/`.
7. Postgres upsert when `DATABASE_URL` is configured.

## Modeling notes

- **Win probability** — tuned betting-blind Elo (K=3, +24 home, MOV-weighted
  updates, 60% season carryover), identical to the backtested engine.
- **Scores** — expected margin is a linear map from Elo win probability
  (refit each run on the engine's own history), expected total from recent
  league scoring; each side draws negative-binomial runs; ties resolved by a
  one-run "extra innings" bump. Reported line is the rounded conditional
  mean given the picked side wins (modal exact scorelines are degenerate).
- **Futures** — 2,000 sims/day replay the remaining schedule with live Elo
  updates. 12-team playoff format (3 division winners seeded 1–3, 3 wild
  cards); within-sim ties broken uniformly at random (real MLB head-to-head
  tiebreakers are not modeled).
- **Run differential** shown anywhere is season-to-date as of the pull —
  never full-season hindsight.
