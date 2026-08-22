# soccer/clubs/daily — the club-soccer daily pipeline

Runs once per day (GitHub Actions, `.github/workflows/soccer-daily.yml`) —
the club-football sibling of `mlb/daily`: refresh results, rebuild the
glued Elo, grade what played, predict what's next, Monte Carlo the tables,
publish the site JSON.

## Module map

| Module | Job |
|---|---|
| `run.py` | Orchestrator CLI — `python -m soccer.clubs.daily.run [--date D] [--skip-fetch] [--season-sims N]` |
| `config.py` | Paths, slate window, sim defaults |
| `state.py` | One glued Elo replay + in-run outcome/score model fits, shared by every step |
| `scoring.py` | Elo expectation → Poisson goal rates (margin map refit each run; league totals from the last 2 seasons) |
| `predict.py` | Slate for [D, D+2): W/D/L probabilities, pick, most likely scoreline; persisted as `slate_{D}.csv` |
| `grade.py` | Grade persisted slates once results land; running ledger at `data/soccer_clubs/predictions/grades.csv` |
| `simulate.py` | Per-league rest-of-season Monte Carlo with live in-sim Elo (title / top-4 / relegation / expected points) |
| `export_site.py` | `web/public/data/soccer/latest.json` + per-day history snapshots |

## Data flow for a run dated D (UTC)

1. `fetch_results` + `fetch_uefa` refresh the committed CSVs (best-effort;
   the 10:00 UTC schedule means the prior evening's European matches are
   final and upstream has usually caught up).
2. The glued replay (`model/europe.py`) rebuilds every league pool from
   scratch — deterministic, no incremental state to corrupt — and the
   outcome + score models are refit from that history in-run (no pickle
   drift in CI).
3. Every previously persisted slate row whose result has now landed is
   graded into the ledger. Grading is idempotent by match, so overlapping
   slate windows and postponed fixtures resolve cleanly whenever they play.
4. The slate for [D, D+2) is predicted and persisted — tomorrow's grading
   input. Predictions only ever use matches completed before the run.
5. Rest-of-season Monte Carlo per league (default 1000 sims): sampled
   Poisson scores with live Elo updates inside each sim; tie-breaks
   uniform (goal difference is not modeled as a tiebreaker).
6. Site JSON + history snapshot + the portable ratings artifact.

## Modeling notes

- **W/D/L** — the multinomial logistic layer from `model/train.py`
  (Elo gap + squad-economics differentials), refit in-run on identical
  training rows. Holdout (2024-25 + 2025-26): log loss 0.9902 vs 1.0750
  class-frequency baseline.
- **Scores** — independent Poisson per side; expected margin is linear in
  the Elo home expectancy, totals are league-specific. Independent is
  deliberate: observed 1-1 (11.7%) ≈ independent-Poisson 1-1 (11.8%) on
  25k matches, so Dixon–Coles is left on the roadmap.
- **Fixtures** — from the openfootball country repos, which publish new
  seasons before football.json. A league whose repo lags (Ligue 1 at
  2026-27 launch) simply reports `no_fixtures` until upstream catches up.
- **No emails yet** — the MLB pipeline's three-email machinery is the
  template if/when this earns an inbox slot.
