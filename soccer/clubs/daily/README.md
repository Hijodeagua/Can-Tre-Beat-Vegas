# soccer/clubs/daily — the club-soccer daily pipeline

Runs once per day (GitHub Actions, `.github/workflows/soccer-daily.yml`) —
the club-football sibling of `mlb/daily`: refresh results, rebuild the
glued Elo, grade what played, predict what's next, Monte Carlo the tables,
publish the site JSON, and render the twice-weekly update email.

## Module map

| Module | Job |
|---|---|
| `run.py` | Orchestrator CLI — `python -m soccer.clubs.daily.run [--date D] [--skip-fetch] [--season-sims N] [--force-email]` |
| `config.py` | Paths, slate window, sim defaults, email weekdays |
| `state.py` | One glued Elo replay + in-run outcome/score model fits, shared by every step |
| `scoring.py` | Elo expectation → Poisson goal rates (margin map refit each run; league totals from the last 2 seasons) |
| `predict.py` | Slate for [D, D+2): W/D/L probabilities, pick, most likely scoreline; persisted as `slate_{D}.csv` |
| `grade.py` | Grade persisted slates once results land; running ledger at `data/soccer_clubs/predictions/grades.csv` |
| `simulate.py` | Per-league rest-of-season Monte Carlo with live in-sim Elo (title / UCL / UEL / relegation / expected points + position) |
| `export_site.py` | `web/public/data/soccer/latest.json` (incl. `elo_history` for the site's daily-updating Elo chart) + per-day history snapshots |
| `emails.py` | The update email HTML: week's fixtures, past week + rolling tracker, Opta-style final-table forecasts |

## Data flow for a run dated D (UTC)

1. `fetch_results` + `fetch_uefa` + `fetch_mls` + `fetch_xg` refresh the committed CSVs
   (best-effort; the 10:00 UTC schedule means the prior evening's European
   matches are final and upstream has usually caught up). Order matters:
   `fetch_results` rewrites `results.csv` whole from the openfootball
   leagues alone, so `fetch_mls` — which only ever touches its own rows —
   always runs after it, or MLS would be silently dropped.
2. The glued replay (`model/europe.py`) rebuilds every league pool from
   scratch — deterministic, no incremental state to corrupt — and the
   outcome + score models are refit from that history in-run (no pickle
   drift in CI).
3. Every previously persisted slate row whose result has now landed is
   graded into the ledger. Grading is idempotent by match, so overlapping
   slate windows and postponed fixtures resolve cleanly whenever they play.
4. The slate for [D, D+2) is predicted and persisted — tomorrow's grading
   input. Predictions only ever use matches completed before the run.
5. Rest-of-season Monte Carlo per league (default 10000 sims): sampled
   Poisson scores with live Elo updates inside each sim; tie-breaks
   uniform (goal difference is not modeled as a tiebreaker). Per club:
   expected points and finishing position plus P(title), P(top 4 = UCL),
   P(5th–6th = UEL), P(bottom 3 = relegation).
6. Site JSON + history snapshot + the portable ratings artifact.
7. The update email (`reports/soccer/{D}/update.html`) is rendered every
   run; the manifest marks it sendable only on Mondays and Thursdays
   (`EMAIL_WEEKDAYS`), the workflow delivers it over the same SMTP
   secrets as the MLB emails, and `data_jobs/email_ledger.py` keeps
   reruns from double-sending.

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
  MLS reports the same status permanently — its source is a completed-match
  log, not a fixture list, so it never has a slate or futures sim, only
  ratings and squad economics.
- **Cross-league ranking for an unglued league** — MLS (or any future
  league with `glued=False` in `data/leagues.py`) has no shared matches
  against the ten UEFA-glued leagues, so it can't be placed on their Elo
  scale by measurement. `model/value_anchor.py` fits ln(squad value) ->
  Elo across the glued leagues' clubs (R²~0.7, ~190 clubs as of 2026) and
  applies that line to the unglued league's own market-value upload,
  exported as `anchoredElo`/`anchoredTop4Elo`/… alongside the league's
  own (unglued) `avgElo`. The site's League Rankings tab uses the anchor
  where available and marks it with a dagger — never presented as
  equivalent to a measured rating. Onboarding a new unglued league needs
  only a `League` entry with `glued=False`, a fetcher, and a
  `market_values` upload under that league's key; nothing in
  `value_anchor.py` or the exporters is MLS-specific.
- **Emails** — one combined update email, twice a week (Mon/Thu), built
  on the MLB pipeline's manifest + send-ledger machinery. A separate
  weekly models check (`data_jobs/reports/models_check.py`,
  `.github/workflows/weekly-models-check.yml`) covers cross-model
  performance and feature importances.
