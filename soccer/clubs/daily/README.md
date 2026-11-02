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
| `predict.py` | Slate for [D, D+2): W/D/L probabilities, pick, likeliest scoreline consistent with that pick, and every per-side number behind them (`SIDE_METRICS`); persisted as `slate_{D}.csv`, which keeps its original columns |
| `trends.py` | The week's fixtures against the same clubs' own history — the email's aggregate and biggest-movers tables |
| `grade.py` | Grade persisted slates once results land; running ledger at `data/soccer_clubs/predictions/grades.csv` |
| `simulate.py` | Per-league rest-of-season Monte Carlo with live in-sim Elo (title / UCL / UEL / relegation / expected points + position) |
| `../model/mls_forecast.py` | MLS's equivalent, run from `run.py`: Supporters' Shield, conference seeding and the MLS Cup bracket, off a schedule reconstructed from the league's format |
| `export_site.py` | `web/public/data/soccer/latest.json` (incl. `elo_history` + `elo_projection` for the site's daily-updating Elo chart and its projected half) + per-day history snapshots |
| `emails.py` | The update email HTML: week's fixtures, past week + rolling tracker, Opta-style final-table forecasts |

## Data flow for a run dated D (UTC)

1. `fetch_xg` (a shim over `data/understat.py`) + `fd_results` +
   `fetch_results` + `fetch_uefa` + `fetch_mls` + `results_overlay` +
   `fetch_shots` refresh the committed CSVs (best-effort; the 10:00 UTC
   schedule means the prior evening's European matches are final and
   upstream has usually caught up). Order matters: the two secondary
   results sources refresh first so `fetch_results` can overlay their
   finals onto the openfootball fixture list it rewrites `results.csv`
   from; `fetch_mls` — which only ever touches its own rows — runs after
   it, or MLS would be silently dropped; `results_overlay` runs again on
   the committed file so a failed openfootball fetch still gets the day's
   finals; `fetch_shots` goes last because it snaps onto `results.csv`
   finals. `state.feed_status` then logs each form feed's freshness and
   `state.results_status` each league's results freshness (below).
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
6. The MLS forecast (`MLS_SIMS` replays, default 20000): Supporters'
   Shield, conference seeding, playoff qualification and the MLS Cup
   bracket. MLS has no upcoming-fixture rows to replay, so the remaining
   regular season is reconstructed from the league's format first — exact
   for the conference round robin, per-club home/away quotas for the
   cross-conference matches, which each replay pairs up its own way. The
   reconstruction is re-verified every run and the block is skipped (not
   guessed at) if the season stops matching the format. It lands under
   `mls_forecast` in the site JSON rather than in `futures`, since it
   answers different questions and carries different columns — and it
   carries its own chart data (`chart`: Elo by matches played, history
   plus projection) and `bracket` block, because MLS's chart cannot use
   the shared dated `elo_history`/`elo_projection` pair.
7. Site JSON + history snapshot + the portable ratings artifact.
8. The update email (`reports/soccer/{D}/update.html`) is rendered every
   run; the manifest marks it sendable only on Mondays and Thursdays
   (`EMAIL_WEEKDAYS`), the workflow delivers it over the same SMTP
   secrets as the MLB emails, and `data_jobs/email_ledger.py` keeps
   reruns from double-sending.

## Modeling notes

- **W/D/L** — the learner from `model/train.py` (`LEARNER`, a random
  forest today), refit in-run on identical training rows. Inputs are the
  home and away Elo, the venue-adjusted Elo gap, squad-economics
  differentials, rolling xG and shots-on-target form, the whole Understat
  advanced layer, and league/tier/season context — every one of them as a
  home-minus-away difference **and** as the two per-side levels it was
  made from (see SPEC.md, "Levels as well as differences"). Holdout
  (2024-25 + 2025-26, 7,906 matches): log loss 1.0134, against 1.0260
  Elo-gap-only and 1.0763 class-frequency.
- **Per-side stats** — the model trains on differences, but the site
  publishes both halves of every one: `state.outcome_probs` re-attaches
  the features with `keep_sides=True`, `predict.SIDE_METRICS` names and
  groups them, and each slate row carries a nested `sides` block. The
  prediction is bit-identical either way — the learner is handed exactly
  `FEATURES` — and the training frame never passes `keep_sides`, so it
  keeps precisely the columns it always had. A metric with no reading is
  absent rather than zero, and `wage_z` is withheld entirely while no
  wage source fills it.
- **Week against trend** (`trends.py`) — the email's answer to "why does
  this slate look like this": every club playing this week measured
  against **its own** mean in completed prior seasons, never a league
  average, with the gaps scaled by how much clubs differ on that metric
  so an xG swing and a PPDA swing can be ranked together.
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

## Results sources and freshness

- **Fixture backbone: openfootball** (`data/fetch_results.py`). It
  publishes every league's full-season fixture list and most finals, but
  it stalled on 2026-09-20 for 2026-27 (no played match in any of the six
  leagues after that date) and every run stayed green for six weeks.
- **Overlay: Understat + football-data.co.uk** (`data/results_overlay.py`).
  Understat's processed table (`understat_matches.csv`, the xG feed) carries
  finals for the five top flights; football-data.co.uk (`data/fd_results.py`
  → `fd_results.csv`, the same publisher the shots feed uses) carries the
  Championship as well. Precedence is openfootball → Understat →
  football-data: a later source never overwrites an earlier score, only
  cross-checks it. A source row matches a fixture on the exact
  (league, date, home, away) key, or on the same pairing within
  `WINDOW_DAYS` (7) — openfootball lists provisional dates that the leagues
  move by a day or two, and the fixture is moved to the date it was played
  on. Score disagreements are logged and the openfootball score kept (they
  are awarded results: Sassuolo–Pescara 2016, Bastia–Lyon 2017, Verona–Roma
  2020). A source row with no fixture to land on is never appended. Every
  run writes `data/results_overlay_report.json` with the counts, the
  conflicts and the unmatched rows.
- **Results freshness** (`state.results_status`, thresholds in
  `config.py`): a fixture is *overdue* when it is dated more than
  `RESULTS_GRACE_DAYS` (2) before the run date and has no score; a league is
  *stale* when more than `RESULTS_OVERDUE_MAX` (3) of its fixtures are
  overdue. A stalled feed accumulates a whole round within days, while a
  genuinely postponed match stays a lone overdue row. Breaks schedule
  nothing, so they overdue nothing. The block is exported to
  `latest.json` as `results_freshness`, printed at the top of the update
  email (as-of date, a warning per stale league, STALE tags on that
  league's headings) and on the site, and read back by
  `freshness_check.py`, which the workflow runs last to fail the run when
  any league is stale.
- **Feed freshness** (`state.feed_status`): while league fixtures are being
  played (`SEASON_ACTIVE_LOOKBACK_DAYS`), each form feed is reported
  against `FEED_TOLERANCE_IN_SEASON_DAYS` (10) instead of its 130-day
  model guard, so a six-week-old feed can no longer print `fresh: true`
  mid-season. The guard itself is a model parameter and is unchanged.
