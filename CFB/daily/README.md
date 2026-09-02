# CFB/daily — the college-football daily pipeline

Runs once per day (GitHub Actions, `.github/workflows/cfb-daily.yml`) — the
college sibling of `mlb/daily` and `soccer/clubs/daily`: refresh results,
replay the Elo, grade what played, predict what's next, Monte Carlo the
rest of the regular season, publish the site JSON, and render the
twice-weekly update email.

## Module map

| Module | Job |
|---|---|
| `run.py` | Orchestrator CLI — `python -m CFB.daily.run [--date D] [--skip-fetch] [--season-sims N] [--force-email]` |
| `config.py` | Paths, slate window, sim defaults, email weekdays, the frozen always-home baseline |
| `state.py` | One full Elo replay + in-run score-model fits, shared by every step; `as_of()` masks results on/after the run date |
| `scoring.py` | Elo edge → expected margin (linear fit, refit each run) carved out of a matchup-specific total (EWMA points for/against) |
| `predict.py` | Slate for [D, D+2): win probability, pick, expected score; persisted as `slate_{D}.csv` |
| `grade.py` | Grade persisted slates once finals land; running ledger at `data/college_football/predictions/grades.csv` with the paired Δlog-loss vs. always-pick-home |
| `simulate.py` | Vectorized rest-of-season Monte Carlo with live in-sim Elo: expected wins, bowl / undefeated odds, CCG berth, conference title |
| `export_site.py` | `web/public/data/cfb/latest.json` (full FBS board, conference table, slate, ledger, futures, `elo_history`) + per-day snapshots |
| `emails.py` | The update email: Elo top 25, the week's games, rolling tracker + past week, season forecasts |

## Data flow for a run dated D (ET)

1. `CFB.data.fetch_schedule` replaces the current season's rows in
   `games.csv` from cfbfastR-data and fills the trailing week's finals from
   ESPN's scoreboard (best-effort; the committed spine carries a failed
   fetch).
2. Every result dated on or after D is masked, then the tuned Elo is
   replayed over all ~20k games from 2001 and the score model is refit from
   that history. Live, the mask is a no-op (no game dated today is final at
   6 am ET); backdated, it makes `--date` reproduce exactly what that
   morning's run would have produced — no hindsight anywhere.
3. Every persisted slate row whose final has now landed is graded into the
   ledger. Grading is idempotent by game id, so overlapping slate windows
   (a Saturday game is on Friday's and Saturday's slates) and postponements
   resolve cleanly whenever they play.
4. The slate for [D, D+2) is predicted and persisted — a later run's
   grading input.
5. Rest-of-season Monte Carlo (10,000 replays, ~2 s): per program, expected
   wins, P(6+ wins), P(undefeated regular season), P(reach the conference
   championship game) and P(conference title). A scheduled CCG on the spine
   is used as-is; a played one is final; otherwise the top two by
   conference record meet at a neutral site.
6. Site JSON + history snapshot.
7. The update email (`reports/cfb/{D}/update.html`) is rendered every run;
   the manifest marks it sendable only on Mondays and Thursdays
   (`EMAIL_WEEKDAYS`, the same days as the soccer email), the workflow
   delivers it over the shared SMTP secrets, and `data_jobs/email_ledger.py`
   keeps reruns from double-sending.

## Modeling notes

- **Win probability** — straight from the tuned Elo (`CFB/model/elo.py`),
  home edge applied except at neutral sites.
- **Scores** — expected margin is linear in the Elo difference (about 18
  Elo per point, refit each run); the expected total is matchup-specific
  from each program's recent points for/against (half-life 10 games,
  shrunk toward the FBS mean). Reported at one decimal, winner first: an
  average, not a literal final.
- **Grading baseline** — always-pick-home at the frozen 2015-25 rate
  (p = 0.632; coin flip at neutral sites), scored on the same games so the
  ledger's Δlog-loss is a paired difference with a standard error, the way
  the MLB ledger does it. The ledger also carries an FBS-vs-FBS-only row,
  because hit rate on the full slate is inflated by FCS games.
- **What the sim does not do** — the 12-team playoff (a committee pick),
  multi-team conference tiebreakers (random), and bowl-eligibility fine
  print (one FCS win, APR waivers).
