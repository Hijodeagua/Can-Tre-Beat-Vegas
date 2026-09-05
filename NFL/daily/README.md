# NFL/daily — the NFL daily pipeline

Runs once per day (GitHub Actions, `.github/workflows/nfl-daily.yml`) —
the NFL sibling of `CFB/daily`, `mlb/daily` and `soccer/clubs/daily`:
refresh results, replay the Elo, grade what played, predict the next
week, Monte Carlo the rest of the season and the playoff bracket, publish
the site JSON, and render the twice-weekly update email.

## Module map

| Module | Job |
|---|---|
| `run.py` | Orchestrator CLI — `python -m NFL.daily.run [--date D] [--skip-fetch] [--season-sims N] [--force-email]` |
| `config.py` | Paths, email weekdays, sim defaults, the frozen always-home baseline |
| `state.py` | One full Elo replay + in-run score-model fits, shared by every step; `as_of()` masks results on/after the run date |
| `scoring.py` | Elo edge → expected margin (linear fit, refit each run) carved out of a matchup-specific total (EWMA points for/against); the mean margin multiplier for the sim |
| `predict.py` | Slate = the next NFL week's unplayed games: win probability, pick, the model's own line, expected score; persisted as `slate_{D}.csv` |
| `grade.py` | Grade persisted slates once finals land; running ledger at `data/nfl/predictions/grades.csv` with the paired Δlog-loss vs. always-pick-home and a per-week breakdown |
| `simulate.py` | Vectorized rest-of-season Monte Carlo with live in-sim Elo, then the seven-team bracket: expected wins, division, playoffs, #1 seed, conference title, Super Bowl |
| `export_site.py` | `web/public/data/nfl/latest.json` (all 32, divisions, slate, ledger, futures, `elo_history`) + per-day snapshots |
| `emails.py` | The update email: power ratings, this week's games, rolling tracker + past week, season forecasts |

## Data flow for a run dated D (ET)

1. `NFL/model/schedule.py --refresh` re-downloads the nflverse games file
   (best-effort; the committed spine carries a failed fetch).
2. Every result dated on or after D is masked, then the tuned Elo is
   replayed over all ~7,000 games from 1999 and the score model is refit
   from that history. Live, the mask is a no-op (no game dated today is
   final at 6 am ET); backdated, `--date` reproduces exactly what that
   morning's run would have produced — no hindsight anywhere.
3. Every persisted slate row whose final has now landed is graded into
   the ledger. Grading is idempotent by game id and reads slate files
   oldest first, so **a pick locks the first morning its game appears on
   a slate** — a Sunday game is graded on Tuesday's prediction even
   though Saturday's run re-predicted it with fresher ratings.
4. The slate — every unplayed game of the next NFL week — is predicted and
   persisted. A Tuesday run previews the whole week; a Sunday run shows
   what's left of it.
5. Rest-of-season Monte Carlo (10,000 replays, ~1 s): the remaining
   regular season with live in-sim Elo, then seeds 1-7 per conference and
   the bracket (Wild Card at the higher seed, the #1 seed off its bye and
   carrying the rest bonus, Championship at the higher seed, Super Bowl
   neutral). A playoff game already final on the spine is honoured
   whenever the bracket reproduces that matchup.
6. Site JSON + history snapshot.
7. The update email (`reports/nfl/{D}/update.html`) is rendered every run;
   the manifest marks it sendable only on Tuesdays and Thursdays
   (`EMAIL_WEEKDAYS`), the workflow delivers it over the shared SMTP
   secrets, and `data_jobs/email_ledger.py` keeps reruns from
   double-sending.

## Modeling notes

- **Win probability** — straight from the tuned Elo (`NFL/elo/engine.py`),
  home edge applied except at neutral sites, rest edge for a side off its
  bye.
- **The Elo line** — the expected margin from the home side, which is the
  model's own spread. It has never seen a market number, which is the
  whole point of showing it: it is the line to compare against the books,
  not one derived from them.
- **Scores** — expected margin is linear in the Elo difference (about 24
  Elo per point, refit each run); the expected total is matchup-specific
  from each team's recent points for/against (half-life 8 games, shrunk
  toward the league mean). Reported at one decimal, winner first: an
  average, not a literal final.
- **Grading baseline** — always-pick-home at the frozen 2015-25 rate
  (p = 0.55; coin flip at neutral sites), scored on the same games so the
  ledger's Δlog-loss is a paired difference with a standard error. A tie
  scores both halves and counts as a wrong pick.
- **What the sim does not do** — the real tiebreaker ladder (head-to-head,
  common games, strength of victory; it uses division / conference record
  then random), simulated ties, and margins (ratings move by K times the
  fitted mean margin multiplier).
