# Model sources — where every feature comes from, and what it's worth

One table per model: the feature, the file it lives in, the upstream it
came from, the fetcher that pulls it, and its measured worth. Built for
going model by model and finding better inputs.

Worth = increase in log loss when the input is removed, on seasons the
model never saw. Bigger is more load-bearing. Method differs by model
(permutation for the fitted soccer logistic, component ablation for the
Elo engines), so compare within a table, not across.

Regenerate the worth column: `python -m data_jobs.build_importance`.
Feature definitions and the candidate list: [MODEL_FEATURES.md](MODEL_FEATURES.md).

---

## Club soccer

Outcome model: multinomial logistic, {home win, draw, away win}.
Baseline log loss **1.01755** on 7,827 holdout matches.

| Feature | Local file | Upstream | Fetcher | Worth |
|---|---|---|---|---|
| `elo_gap` | derived from `results.csv` + `uefa_results.csv` | openfootball (football.json + country repos, public domain); openfootball/champions-league | `fetch_results.py`, `fetch_uefa.py` | **+0.048** |
| `value_diff_z` | `soccer/clubs/data/market_values/values_*.csv` | Transfermarkt, **hand-uploaded** | none — manual | **+0.011** |
| `sot_net_diff` | `soccer/clubs/data/shots_matches.csv` | football-data.co.uk (live), datasets/football-datasets GitHub mirror (fallback) | `fetch_shots.py` | **+0.006** |
| `xg_net_diff` | `soccer/clubs/data/xg_matches.csv` (legacy shape of `understat_matches.csv`) | understat.com `getLeagueData/{league}/{season}` JSON, legacy HTML blob fallback | `understat.py` (`fetch_xg.py` is a shim) | +0.0001 on all rows; **+1.78 SE where both clubs had live form** — feed dead since 2025-01-04, fetcher replaced, first live refresh pending from Actions |
| `spend_diff_z` | `soccer/clubs/data/club_season_transfers.csv` | ewenme/transfers (Transfermarkt fees) | `fetch_transfers.py` | **0.000 — upstream ends 2022-23** |
| `net_diff_z` | same | same | same | **0.000 — same** |
| `wage_diff_z` | `market_values/values_*.csv` (optional column) | Transfermarkt, hand-uploaded | none — manual | **0.000 — column never supplied** |

Advanced Understat metrics (npxG, xPts, PPDA, deep completions), rest and
congestion are collected and evaluated but not promoted — see
[ADVANCED_METRICS.md](ADVANCED_METRICS.md).

Ratings, scorelines and MLS:

| Piece | Local file | Upstream | Fetcher |
|---|---|---|---|
| Elo replay (10 pools) | `results.csv` | openfootball | `fetch_results.py` |
| Cross-league glue (0.75 weight) | `uefa_results.csv` | openfootball/champions-league — CL from 2014-15, EL 2020-21, Conference 2021-22 | `fetch_uefa.py` |
| MLS ratings | `results.csv` (MLS rows) | philo92/mls-elo, CC-BY 4.0, 1996→ | `fetch_mls.py` |
| Poisson goal rates | derived, last 2 completed seasons | — | refit every run |

Training windows:

- **Elo per pool** — one-step-ahead Brier, first two seasons of each pool are burn-in, everything from **2024-25** held out. Last tuned **2026-08-22**; params committed 2026-09-10.
- **Outcome model** — trained on seasons **< 2024-25**, scored on 2024-25 onward. Refit by hand (`python -m soccer.clubs.model.train`), not by the daily job; artifact last committed **2026-09-12**.
- **Elo ratings themselves** — replayed from scratch (58,908 matches) every daily run, so there is no rating state on disk to drift.

Blockers worth knowing before you go shopping: understat.com and
football-data.co.uk are both blocked by the dev sandbox proxy, so those
fetchers only actually run from the GitHub Actions job. Any new soccer
feed needs to be reachable from Actions, and ideally have a
raw.githubusercontent mirror for local work.

---

## NFL

Elo only — no fitted layer, no market data. Baseline log loss **0.62424**
on 586 holdout games (always-pick-home is 0.691).

| Component | Local file | Upstream | Fetcher | Worth |
|---|---|---|---|---|
| Margin of victory (ln-damped, cap 45) | `data/schedules/nflverse_games.csv` | nflverse | `NFL/model/schedule.py` | **+0.025** |
| Season regression (40% → 1500) | same | same | same | **+0.019** |
| Home advantage (+48, 0 at neutral) | same (`location`) | same | same | +0.002 |
| Postseason K multiplier | same (`game_type`) | same | same | 0.000 — tuned to 1.0, already off |
| Rest / bye bonus (+20 at 10+ days) | same (`home_rest`, `away_rest`) | same | same | **−0.0006 — better without it** |
| Expected margin + total | derived from the replay | — | refit every run | not ablated |

Training windows:

- **Elo** — coordinate descent on one-step-ahead log loss, seasons **2005–2023**, holdout **2024–25**, 1999–2004 burn-in. Last tuned **2026-09-05**. Holdout 0.624 vs 0.691 always-home.
- **Score model** — refit from the engine's own replay on every daily run (last 10 seasons), so nothing is pickled.
- **`NFL/model/v2/` (research only, not on the board)** — 45 features, LightGBM, walk-forward one retrain per season 2015–2025. Sees the closing line, which is why it can't feed a betting-blind board.

---

## College football

Elo only. Baseline log loss **0.48462** on 2,038 holdout games.

| Component | Local file | Upstream | Fetcher | Worth |
|---|---|---|---|---|
| Pooled FCS opponent rating (950) | `data/college_football/games.csv` (`home_division`) | cfbfastR-data (sportsdataverse, ESPN-derived), raw.githubusercontent, no key | `CFB/data/fetch_schedule.py` | **+0.038** |
| Margin of victory (cap 80) | same | same | same | **+0.037** |
| Season regression (30%) | same | same | same | **+0.028** |
| Conference regression (0.75 blend) | same (`home_conference`, per season) | same | same | **+0.011** |
| Home advantage (+50) | same (`neutral_site`) | same | same | **+0.011** |
| FBS entry rating (1250) | same | same | same | +0.001 |
| Expected margin + total | derived from the replay | — | refit every run | not ablated |

Training windows:

- **Elo** — coordinate descent on one-step-ahead log loss, seasons **2005–2023**, holdout **2024–25**. Last tuned **2026-09-02**.
- Upstream gaps already known: bowls and the CFP only exist from 2024, neutral-site flag blank for 2001–02.

The FCS row is the headline. One synthetic 950-rated team stands in for
every non-FBS opponent — about 13% of the schedule — and it is the most
load-bearing component in the model, ahead of home advantage. cfbFastR
carries the actual FCS opponent, so rating them individually or in tiers
is available today.

---

## MLB

Elo plus three adjustments applied by the daily pipeline. Baseline log
loss **0.67965** on 34,665 games since 2012; always-pick-home is 0.691, so
the engine buys ~0.013 nats. Baseball is close to a coin flip at the game
level and the model says so.

| Component | Local file | Upstream | Fetcher | Worth |
|---|---|---|---|---|
| Season carryover (0.6) | `data/mlb/games_2009_2026.csv` | Retrosheet game logs 2009–2025 + MLB statsapi for 2026 | `mlb/build_games.py` | **+0.003** |
| Home advantage (+24) | same | same | same | **+0.002** |
| Margin of victory | same | same | same | **+0.001** |
| Starting pitcher (3.0 Elo per rGS point) | `data/mlb/pitcher_starts.csv` | retrosplits 2010–2025 + MLB statsapi boxscores | `mlb/build_starts.py`, `mlb/build_starts_statsapi.py` | **not measured** — applied in the daily path, not the replay |
| Rest (+2.3/day, cap 3) | derived from game dates | — | — | **not measured** — same reason |
| Travel (−0.31·mi^⅓, cap −4) | static park-coordinate table in `mlb/adjustments.py` | hand-seeded from statsapi venues | — | **not measured** — same reason |
| Run-environment rates | derived from the replay | — | refit every run | not ablated |

Training windows:

- **Elo** — grid search on log loss, evaluated **2012 onward** (2009–2011 burn-in). Tuned k=3, home +24, carryover 0.6, MOV on: 0.67961 vs 0.69096 baseline. Params committed **2026-09-10**.
- **Starting-pitcher knobs** (C=3.0, half-life 20 starts) — tuned on **2012–2021 only**, out-of-sample tested in `research/SP-BACKTEST.md`. 538's published C=4.7/half-life 10 was not distinguishable out of sample; this pair was.
- **Active model version** `v2-sp`, cut over 2026-08-15. v1's ledger is frozen and still visible on the model card.

First job here is closing the measurement gap: three of the six inputs
aren't in the ablation at all, because the replay doesn't apply them.

---

## NBA — nothing exists yet

There is no NBA model. What's in the repo today:

| Asset | Local file | Upstream | Fetcher |
|---|---|---|---|
| Odds snapshots | `data/nba/nba_odds_api_data_*.csv` | The Odds API (free tier), h2h/spreads/totals | `data_jobs/nbaodds.py`, `data_jobs/odds_api/` |
| Results (optional, manual) | `data/nba/actual_games/*.csv` | Basketball Reference, hand-dropped | none |

To stand up an NBA board on the same pattern as the other four, in order:

1. **A game spine.** One row per game: date, teams, scores, neutral flag,
   rest days. The nflverse-equivalent is **nba_api** (the official stats
   endpoints, Python client, no key) or **Basketball Reference** monthly
   pages, which `data/nba/actual_games/` is already shaped for. A spine
   back to ~2010 is enough to tune on and hold out recent seasons.
2. **An Elo engine.** Copy `CFB/model/elo.py`'s shape: logistic
   expectation, home advantage in Elo points, ln-damped MOV with a cap,
   season regression. The NBA specifics worth tuning from day one are
   **rest / back-to-backs** (a bigger effect than any other league here)
   and **travel**, both computable from the spine alone — the MLB
   `adjustments.py` formulas port directly.
3. **A daily pipeline.** `predict.py` / `grade.py` / `simulate.py` /
   `export_site.py` are the same four files in every sport; the CFB set is
   the closest template since it has no playoff modelling to copy.
4. **Then** the NBA-specific inputs, in rough order of expected value:
   rest and schedule density (free), **player availability** — the single
   biggest signal in basketball and the one the other four sports don't
   have an equivalent for — then possession-level efficiency (offensive
   and defensive rating per 100), which nba_api serves directly.

Availability is the one to think about early: injury/rest reports are
published as free text and change hours before tip-off, so the fetcher is
the hard part, not the model. It is also why an NBA Elo with no roster
input will look worse than the NFL one no matter how well it's tuned.

---

## Plain explainer

Copy-paste for the site, a README, or anywhere the numbers need a
sentence. No hedging, no metric jargon.

> **How this works.** Every team carries one number, its Elo rating. Beat
> a team rated above you and your number goes up; lose to a team below you
> and it drops further than a normal loss would. The size of the move
> depends on the margin and on how surprising the result was. Ratings are
> rebuilt from every game in the archive each morning, so nothing drifts.
>
> **How a pick is made.** The gap between two ratings converts directly
> into a win probability, adjusted for home field and rest. In soccer, a
> fitted model reads that gap alongside squad value and recent shooting
> form. No model ever sees a betting line — the point is to compare
> against the market, not to copy it.
>
> **How a forecast is made.** The rest of the season is replayed thousands
> of times, sampling each remaining game from the ratings and updating
> those ratings inside every run. Counting how often each team wins the
> title, makes the playoffs or goes down gives the percentages on the
> table.
>
> **Why the projected ratings wander.** Each line on the projection chart
> is one simulated season, not an average of them. Averaging is
> meaningless here: a fair game is equally likely to move a rating up or
> down, so the average of thousands of seasons is a flat line at today's
> rating. The individual seasons cross each other constantly, which is
> what a forecast actually looks like.
>
> **How good is it.** Every pick is graded against the result and against
> a naive "always pick the home team" baseline. Log loss is the score:
> lower is better, 0.693 is a coin flip. The NFL model runs 0.624 against
> a 0.691 baseline. The MLB model runs 0.680 against 0.691 — baseball is
> nearly a coin flip at the game level and no model changes that.
