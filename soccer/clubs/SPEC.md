# Club Elo Models — Top-5 European Leagues + MLS

Per-country Elo models covering each of the top-5 European leagues **and
its second division** (EPL + Championship, Bundesliga + 2. Bundesliga, La
Liga + Segunda, Serie A + Serie B, Ligue 1 + Ligue 2), plus MLS on its own
unglued pool — the club-football sibling of the international model in
`soccer/model/`. Same DNA (logistic expectation, margin-of-victory
multiplier, draws as 0.5, a multinomial outcome layer on the Elo gap), with
the structure club league play actually has: promotion/relegation inside
one country pool, and summer squad churn.

## Elo rules

- **One pool per country.** The top flight and its second division share an
  Elo pool, so promotion and relegation are just clubs changing which
  fixtures they play: a relegated club keeps playing rated matches, a
  promoted club arrives carrying its actual second-division form. Pools
  stay closed across countries (an EPL 1600 and a Ligue 1 1600 are not
  claims about each other) except through the UEFA glue below.
- **Fresh start at each pool's first upstream season** (see data table
  below).
- **Season rollover regression** — at every season boundary all known
  ratings regress toward 1500 by a tuned fraction ρ: squads churn over a
  summer, and last May's rating overstates what returns in August.
- **Division-switch carry** — promotion selects overperformers, so a
  promoted club's carried D2 rating overstates it (winner's curse; full
  carry measurably *hurt* the top-flight holdout). On a club's first match
  after switching tiers its rating is blended toward the new tier's entry
  level: `r ← entry + carry × (r − entry)`, with carry tuned per pool.
  carry = 0 recovers a flat entry rating, carry = 1 is full carry; the
  tuned values run from 0.25 (EPL — Championship form transfers least) to
  1.0 (Ligue 1).
- **Entry ratings** — a club never seen before enters at a tuned rating:
  `entry_rating` for a first top-flight appearance, the lower
  `entry_rating_t2` for a club coming up into the second division from the
  third tier (also the blend anchor for its division).
- **Margin-of-victory multiplier** — eloratings.net convention: ×1 for
  1-goal wins, ×1.5 for 2, ×(11+N)/8 for N ≥ 3.
- **Home advantage** — tuned per pool, shared by both divisions, added to
  the home side inside the expectation. League files carry no neutral-site
  flag; the rare neutral match (a Championship playoff final) is absorbed
  as noise.
- **Draws** count as 0.5.

## Tuned parameters

`model/tune.py` grid-searches (K, home advantage, ρ, both entry ratings,
division carry) per pool over a two-division replay, minimizing the
**one-step-ahead Brier score on top-flight matches** — the second division
is replayed (that's where promoted clubs' ratings come from) but not
scored. The first two seasons of each pool are burn-in and 2024-25 onward
is excluded — that stays the untouched holdout for `train.py`. Current
values live in `model/artifacts/tuned_params.json`; expect K ≈ 10–14, home
advantage ≈ 45–75 Elo, ρ ≈ 0.05–0.15 (lower than the single-division
version needed — promotion/relegation is now real matches, not blanket
shrinkage).

Adding the second divisions was validated head-to-head on the untouched
holdout (identical Elo-gap-only outcome layers): pooled log loss
**0.98986 vs 0.99040** single-division, improving both promoted-club
matches (0.96289 vs 0.96337, n=876) and everything else (0.99885 vs
0.99941). Without the tuned carry — i.e. carrying D2 ratings up unshrunk —
the pooled model was *worse* (0.99203); the winner's-curse correction is
what makes the D2 data pay.

## UEFA cross-league glue

The five pools are closed economies — except in Europe. `model/europe.py`
replays the leagues *and* the UEFA club competitions (Champions League from
2014-15, Europa League from 2020-21, Conference League from 2021-22; data
in `data/uefa_results.csv`) in one chronological stream. A UEFA match
between two tracked clubs exchanges rating points between their league
pools zero-sum, with K = the mean of the two leagues' tuned Ks ×
`UEFA_WEIGHT`, the home club's league home advantage (dropped for
neutral-venue finals), and the usual MOV multiplier. Matches against clubs
outside the five leagues (Porto, Ajax, …) are skipped — no rating exists
for the opponent. MLS never enters this glue at all: it's a different
confederation (CONCACAF, not UEFA) and never plays these competitions, so
its pool stays fully independent — its Elo numbers are not comparable to
the ten glued leagues', by construction, not by omission.

That's ~65 cross-league matches a season against ~1,750 league matches, so
the effect is modest by construction, but it is the only competitive signal
linking the pools: with it, cross-league Elo comparisons mean something.
Validated on the league holdout: log loss 0.99058 glued (weight 0.75,
interior optimum) vs 0.99094 unglued.

## Squad economics (Transfermarkt layer)

`data/fetch_transfers.py` pulls Transfermarkt transfer fees from
[ewenme/transfers](https://github.com/ewenme/transfers) and aggregates them
to club-season gross spend / sales / net (`data/club_season_transfers.csv`,
~1,100 club-seasons; upstream fees currently run through 2022-23 and the
aggregate extends whenever upstream resumes). `model/features.py` turns
these into home-minus-away differentials z-scored within league-season,
plus `value_diff_z` / `wage_diff_z` from the optional squad-value uploads
(`data/market_values/`, schema in its README — transfermarkt.com itself is
proxy-blocked in the hosted dev environment, so values are populated
locally). Everything 0-imputes when missing: the model degrades to
Elo-only.

Honest holdout read (2021-22 + 2022-23, the last transfer-covered seasons):
net spend improves log loss 0.99081 → 0.99064 — small but directionally
sane (spend → home wins). Not a rating replacement; carried as features.

## xG form (Understat layer)

`data/xg_matches.csv` holds per-match xG for both sides — the five top
flights back to 2014-15, backfilled from the archived worldfootballR_data
Understat mirror and refreshed by `data/fetch_xg.py` (understat.com's
embedded `datesData` JSON, one request per league-season; runs only from
the daily Actions job — the dev sandbox proxy blocks the host).
`model/xg.py` turns it into one feature, `xg_net_diff`: each side's
rolling mean xG-net (for − against) over its last 10 league matches,
differenced, strictly pre-match, with a 130-day staleness guard (stale
form is worse than none) and 0 wherever Understat has no coverage
(second divisions, MLS, pre-2014).

Validated on the 2023-24 holdout (the last fully-backfilled season):
logistic log loss 0.9662 → 0.9620, +2.1 SE paired — the first form-style
feature to survive here (results-form and rest days were rejected long
ago; xG form carries chance-creation signal that neither Elo nor the
table has). In the shipped model it lands as the strongest non-Elo
coefficient (±0.11 vs squad value's ±0.09).

**Current status: dormant, fetcher replaced.** `xg_matches.csv` has not
moved past 2025-01-04, so the staleness guard zeroes this feature on
every live slate. `data/understat.py` now pulls the
`getLeagueData/{league}/{season}` JSON endpoint (legacy HTML blob as
fallback) and writes both the legacy file and the wider
`understat_matches.csv` (npxG, xPts, PPDA, deep completions); the daily
job probes the endpoint and runs the fetch. Where both clubs had live
form on the 2024-25+ holdout the feature is worth +1.78 SE, so it comes
back the day the feed does. The wider metrics are collected and evaluated
(`model/advanced.py`, `model/eval_advanced.py`) but not promoted — see
`docs/ADVANCED_METRICS.md`.

## Shot form (football-data.co.uk layer)

`data/shots_matches.csv` holds shots and shots on target for both sides
of every top-5-league match back to each league's first season in
results.csv (25,468 matches). `data/fetch_shots.py` refreshes it from
football-data.co.uk, falling back to the public-domain
`datasets/football-datasets` GitHub mirror — the mirror lags a season but
is reachable from the dev sandbox, so `--source mirror` is what produced
the committed backfill and what a developer can rerun locally.

`model/shots.py` turns it into one feature, `sot_net_diff`: each side's
rolling mean shots-on-target net (on target for − on target against) over
its last 10 league matches, differenced, strictly pre-match, with the same
130-day staleness guard and the same 0-impute where there is no coverage.
The window/warm-up/staleness/attach machinery is shared with the xG layer
in `model/form.py`.

Validated on the 2024-25 + 2025-26 holdout, added on top of the *full*
existing feature set: log loss 1.0199 → 1.0180, +2.9 SE paired, and it
improves each of the three season splits tried (2023-24, 2024-25,
2025-26). Total shots validate too (+2.8 SE) but add nothing on top of
shots on target, so only the on-target feature ships.

Two things it buys over the xG feature beside it: coverage (41.5% of
training rows carry shot form against xG's 32.2%, and 57% of a live
slate against xG's 0%), and liveness. On the same holdout, Elo +
economics + shot form (1.0181) beats Elo + economics + xG form (1.0199).
They are complements, not substitutes — both stay in the model, and a
revived Understat feed makes the pair stronger.

Coverage is top-5 flights only. football-data.co.uk publishes the second
divisions too, but the mirror does not, so their name mapping can't be
derived or verified from the sandbox; those leagues 0-impute exactly as
they already do for xG.

## Probability model

Multinomial logistic regression over {home win, draw, away win} on the
venue-adjusted Elo gap plus the squad-economics differentials, pooled
across all ten divisions — the gap→probability curve is shared (training
on both tiers beats tier-1-only *for* tier-1), while each match's gap
already comes from its own tuned, UEFA-glued country pool. Temporal
validation on the two held-out seasons (2024-25 + 2025-26, never seen by
tuning or training):

| | log loss |
|---|---|
| Top-flight holdout (4,602 matches) | **0.9928** |
| Second-division holdout (3,304 matches) | 1.0422 |
| All divisions (7,906 matches) | 1.0134 |
| Elo-gap only | 1.0260 |
| class-frequency baseline | 1.0763 |

Second divisions are genuinely harder to predict — flatter, draw-heavier —
which the numbers say plainly. Per-league top-flight holdout log loss runs
0.967 (La Liga, Serie A) to 1.008 (EPL), beating the frequency baseline
everywhere. Artifacts: `model/artifacts/outcome_model.pkl`, `metrics.csv`.

### Levels as well as differences

Every feature is a home-minus-away difference, and every one of them now
*also* enters as the two numbers it was made from (`keep_sides=True` on
both attach chains; `adv.ALL_ADVANCED_SIDES`, `features.SIDE_FEATURES`,
`features.SIDE_RAW_FEATURES`). A difference imposes
f(home, away) = f(home − away) — it tells the model that 1.9 against 1.5
and 0.7 against 0.3 are the same match, when two strong attacks produce a
different game from two weak ones. The Elo columns had carried their own
levels for exactly this reason; this extends it to the rest.

Measured on the 2024-25+ holdout with the production forest, before
committing to it:

| Feature set | n | log loss |
|---|---:|---:|
| differences only | 44 | 1.01441 |
| per-side levels only | 77 | 1.01484 (0.4 SE — noise) |
| **both** | 102 | **1.01337** (1.7 SE better) |

So the levels are not a breakthrough: the differences were already
carrying almost all of it, and dropping them entirely would cost nothing
measurable either. Keeping both is the best measured configuration, it is
free, and it makes the model and the published match card agree on what
was used.

A caution learned the hard way while measuring this: the per-side column
lists are **enumerated**, never collected by scanning for a
`home_`/`away_` prefix. The replay history carries `home_score` and
`away_score` beside the features, and a prefix scan sweeps the match
result into the model — which shows up as a holdout log loss of 0.39 at
95% accuracy, a number no football model can reach and therefore a
leak. `tests/test_slate_sides.py` asserts the lists cannot drift into it.

### Squad value: the z *and* the euros

`value_z` is the club's squad value as a z-score within its own
league-season, and it stays the primary feature — it is the only fair
comparison, since a £900m Premier League squad and a €900m Bundesliga
squad buy very different league positions. But the z destroys absolute
scale by construction: the richest club in every league scores about the
same z whether it is Manchester City or Feyenoord. So the raw
`squad_value_eur_m` now rides along per side as well (on one recent slate
that is PSG at €1,480m against Bolton at €24.65m — a distinction the z
cannot express), and with the league one-hots and `season_idx` in the
feature set the learner can use whichever it needs.

The raw euro figures are carried per side only and never differenced.
The z differentials fill a missing side with 0.0, which for a z means
"assume league-average"; the same fill on a euro figure would mean
"assume this club is worth nothing", and the resulting difference would
be the other club's entire squad value masquerading as a gap.

Permutation importance over the 102-feature model says the levels are
genuinely being used, which the aggregate holdout number was too blunt
to show:

| Rank | Feature | Δ log loss |
|---:|---|---:|
| 1 | `elo_gap` | +0.00948 |
| 2 | `value_diff_z` | +0.00681 |
| **3** | **`home_value_z`** | **+0.00199** |
| 4 | `elo_away_pre` | +0.00158 |
| 5 | `elo_home_pre` | +0.00144 |
| 6 | `xpts_ewm_diff` | +0.00083 |
| **7** | **`away_squad_value_eur_m`** | **+0.00079** |
| 8 | `away_value_z` | +0.00077 |

The differences still lead — `elo_gap` is far and away the most
important single column — but a per-side level ranks third, above both
raw Elo columns, and the raw euro figure outranks the z it was computed
from. Nine of the 102 columns are constant (the per-side wage and
transfer-spend z-scores, whose feeds are empty) and contribute nothing.
Artifact: `model/artifacts/importance.json`, rebuilt with
`python -m data_jobs.build_importance --only soccer`.

Features tested and *rejected* (they made holdout log loss worse): last-5
form differential, rest-day differential — Elo already carries that
information.

## Daily pipeline

`daily/` runs the whole thing once a day (see `daily/README.md`): refresh
results + UEFA, rebuild the glued Elo, grade persisted slates into a
running ledger, predict the next two days' fixtures (W/D/L, pick, most
likely Poisson scoreline), Monte Carlo each league's remaining season
(title / top-4 / relegation with live in-sim Elo), and publish
`web/public/data/soccer/latest.json`. Fixtures come from the openfootball
country repos, which publish new seasons before football.json does —
that's what makes the runner live in 2026-27 today.

## Data

Source: [openfootball/football.json](https://github.com/openfootball/football.json)
(public domain), one JSON per league-season, refreshed by
`data/fetch_results.py` into the committed `data/results.csv` (canonical
club names, played matches plus any current-season fixtures the upstream
has published).

| League | key | upstream code | seasons |
|---|---|---|---|
| Premier League | `epl` | `en.1` | 2010-11 → |
| Championship | `championship` | `en.2` | 2010-11 → |
| Bundesliga | `bundesliga` | `de.1` | 2010-11 → |
| 2. Bundesliga | `bundesliga_2` | `de.2` | 2012-13 → |
| La Liga | `la_liga` | `es.1` | 2012-13 → |
| Segunda División | `la_liga_2` | `es.2` | 2012-13 → |
| Serie A | `serie_a` | `it.1` | 2013-14 → |
| Serie B | `serie_b` | `it.2` | 2013-14 → |
| Ligue 1 | `ligue_1` | `fr.1` | 2014-15 → |
| Ligue 2 | `ligue_2` | `fr.2` | 2014-15 → (hole 2021-24) |
| MLS | `mls` | — (own fetcher) | 2013 → |

MLS is not from openfootball — `data/fetch_mls.py` pulls the whole match
history from [philo92/mls-elo](https://github.com/philo92/mls-elo) (one
CSV, 1996 → present, already unified to each club's current name; we start
replay at 2013 as a deliberate quality cutoff). Calendar-year seasons
("2020", not "2020-21") since an MLS season never crosses New Year's;
`leagues.next_season`/`current_season_for` handle both formats. The source
is a completed-match log with no upcoming fixtures, so MLS has no daily
slate and falls out of the European futures Monte Carlo — but it does get
its own forecast, off a schedule reconstructed from the league's format
rather than read from a fixture list (see **MLS forecast** below).
Because `fetch_results.py` owns `results.csv` outright
and rewrites it from the openfootball leagues alone, `fetch_mls.py` always
runs *after* it in `daily/run.py`'s refresh step, merging in rather than
overwriting.

### Publishing both halves of a differential

Every feature the outcome model sees is a home-minus-away difference —
Elo gap, squad economics, xG and shots form, the whole advanced layer.
That is the right shape for training and the wrong shape for reading: a
published "+0.4 xG created" is a different match when it is 1.9 against
1.5 than when it is 0.7 against 0.3, and the differential cannot tell
them apart.

So each of the three feature layers grew a `keep_sides` switch
(`model/features.py`, `model/advanced.py`, and the per-club `net()` the
rolling-form layer already had). It is **off by default**: the training
path calls them exactly as it always did and its frame is unchanged,
while `daily/state.py`'s `outcome_probs` turns it on for the slate. The
learner is handed exactly `FEATURES` in both cases, so the prediction is
bit-identical; the slate frame simply carries more columns alongside it.

`daily/predict.py` owns the catalogue (`SIDE_METRICS`): for each of ~33
per-side numbers, its two source columns, a label, a group and whether
up is good — which the site reads from the published
`slate_metrics` block rather than keeping a second copy that can drift.
Two rules keep it honest: a metric with no reading for a side is
**absent, not zero** (a club short of the rolling window's warm-up has
not been measured, and 0 would claim it is average), and `wage_z` is
withheld entirely while no source fills it, since it is 0.0 on all 2,207
rows and would otherwise read as "average wage bill".

The persisted `slate_{D}.csv` deliberately keeps its original columns:
`grade.py` reads it back and every past prediction lives in it, so the
per-side numbers ride the in-memory frame to the site JSON instead.

`daily/trends.py` is the other half — the weekly email's answer to why a
slate looks the way it does. Every club playing this week is measured
against **its own** mean in completed prior seasons rather than a league
average (a league baseline only ever reports that good clubs are good),
and the gaps are scaled by how much clubs differ on that metric so an xG
swing and a PPDA swing can be ranked against each other.

### MLS forecast

MLS is the one league whose remaining fixtures have to be *derived* rather
than read. Its upstream is an Elo-history log of played matches, so there
are no unplayed rows for `daily/simulate.py` to replay — and for a long
time that meant the league had ratings but no forecast at all.

`data/mls.py` reconstructs the run-in from the format instead. MLS is 30
clubs, 15 per conference, no promotion or relegation, and 34 matches each:
28 against conference rivals (every rival once home and once away) and 6
cross-conference, 3 at home and 3 away against 6 different opponents. The
first half of that is an exact reconstruction — an intra-conference
fixture is still owed if and only if that ordered (home, away) pair has
not been played. The second half is not: which six cross-conference
opponents a club draws is a scheduling decision the results log does not
reveal, so only each club's remaining count of them, split home and away,
is recoverable.

So the two halves are treated differently, and the difference is the
point. The conference fixtures go into every simulation unchanged; the
cross-conference ones are drawn fresh in each simulation from the pairings
consistent with the quotas, which puts the genuine uncertainty about those
~25 matches into the spread of the odds rather than into one invented
schedule every run shares. `mls.verify_structure()` re-checks the whole
reconstruction each run — every club must land on exactly 34 matches and
the two conferences' cross-conference quotas must clear against each
other — and the pipeline publishes nothing rather than publishing odds
built on a schedule that cannot happen. It correctly refuses 2023 and
2024, which were 29-club seasons with a different shape.

`model/mls_forecast.py` then runs the same machinery as every other
league — live in-sim Elo with the MLS pool's tuned K and home advantage,
scorelines from the independent-Poisson grid — and carries each simulated
season through the playoff bracket: 9 qualifiers per conference, 8 hosts 9
in a single Wild Card match, Round One is a best-of-3 with the higher seed
hosting games 1 and 3, and every round after that is one match at the
higher seed's ground. A drawn playoff match goes to a shootout, which the
sim treats as a coin flip — the home side's real edge is already in the
90 minutes through `home_advantage`, and giving it a second one in the
shootout would be double-counting.

It also publishes the two things the site draws: a per-conference Elo
chart and the playoff bracket.

The chart's x axis is **matches played, not dates**. That started as the
only honest option — the remaining fixtures have no dates, and estimating
them from the season's own cadence is not good enough to plot against
(run on 2025 at the three-quarter mark, that estimate puts the final
matchday on 12 September against an actual 18 October, five weeks out) —
but it is also the better axis for MLS: clubs sit up to three games
apart, so a date axis shows a club level with one that has already spent
its games in hand, and this one shows it behind with matches to come.
History is one point per match played, closing on the live rating;
the projection continues from that same point out to 34, as a median
simulated season with a 10th/90th band and a few whole sample seasons —
the same three readings as the European chart, and with the mean
omitted for the same reason (a fair game's expected Elo change is zero,
so averaging the sims returns today's rating for everyone).

The bracket is drawn from the same sims: per conference, how often each
club lands on each of seeds 1-9, plus the MLS Cup matchups that come up
most often. The site fills the slots by **expected finishing position**
rather than by each slot's most likely occupant, which matters more than
it sounds: slot modes are marginals, and taken independently they put one
club top of two different slots and leave another out of the bracket
entirely — true of each marginal, nonsense as a bracket. Expected finish
gives every club exactly one slot, and each slot still carries how often
that club really finishes there (rarely above a third — a seed is a fine
distinction) with the runners-up named underneath.

One thing is deliberately missing: the tiebreakers below goals for
(disciplinary points, away/home goal splits) are not modeled, so clubs
still level on points, wins, goal difference and goals for are separated
at random.

`model/backtest_mls.py` scores the whole thing walk-forward against a
completed season. On 2025 at three cutoffs (35%, 50%, 70% of the season
played), expected-points MAE tightens 5.96 → 4.81 → 3.12 with bias never
worse than +0.32, and playoff-qualification Brier runs 0.09 / 0.11 / 0.04
against the 0.240 you get from quoting the 18-in-30 base rate at
everybody. The honest caveat is in the module and repeated here: 2025 is
the only completed 30-club season, so those are three correlated views of
one season, not three seasons. Enough to catch a broken model, not enough
to call it calibrated — and the Shield and Cup lines in particular are
single anecdotes.

Wrinkles handled in the fetch, so nothing downstream sees them:

- **Name drift.** The upstream renamed most clubs to long legal names midway
  through its history (2020-21 wave for en/de/es/it, 2023-24 for fr) —
  "Manchester City" vs "Manchester City FC" would split one club's history
  in two. `data/leagues.py` maps every historical spelling to the current
  canonical name.
- **Score shapes.** Finals normally arrive as `score.ft = [h, a]`; the
  newest season files serialize 0-0 finals as a bare `score = [0, 0]`. Both
  are accepted.
- **Dual-source seasons.** Both layers are fetched per season and the one
  with more played matches wins — the json layer never got Championship
  2016-18 and stalled mid-season on some recent D2 files, while the
  country txt repos fill the Segunda/Serie B 2021-24 json gaps.
- **Known holes.** Ligue 1 2019-20 stops at the COVID abandonment (279
  matches); one cancelled Ligue 1 2025-26 match. Second divisions carry a
  few upstream warts: Ligue 2 is missing 2021-22 → 2023-24 entirely (no
  reachable source), and 2. Bundesliga / Segunda / Serie B 2025-26 stall
  partway (99/131/309 matches) — the season rollover regression absorbs
  the staleness. Scoreless past-season rows are dropped rather than
  guessed.
- **New seasons.** Every fetch probes one season past the current one, so
  new season files start flowing in as soon as openfootball publishes them
  — no code change. (2026-27 fixtures are live for all four top flights
  except Ligue 1, plus the Championship.)

## Pipeline

```
soccer/clubs/
├── SPEC.md                  # this file
├── data/
│   ├── leagues.py           # league registry + canonical-name aliases (+ UEFA aliases)
│   ├── mls.py               # MLS conferences, season shape, remaining-schedule rebuild
│   ├── football_txt.py      # parser for the openfootball Football.TXT format
│   ├── fetch_results.py     # football.json + country-repo txt → results.csv
│   ├── fetch_uefa.py        # champions-league repo → uefa_results.csv
│   ├── fetch_mls.py         # philo92/mls-elo → results.csv (merges "mls" rows only)
│   ├── fetch_xg.py          # understat.com → xg_matches.csv (Actions-only; merge posture)
│   ├── fetch_shots.py       # football-data.co.uk (+ GitHub mirror) → shots_matches.csv
│   ├── fetch_transfers.py   # ewenme/transfers → club_season_transfers.csv
│   ├── market_values/       # optional squad value / wage uploads (see README)
│   ├── results.csv          # committed league results + current-season fixtures
│   ├── xg_matches.csv       # committed per-match xG (top-5 flights, 2014-15 →)
│   ├── shots_matches.csv    # committed per-match shots + shots on target (top-5)
│   ├── uefa_results.csv     # committed UCL/UEL/UECL results, league-mapped
│   └── club_season_transfers.csv
├── model/
│   ├── elo.py               # ClubEloEngine (per-league pools, rollover, entry rating)
│   ├── europe.py            # UEFA cross-league glue replay
│   ├── form.py              # shared rolling-form machinery (window, warm-up, staleness)
│   ├── xg.py                # rolling xG-form feature (xg_net_diff)
│   ├── shots.py             # rolling shot-form feature (sot_net_diff)
│   ├── features.py          # spend / value / wage differentials (z within league-season)
│   ├── mls_forecast.py      # MLS Shield / conference seeding / MLS Cup bracket sim
│   ├── backtest_mls.py      # walk-forward scoring of that forecast
│   ├── tune.py              # per-league parameter grid search
│   ├── train.py             # pooled multinomial outcome model + temporal validation
│   ├── export_ratings.py    # → artifacts/club_elo_ratings.json (glued, all leagues)
│   └── artifacts/           # tuned_params.json, outcome model, metrics, ratings JSON
└── daily/                   # the daily runner (see daily/README.md)
```

```bash
python -m soccer.clubs.data.fetch_results     # refresh league results + fixtures
python -m soccer.clubs.data.fetch_uefa        # refresh UCL/UEL/UECL results
python -m soccer.clubs.data.fetch_mls         # refresh MLS results (run after fetch_results)
python -m soccer.clubs.data.fetch_transfers   # refresh transfer aggregates
python -m soccer.clubs.data.fetch_xg          # refresh per-match xG (Actions only)
python -m soccer.clubs.data.fetch_shots       # refresh per-match shots (--source mirror works locally)
python -m soccer.clubs.model.tune             # re-tune per-league parameters
python -m soccer.clubs.model.train            # outcome model + holdout metrics
python -m soccer.clubs.model.export_ratings   # -> artifacts/club_elo_ratings.json
python -m soccer.clubs.model.mls_forecast     # MLS Shield / playoff / MLS Cup odds
python -m soccer.clubs.model.backtest_mls     # score that forecast on a past season
python -m soccer.clubs.daily.run              # the whole daily pipeline
```

`club_elo_ratings.json` is the export bridge: per league, the current
membership's ratings plus provenance (params, seasons, match counts), in the
same spirit as the international `elo_ratings.json` the World Cup Tickets
site consumes.

## Roadmap

- [x] Results dataset + refresh for all five leagues
- [x] Per-league club Elo with tuned K / home advantage / rollover / entry
- [x] Pooled W/D/L outcome model, temporally validated vs. baseline
- [x] Ratings export bridge (`club_elo_ratings.json`)
- [x] European competition results (UCL/UEL/UECL) as cross-league glue,
  validated to help on the league holdout
- [x] Transfermarkt transfer-spend features + market-value/wage upload slot
- [x] Daily runner: slate predictions, graded ledger, league-table Monte
  Carlo, site JSON (`daily/`, workflow `soccer-daily.yml`)
- [x] Second divisions in-pool: promotion carry-in from real D2 form with a
  tuned winner's-curse blend, validated to beat the flat entry rating on
  the top-flight holdout; D2 slates predicted and graded daily
- [x] Squad market values populated from Transfermarkt screenshots
  (`data/market_values/`) — every league-season with match data now has
  value + full squad-composition stats; wages still unpopulated (no source
  wired up yet)
- [x] A soccer page in `web/` (`/soccer`) reading
  `web/public/data/soccer/latest.json` — league rankings, daily slate,
  club ratings
- [x] MLS: its own unglued Elo pool (`data/fetch_mls.py`,
  philo92/mls-elo), squad economics, ratings and the rankings page —
  no daily slate (the source has no upcoming-fixture data)
- [x] MLS forecast: Supporters' Shield, conference seeding and the MLS
  Cup bracket (`model/mls_forecast.py`), run daily off a remaining
  schedule reconstructed from the league's format. Walk-forward on 2025
  (`model/backtest_mls.py`): expected-points MAE 3.1 with +0.1 bias at
  the three-quarter mark, playoff-qualification Brier 0.044 against a
  0.240 base-rate baseline
- [x] xG layer: per-match xG committed + Actions-refreshed, rolling
  xG-net form as the strongest non-Elo model feature (validated +2.1 SE
  on the 2023-24 holdout)
- [x] Shot layer: per-match shots + shots on target committed and
  Actions-refreshed, rolling shots-on-target form as a validated model
  feature (+2.9 SE on the 2024-25 + 2025-26 holdout) and the live
  chance-creation signal while Understat is stale
- [~] Revive the Understat feed — fetcher rewritten against the
  `getLeagueData` endpoint; waiting on the first Actions refresh to
  confirm the live response shape (the sandbox cannot reach the host)
- [ ] Promote any of the advanced Understat features (npxG, xPts, PPDA,
  deep, rest) — only after a season of live rows; today's ablation is
  scored on a dead feed and says nothing about them
- [ ] Shot coverage for the second divisions (needs a football-data.co.uk
  name mapping that can't be derived from the sandbox-reachable mirror)
- [ ] Second-division futures (promotion odds) — one config flip in
  `daily/run.py` once wanted
- [ ] Wage bills populated (Capology/FBref, still manual)
- [ ] Odds API soccer keys (`soccer_epl`, …) on the `/vegas` slate, model
  picks with edge-vs-market (quota permitting)
- [ ] A rating pool for non-top-5 European clubs so every UEFA match
  (not just top-5 pairings) feeds the glue
- [ ] Dixon–Coles low-score correction if the Poisson calibration drifts
