# Top-5 European Leagues — Club Elo Models

Per-league Elo models for the Premier League, Bundesliga, La Liga, Serie A
and Ligue 1 — the club-football sibling of the international model in
`soccer/model/`. Same DNA (logistic expectation, margin-of-victory
multiplier, draws as 0.5, a multinomial outcome layer on the Elo gap), with
the structure club league play actually has: closed per-league pools,
promotion/relegation, and summer squad churn.

## Elo rules

- **Separate pool per league.** Domestic results never compare clubs across
  leagues, so each league is its own closed Elo economy. An EPL 1600 and a
  Ligue 1 1600 are not claims about each other — don't rank across leagues.
- **Fresh start at each league's first upstream season** (see data table
  below), every club at 1500.
- **Season rollover regression** — at every season boundary all known
  ratings regress toward 1500 by a tuned fraction ρ: squads churn over a
  summer, and last May's rating overstates what returns in August.
- **Promotion / relegation** — a club never seen before enters at a tuned
  entry rating below base (promoted sides are usually worse than the average
  incumbent). A relegated club's rating keeps regressing while it is out of
  the league and is picked back up on return, so a bounce-back keeps some of
  its old level without returning at full strength.
- **Margin-of-victory multiplier** — eloratings.net convention: ×1 for
  1-goal wins, ×1.5 for 2, ×(11+N)/8 for N ≥ 3.
- **Home advantage** — tuned per league, added to the home side inside the
  expectation. League CSVs carry no neutral-site flag; the rare neutral
  match is absorbed as noise.
- **Draws** count as 0.5.

## Tuned parameters

`model/tune.py` grid-searches (K, home advantage, ρ, entry rating) per
league, minimizing the **one-step-ahead Brier score** of the Elo expectation
(each match predicted using only earlier matches). The first two seasons of
each league are burn-in and 2024-25 onward is excluded — that stays the
untouched holdout for `train.py`. Current values live in
`model/artifacts/tuned_params.json`; expect K ≈ 10–14 (far below the
international K's — 38 games a season against familiar opponents means each
result carries less news), home advantage ≈ 45–60 Elo, ρ ≈ 0.10–0.15.

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
for the opponent.

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

## Probability model

Multinomial logistic regression over {home win, draw, away win} on the
venue-adjusted Elo gap plus the squad-economics differentials, pooled
across the five leagues — the gap→probability curve is shared, while each
league's gaps already come from its own tuned (and UEFA-glued) pool.
Temporal validation on the two held-out seasons (2024-25 + 2025-26, never
seen by tuning or training):

| | log loss | accuracy |
|---|---|---|
| Full model (glued Elo + economics) | **0.9902** | 52.2% |
| Elo-only | 0.9906 | — |
| class-frequency baseline | 1.0750 | — |

Per-league holdout log loss runs 0.977 (La Liga) to 1.011 (EPL), beating
the frequency baseline everywhere. Artifacts:
`model/artifacts/outcome_model.pkl`, `metrics.csv`.

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
| Bundesliga | `bundesliga` | `de.1` | 2010-11 → |
| La Liga | `la_liga` | `es.1` | 2012-13 → |
| Serie A | `serie_a` | `it.1` | 2013-14 → |
| Ligue 1 | `ligue_1` | `fr.1` | 2014-15 → |

Wrinkles handled in the fetch, so nothing downstream sees them:

- **Name drift.** The upstream renamed most clubs to long legal names midway
  through its history (2020-21 wave for en/de/es/it, 2023-24 for fr) —
  "Manchester City" vs "Manchester City FC" would split one club's history
  in two. `data/leagues.py` maps every historical spelling to the current
  canonical name.
- **Score shapes.** Finals normally arrive as `score.ft = [h, a]`; the
  newest season files serialize 0-0 finals as a bare `score = [0, 0]`. Both
  are accepted.
- **Known holes.** Ligue 1 2019-20 stops at the COVID abandonment (279
  matches); the upstream never filled the last matchday of La Liga and
  Serie A 2024-25 (370/380 each) and one cancelled Ligue 1 2025-26 match.
  Scoreless past-season rows are dropped rather than guessed.
- **New seasons.** Every fetch probes one season past the current one, so
  the 2026-27 files start flowing in as soon as openfootball publishes them
  — no code change.

## Pipeline

```
soccer/clubs/
├── SPEC.md                  # this file
├── data/
│   ├── leagues.py           # league registry + canonical-name aliases (+ UEFA aliases)
│   ├── football_txt.py      # parser for the openfootball Football.TXT format
│   ├── fetch_results.py     # football.json + country-repo txt → results.csv
│   ├── fetch_uefa.py        # champions-league repo → uefa_results.csv
│   ├── fetch_transfers.py   # ewenme/transfers → club_season_transfers.csv
│   ├── market_values/       # optional squad value / wage uploads (see README)
│   ├── results.csv          # committed league results + current-season fixtures
│   ├── uefa_results.csv     # committed UCL/UEL/UECL results, league-mapped
│   └── club_season_transfers.csv
├── model/
│   ├── elo.py               # ClubEloEngine (per-league pools, rollover, entry rating)
│   ├── europe.py            # UEFA cross-league glue replay
│   ├── features.py          # spend / value / wage differentials (z within league-season)
│   ├── tune.py              # per-league parameter grid search
│   ├── train.py             # pooled multinomial outcome model + temporal validation
│   ├── export_ratings.py    # → artifacts/club_elo_ratings.json (glued, all leagues)
│   └── artifacts/           # tuned_params.json, outcome model, metrics, ratings JSON
└── daily/                   # the daily runner (see daily/README.md)
```

```bash
python -m soccer.clubs.data.fetch_results     # refresh league results + fixtures
python -m soccer.clubs.data.fetch_uefa        # refresh UCL/UEL/UECL results
python -m soccer.clubs.data.fetch_transfers   # refresh transfer aggregates
python -m soccer.clubs.model.tune             # re-tune per-league parameters
python -m soccer.clubs.model.train            # outcome model + holdout metrics
python -m soccer.clubs.model.export_ratings   # -> artifacts/club_elo_ratings.json
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
- [ ] Squad market values + wages populated (locally) and validated
- [ ] Odds API soccer keys (`soccer_epl`, …) on the `/vegas` slate, model
  picks with edge-vs-market (quota permitting)
- [ ] A soccer page in `web/` reading `web/public/data/soccer/latest.json`
- [ ] Promotion carry-in: seed promoted clubs from second-division form
  instead of a flat entry rating
- [ ] A rating pool for non-top-5 European clubs so every UEFA match
  (not just top-5 pairings) feeds the glue
- [ ] Dixon–Coles low-score correction if the Poisson calibration drifts
