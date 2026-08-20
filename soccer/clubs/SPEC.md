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

## Probability model

Multinomial logistic regression over {home win, draw, away win} on the
venue-adjusted Elo gap, pooled across the five leagues — the gap→probability
curve is shared, while each league's gaps already come from its own tuned
pool. Temporal validation on the two held-out seasons (2024-25 + 2025-26,
never seen by tuning or training):

| | log loss | accuracy |
|---|---|---|
| Elo model | **0.9909** | 51.9% |
| class-frequency baseline | 1.0750 | — |

Per-league holdout log loss runs 0.978 (La Liga) to 1.012 (EPL), beating the
frequency baseline everywhere. Artifacts: `model/artifacts/outcome_model.pkl`,
`metrics.csv`.

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
│   ├── leagues.py           # league registry + canonical-name aliases
│   ├── fetch_results.py     # openfootball → results.csv (best-effort, network-gated)
│   └── results.csv          # committed normalized results, all five leagues
└── model/
    ├── elo.py               # ClubEloEngine (per-league pools, rollover, entry rating)
    ├── tune.py              # per-league parameter grid search
    ├── train.py             # pooled multinomial outcome model + temporal validation
    ├── export_ratings.py    # → artifacts/club_elo_ratings.json (all leagues)
    └── artifacts/           # tuned_params.json, outcome model, metrics, ratings JSON
```

```bash
python -m soccer.clubs.data.fetch_results     # refresh results (best effort)
python -m soccer.clubs.model.tune             # re-tune per-league parameters
python -m soccer.clubs.model.train            # outcome model + holdout metrics
python -m soccer.clubs.model.export_ratings   # -> artifacts/club_elo_ratings.json
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
- [ ] Upcoming-fixture predictions once 2026-27 fixtures land upstream
- [ ] Odds API soccer keys (`soccer_epl`, …) on the `/vegas` slate, model
  picks with edge-vs-market (quota permitting)
- [ ] Promotion carry-in: seed promoted clubs from second-division form
  instead of a flat entry rating
- [ ] European competition results (UCL/UEL) as cross-league glue
