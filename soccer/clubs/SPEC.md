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
| Top-flight holdout (3,503 matches) | **0.9901** |
| Second-division holdout (3,221 matches) | 1.0600 |
| All divisions | 1.0236 |
| class-frequency baseline (all) | 1.0766 |

Second divisions are genuinely harder to predict — flatter, draw-heavier —
which the numbers say plainly. Per-league top-flight holdout log loss runs
0.975 (La Liga) to 1.010 (EPL), beating the frequency baseline everywhere.
Artifacts: `model/artifacts/outcome_model.pkl`, `metrics.csv`.

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
is a completed-match log with no upcoming fixtures, so MLS naturally has no
daily slate or futures Monte Carlo — ratings, squad economics and the
rankings page only. Because `fetch_results.py` owns `results.csv` outright
and rewrites it from the openfootball leagues alone, `fetch_mls.py` always
runs *after* it in `daily/run.py`'s refresh step, merging in rather than
overwriting.

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
│   ├── football_txt.py      # parser for the openfootball Football.TXT format
│   ├── fetch_results.py     # football.json + country-repo txt → results.csv
│   ├── fetch_uefa.py        # champions-league repo → uefa_results.csv
│   ├── fetch_mls.py         # philo92/mls-elo → results.csv (merges "mls" rows only)
│   ├── fetch_xg.py          # understat.com → xg_matches.csv (Actions-only; merge posture)
│   ├── fetch_transfers.py   # ewenme/transfers → club_season_transfers.csv
│   ├── market_values/       # optional squad value / wage uploads (see README)
│   ├── results.csv          # committed league results + current-season fixtures
│   ├── xg_matches.csv       # committed per-match xG (top-5 flights, 2014-15 →)
│   ├── uefa_results.csv     # committed UCL/UEL/UECL results, league-mapped
│   └── club_season_transfers.csv
├── model/
│   ├── elo.py               # ClubEloEngine (per-league pools, rollover, entry rating)
│   ├── europe.py            # UEFA cross-league glue replay
│   ├── xg.py                # rolling xG-form feature (xg_net_diff)
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
python -m soccer.clubs.data.fetch_mls         # refresh MLS results (run after fetch_results)
python -m soccer.clubs.data.fetch_transfers   # refresh transfer aggregates
python -m soccer.clubs.data.fetch_xg          # refresh per-match xG (Actions only)
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
  no daily slate/futures (the source has no upcoming-fixture data)
- [x] xG layer: per-match xG committed + Actions-refreshed, rolling
  xG-net form as the strongest non-Elo model feature (validated +2.1 SE
  on the 2023-24 holdout)
- [ ] Second-division futures (promotion odds) — one config flip in
  `daily/run.py` once wanted
- [ ] Wage bills populated (Capology/FBref, still manual)
- [ ] Odds API soccer keys (`soccer_epl`, …) on the `/vegas` slate, model
  picks with edge-vs-market (quota permitting)
- [ ] A rating pool for non-top-5 European clubs so every UEFA match
  (not just top-5 pairings) feeds the glue
- [ ] Dixon–Coles low-score correction if the Poisson calibration drifts
