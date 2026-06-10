# Soccer / World Cup Model — Spec

A custom Elo-based international soccer model, iterating on the spirit of
Silver Bulletin's Pelé model, with squad-quality adjustments from FIFA
ratings data and host/venue effects. Feeds match probabilities for the
next-48-hours slate and (later) a full tournament Monte Carlo simulation.

## Core idea

Elo is the backbone. Squad quality enters as an *adjustment to the Elo gap*,
not a replacement for it:

```
adjusted_gap = elo_gap + α × depth_diff + β × star_diff
```

Where:

- `elo_gap` — our custom Elo (see below), home minus away, including
  venue adjustments
- `depth_diff` — avg FIFA rating of each squad's **top 25** players,
  home minus away, z-scored across all squads in the dataset
- `star_diff` — avg FIFA rating of each squad's **top 5** players,
  home minus away, z-scored
- `α` and `β` are tuned separately by the training step

Expectation: |β| < |α| because star ratings are noisier and more
context-dependent — but the data decides. If β turns out near zero in the
group stage and meaningful in knockouts, that's a finding worth a
conditional (stage-interacted) version of the model.

## Custom Elo rules

- **Fresh start 2006-01-01** — every team starts at 1500 (no inherited
  pre-2006 history). ~20 years / 5 World Cup cycles of evolution.
- **Tiered K-factors** — friendlies weighted the absolute lowest, barely
  moving ratings:

  | Tier | K |
  |---|---|
  | FIFA World Cup (finals) | 60 |
  | Continental finals (Euro, Copa América, AFCON, Asian Cup, Gold Cup) | 50 |
  | World Cup / continental qualification | 40 |
  | Nations Leagues (UEFA, CONCACAF) | 30 |
  | Other competitive tournaments (COSAFA, CECAFA, King's Cup, …) | 15 |
  | **Friendlies** | **5** |

- **Margin-of-victory multiplier** — eloratings.net convention:
  ×1 for 1-goal wins, ×1.5 for 2 goals, ×(11+N)/8 for N ≥ 3.
- **Home advantage** — +80 Elo points to the home side when the match is
  not at a neutral venue (tunable).
- **Draws** count as 0.5.

## Host / venue effects (v1 signals)

2026 is hosted in the US, Mexico, and Canada — host effects matter:

- `home_venue` — non-neutral match at the team's own ground
- `host_country` — team playing a "neutral" tournament match inside its
  own country (the dataset marks World Cup host games neutral; the host
  flag recovers the real edge)

Travel distance / altitude / rest days are v2 candidates.

## FIFA ratings layer (squad strength)

Source: EA FIFA / FC player ratings (sofifa dumps or manual fbref/Kaggle
uploads). Implementation details:

- **Rating vintage** — use the edition published closest to but **before**
  the match date. A player's 2010 rating for the 2010 World Cup, never
  career peaks. Critical for the star metric, since individual peaks are
  short.
- **Top-5 selection** — by overall rating regardless of position; we're
  capturing quality concentration, not formation fit.
- **Normalization** — international-quality players cluster ~75–94
  overall, so raw squad differentials are only a few points. Convert squad
  aggregates to **z-scores across all squads in the dataset** so α and β
  are interpretable in standard-deviation units.
- **2006 backfill** — the oldest widely-available data is FIFA 07/08;
  impute 2006 from the FIFA 07 release (squad quality moves slowly).
- **Decoupled columns** — `depth_diff` and `star_diff` are stored as
  separate per-match columns, never collapsed into one squad score, so we
  can weight them differently by stage (group vs knockout), venue, etc.
  without restructuring the pipeline.

Upload schema: `soccer/data/fifa_ratings/fifa_<edition_year>.csv` with
columns `team,player,overall` (see that directory's README).

## Probability model

Multinomial logistic regression over {home win, draw, away win} on:

- `elo_gap` (venue-adjusted)
- `depth_diff_z`, `star_diff_z` (0-imputed until ratings data lands —
  the model degrades gracefully to Elo-only)
- `host_country` flag
- knockout-stage flag (and, in the conditional version, its interactions
  with the squad features)

Trained on competitive internationals 2006→cutoff, temporally validated on
the most recent seasons, with log loss / accuracy reported against an
Elo-only baseline. Output: per-match W/D/L probabilities for upcoming
fixtures (the 2026 World Cup schedule ships inside the results dataset).

## Data sources

| Source | Contents | Status |
|---|---|---|
| martj42/international_results (GitHub, CC0) | 49k internationals 1872–present incl. 2026 WC fixtures, tournament type, neutral flag | committed at `soccer/data/results.csv` |
| same repo, `shootouts.csv` | penalty shootout winners | committed |
| EA FIFA ratings (sofifa / Kaggle dumps) | player overall by edition | **manual upload** |
| fbref | squad/player stats | manual upload, v2 |
| The Odds API (`soccer_fifa_world_cup`) | bookmaker odds for the slate | wired separately into `data_jobs/` |

## Pipeline

```
soccer/
├── SPEC.md                  # this file
├── data/
│   ├── results.csv          # full international results + 2026 fixtures
│   ├── shootouts.csv
│   └── fifa_ratings/        # fifa_<year>.csv uploads
└── model/
    ├── elo.py               # tiered-K Elo engine (fresh 2006 start)
    ├── squad.py             # vintage-matched depth/star features
    ├── train.py             # multinomial LR + temporal validation
    ├── predict.py           # upcoming-fixture probabilities
    └── artifacts/           # ratings snapshot, model pkl, metrics
```

## Roadmap

- [x] Elo engine, training, validation, upcoming-fixture predictions
- [x] FIFA ratings layer + α/β fit — editions 2014–2020 (FIFA 15–21) loaded
  from GitHub mirrors; the squad model beats the Elo-only baseline (log loss
  0.8803 vs 0.8827 on 2024+ holdout). Early finding: **star power
  (top-5) carries far more signal than depth (top-25)** — β ≈ +0.35 vs
  α ≈ −0.07 on the home-win class, the opposite of the prior.
- [x] SoFIFA API client (`soccer/model/sofifa_client.py`) — pulls
  national-team squad ratings per edition straight from api.sofifa.net (no
  key needed), self-throttled and resumable. Run locally to fill FC 24–26 +
  FIFA 22/23 + FIFA 07–14, then refit.
- [ ] World Cup odds ingestion → model picks on the `/vegas` slate
- [ ] Group vs knockout conditional version
- [ ] Tournament Monte Carlo (group tables, brackets, championship odds)
- [ ] Travel/altitude/rest effects
