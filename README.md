# Can Tre Beat Vegas?

Probably not — but let's see. A Python-based sports betting tracker and modeling
system that snapshots bookmaker odds twice daily, tracks line movement across
sportsbooks, and trains models to pick winners (and covers) against the closing
line.

## The 2025 answer: no

The season is graded and the books are in
([`NFL/inventory/INVENTORY_2025.md`](NFL/inventory/INVENTORY_2025.md),
[`NFL/Weeks.md`](NFL/Weeks.md)). Over 4,350 out-of-sample games from 2010-2025,
walking the model forward one retrain per season:

| | model | closing line |
|---|---|---|
| straight up, 2010-2025 | 64.3% | **66.4%** |
| straight up, 2025 only | 65.5% | **65.8%** |
| ATS, 2010-2025 | 50.5% | — (52.4% needed) |
| totals, 2010-2025 | 49.5% | — (52.4% needed) |

Blending the model into the market makes the market *worse* out of sample
(log loss 0.613 vs 0.610), which is the cleanest statement of the result: after
the closing line has spoken, the model adds nothing. Flat-stake ROI is -4.9% on
spreads and -4.5% on totals, both about what the vig costs.

That is the honest baseline every future change here gets measured against.
Details and the two places an edge might actually live are in
[`NFL/model/v2/README.md`](NFL/model/v2/README.md).

## Features

- **Multi-sport odds ingestion** — The Odds API (free tier) for NFL and NBA
  moneylines, spreads, and totals across major US sportsbooks; World Cup planned
- **Bookmaker tracking** — per-book odds history, sportsbook comparison charts,
  bookmaker variance and accuracy analysis (which bookies move first, which are sharpest)
- **Line movement** — opening vs. closing spread aggregation from timestamped
  odds snapshots (`data/odds_api_data_*.csv`)
- **NFL LightGBM models** — win / ATS / total targets on an Elo + form + market
  feature set, walk-forward validated one retrain per season, with a flat-stake
  betting backtest (`NFL/model/v2/`; the older single-split v1 is kept in
  `NFL/model/` for provenance)
- **Season inventory** — end-of-season audit of what was captured, what was
  missed, and how the closing line itself performed (`NFL/inventory/`)
- **Soccer / World Cup model** — custom international Elo (fresh 2006 start,
  tiered K-factors, friendlies barely weighted) plus a multinomial outcome
  model with host effects and FIFA-rating squad-strength hooks; predicts the
  2026 World Cup slate (`soccer/`, spec in `soccer/SPEC.md`)
- **Top-5 European league club Elo** — per-league Elo for the Premier League,
  Bundesliga, La Liga, Serie A and Ligue 1 (per-league tuned K / home
  advantage / season rollover, promoted-club entry ratings) with a pooled
  W/D/L outcome model, temporally validated on a two-season holdout
  (`soccer/clubs/`, spec in `soccer/clubs/SPEC.md`)
- **MLB daily model** — betting-blind Elo (K=3, +24 home, MOV-weighted)
  tested live against the 2026 season: three emails every morning (futures
  Monte Carlo, today's slate with simulated scores, yesterday's graded
  report card) plus the model card at `whosyurgoat.app/vegas/mlb`
  (`mlb/daily/`, writeup in `reports/mlb_elo_stat_associations.md`)
- **Daily HTML reports** — automated odds breakdowns, bookmaker performance,
  and team odds-history charts (`reports/`)
- **Weekly NFL picks** — model picks vs. Vegas, graded week by week
  (`NFL/Week_*/`, running tally in `NFL/Weeks.md`)
- **Automated refresh** — GitHub Actions fetch odds 2x daily and generate
  reports, tuned to stay inside the free-tier API quota (~60 requests/month)

## Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/Hijodeagua/Can-Tre-Beat-Vegas.git
cd Can-Tre-Beat-Vegas
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.11+ recommended.

### 3. Configure

Set your Odds API key (free tier: 500 requests/month at
[the-odds-api.com](https://the-odds-api.com/)):

```bash
export ODDS_API_KEY=your_key_here
```

No key is needed to work with the committed odds snapshots in `data/` —
the models and reports run entirely off those CSVs.

### 4. Fetch odds

```bash
# Snapshot current NFL + NBA odds (h2h, spreads, totals)
python -m data_jobs.odds_api.fetch_odds --sport all

# Just one sport
python -m data_jobs.odds_api.fetch_odds --sport nfl
```

### 5. Train the NFL models

```bash
# from the repo root
python3 NFL/model/schedule.py --refresh          # optional: pull latest nflverse schedule
python3 -m NFL.model.v2.train --target all --save   # win + ATS + total, walk-forward
```

Saved models, per-season metrics, backtests, feature importances, and the full
out-of-sample prediction files land in `NFL/model/v2/artifacts/`.

### 6. Score upcoming games

```bash
python3 -m NFL.model.v2.predict --season 2026 --week 1 --write
```

Writes to `data/predictions/`.

### 7. Audit a finished season

```bash
python3 -m NFL.inventory.audit --season 2025 --write         # data + Vegas report card
python3 -m NFL.inventory.grade_season --season 2025 --write  # graded weekly ledger
```

### 8. Generate a report

```bash
python -m data_jobs.reports.simplified_daily_report
```

## Web Tracker (`web/`)

A deployable Next.js 14 (App Router) static site — the **next-48-hours
homepage**. For every upcoming NFL and NBA game it shows the consensus line
(moneyline, spread, total), no-vig market win probability, line movement since
open, model picks with edge-vs-market when available, and an expandable
book-by-book odds table. It reads the static JSON in `web/public/data/`,
produced by the Python pipeline; it runs no Python server itself.

```bash
# 1. Regenerate the static JSON from the committed odds snapshots
python -m data_jobs.export_web_json

# 2. Build / run the front end
cd web
npm install
npm run dev        # http://localhost:3000/vegas
npm run build      # production build (static, prerendered)
```

The site is served under the `/vegas` base path (see `web/next.config.mjs`)
so it can be proxied at `whosyurgoat.app/vegas` by the
[hub](https://github.com/Hijodeagua/whosyurgoat-hub). The unified odds
workflow re-exports the JSON after every odds fetch, so a Vercel project
pointed at `web/` redeploys with fresh lines 2x daily.

## Automated refresh

GitHub Actions keep the data flowing without manual pulls:

| Workflow | Schedule | What it does |
|---|---|---|
| `unified-odds.yml` | 2x daily (10:00 / 22:00 UTC) | Fetches NFL + NBA odds snapshots, re-exports web JSON, commits |
| `daily-report.yml` | daily (10:00 UTC) | MLB daily pipeline: pulls new box scores, updates Elo, sends the futures / slate / grade emails, commits site data (see `mlb/daily/README.md`) |

The 2x-daily cadence is deliberate — it captures a morning line and an
evening line before games while conserving the free-tier API quota.

## Project Structure

```
Can-Tre-Beat-Vegas/
├── data_jobs/               # Data ingestion + reporting
│   ├── odds_api/            # The Odds API client
│   │   ├── client.py        # HTTP client + quota tracking
│   │   ├── config.py        # Sports config, team metadata, stadium coords
│   │   ├── fetch_odds.py    # CLI entry point (--sport nfl|nba|all)
│   │   └── processors.py    # Raw API response → tidy CSV
│   ├── reports/             # Daily report generators
│   └── export_web_json.py   # Next-48-hours slate → web/public/data/
├── NFL/
│   ├── model/               # v1 (superseded, kept for provenance)
│   │   ├── features.py      # Rolling feature engineering (79 features)
│   │   ├── schedule.py      # nflverse schedule loader
│   │   ├── line_movement.py # Open vs close aggregator (shared with v2)
│   │   ├── train.py         # Single temporal split + LightGBM
│   │   └── v2/              # Current model
│   │       ├── elo.py       # MOV-adjusted Elo, walk-forward
│   │       ├── dataset.py   # 45 features off the nflverse schedule
│   │       ├── train.py     # Season-by-season walk-forward + backtest
│   │       ├── predict.py   # Score upcoming games → data/predictions/
│   │       └── artifacts/   # Models, metrics, backtests, OOS predictions
│   ├── inventory/           # End-of-season audit
│   │   ├── audit.py         # Capture coverage + Vegas' report card
│   │   ├── grade_season.py  # Writes the graded weekly ledger
│   │   └── INVENTORY_*.md   # Generated season inventory
│   ├── Week_1/ ... Week_22/ # Weekly picks vs Vegas, graded (out of sample)
│   └── Weeks.md             # Season-long results tally
├── soccer/                  # World Cup / international soccer model
│   ├── SPEC.md              # Model spec (Elo + squad-strength adjustments)
│   ├── data/                # International results 1872–present + fixtures
│   ├── model/               # Elo engine, training, fixture predictions
│   └── clubs/               # Top-5 European league club Elo (EPL, Bundesliga,
│                            #   La Liga, Serie A, Ligue 1) — see clubs/SPEC.md
├── data/                    # Odds snapshots + stats
│   ├── odds_api_data_*.csv  # NFL odds snapshots (timestamped)
│   ├── nba/                 # NBA odds snapshots + actual game results
│   ├── schedules/           # Cached nflverse games.csv
│   ├── models/              # Pickled model checkpoints
│   └── predictions/         # Model prediction outputs
├── web/                     # Next.js front end (basePath /vegas)
│   ├── app/                 # Slate homepage + methodology
│   └── public/data/         # Static JSON written by export_web_json.py
├── reports/                 # Generated daily HTML reports + charts
└── .github/workflows/       # Automated odds fetch + report generation
```

## Roadmap

The end state is a **homepage for the next 48 hours of games** — every NFL,
NBA, and World Cup matchup coming up in the next two days, side by side with:

1. **The bookies' view** — current odds per sportsbook, consensus line, and
   how the line has moved since open
2. **My models' view** — win probability and ATS pick, with disagreement vs.
   the market highlighted (that's where the edges live)
3. **Track record** — running tally of model vs. Vegas, by sport and bet type

Steps to get there:

- [x] Static JSON export of the upcoming-48-hours slate (odds + model picks),
  same pattern as the [election tracker](https://github.com/Hijodeagua/Election-models-by-Tre)
  — `data_jobs/export_web_json.py`
- [x] Next.js front end reading that JSON (`web/`, served at `/vegas`)
- [ ] Deploy `web/` to Vercel and point the hub's `/vegas` rewrite at it
- [x] Soccer Elo + outcome model predicting 2026 World Cup fixtures
  (`soccer/`, see `soccer/SPEC.md` for the full roadmap)
- [x] Club Elo for the top-5 European leagues, tuned per league and validated
  against a two-season holdout (`soccer/clubs/`)
- [ ] Odds API soccer keys (`soccer_epl`, …) and club model picks on the
  slate (quota permitting)
- [ ] World Cup odds ingestion (add `soccer_fifa_world_cup` to the Odds API
  config) and soccer model picks on the slate
- [ ] Wire NBA model predictions into the slate for current games
  (the export already joins `data/predictions/*.csv` when dates match)
- [x] NFL model predictions for upcoming games —
  `python3 -m NFL.model.v2.predict --season 2026 --write`
- [ ] Point `export_web_json.py` at the v2 prediction files

### What the 2025 inventory says to fix next

Ranked by how much they'd move the answer, not by effort:

- [ ] **Capture the opener.** 2025 capture began a median of 11 days before
  kickoff, and weeks 1-4, 17 and 21 were missed entirely (83 of 285 games have
  no odds record). Beating the closing line is close to impossible; beating a
  stale opener is the winnable version of this bet, and we can't test it
  without openers.
- [ ] **Store the spread number, not just the juice.** The legacy schema kept
  `-110 / -110` but never `-3.5`, so the entire 2025 season has moneyline-only
  line movement. The current schema fixed this; don't regress it.
- [ ] **Log best-available price per book, not the average.** The snapshots
  already carry 8-10 books. Line shopping is a mechanical edge that doesn't
  require the model to be right about anything, and nothing in the repo
  measures it yet.
- [ ] Wire `line_movement.py` output in as a model feature (now that it
  actually returns 2025 games)
- [ ] EPA / success rate from nflverse play-by-play

## License

MIT
