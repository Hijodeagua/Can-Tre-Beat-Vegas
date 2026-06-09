# Can Tre Beat Vegas?

Probably not — but let's see. A Python-based sports betting tracker and modeling
system that snapshots bookmaker odds twice daily, tracks line movement across
sportsbooks, and trains models to pick winners (and covers) against the closing
line.

## Features

- **Multi-sport odds ingestion** — The Odds API (free tier) for NFL and NBA
  moneylines, spreads, and totals across major US sportsbooks; World Cup planned
- **Bookmaker tracking** — per-book odds history, sportsbook comparison charts,
  bookmaker variance and accuracy analysis (which bookies move first, which are sharpest)
- **Line movement** — opening vs. closing spread aggregation from timestamped
  odds snapshots (`data/odds_api_data_*.csv`)
- **NFL LightGBM models** — straight-up winner (`win`) and against-the-spread
  cover (`ats`) targets, 79 rolling/schedule features, temporal train/val split,
  baseline comparison (`NFL/model/`)
- **Soccer / World Cup model** — custom international Elo (fresh 2006 start,
  tiered K-factors, friendlies barely weighted) plus a multinomial outcome
  model with host effects and FIFA-rating squad-strength hooks; predicts the
  2026 World Cup slate (`soccer/`, spec in `soccer/SPEC.md`)
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
cd NFL/model
python3 schedule.py --refresh   # one-time: pull latest nflverse schedule
python3 train.py --target win   # straight-up winner
python3 train.py --target ats   # cover the spread (drops pushes)
```

Saved models, metrics, and feature importances land in `NFL/model/artifacts/`.

### 6. Generate a report

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
| `daily-report.yml` | daily | Builds the HTML daily report and charts into `reports/` |

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
│   ├── model/               # LightGBM win/ATS models
│   │   ├── features.py      # Rolling feature engineering (79 features)
│   │   ├── schedule.py      # nflverse schedule loader
│   │   ├── line_movement.py # Opening vs closing spread aggregator
│   │   ├── train.py         # Temporal split + LightGBM + baselines
│   │   └── artifacts/       # Saved models, metrics, importances
│   ├── Week_1/ ... Week_7/  # Weekly picks vs Vegas, graded
│   └── Weeks.md             # Season-long results tally
├── soccer/                  # World Cup / international soccer model
│   ├── SPEC.md              # Model spec (Elo + squad-strength adjustments)
│   ├── data/                # International results 1872–present + fixtures
│   └── model/               # Elo engine, training, fixture predictions
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
- [ ] World Cup odds ingestion (add `soccer_fifa_world_cup` to the Odds API
  config) and soccer model picks on the slate
- [ ] Wire NBA model predictions into the slate for current games
  (the export already joins `data/predictions/*.csv` when dates match)
- [ ] NFL model predictions for upcoming games (LightGBM models are trained;
  need an inference script writing to `data/predictions/`)
- [ ] Wire `line_movement.py` output in as a model feature

## License

MIT
