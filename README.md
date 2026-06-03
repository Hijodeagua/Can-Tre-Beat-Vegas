# Can-Tre-Beat-Vegas

> Probably not, but let's see if we can.

A personal sports-betting data project that pulls NFL and NBA odds from
[The Odds API](https://the-odds-api.com/), tracks how sportsbooks move and
how accurate they are, trains predictive models, and produces daily
HTML/PNG reports. There is **no web app** — the project is a collection of
Python data jobs whose outputs are static reports, charts, and CSV
snapshots committed back into the repo by GitHub Actions.

## What it does

1. **Odds collection** — On a schedule, fetches moneyline / spread / totals
   odds for every NFL and NBA game from The Odds API and saves timestamped
   CSV snapshots to `data/` and `data/nba/`. Each snapshot records
   per-sportsbook odds plus extras like divisional-matchup flags and
   travel distance between stadiums/arenas.
2. **Daily NBA report** — Builds a mobile-friendly HTML report (Charlotte
   Hornets focus + league-wide bookmaker accuracy rankings + simple ML
   win/score predictions + historical accuracy tracking) and accompanying
   PNG charts.
3. **NFL model** — Trains a LightGBM classifier on engineered rolling
   team-stat features to predict straight-up winners and against-the-spread
   covers, with a temporal train/val/test split and baseline comparisons.

## Project structure

```
.
├── data/                       # Odds snapshots + datasets (committed)
│   ├── odds_api_data_*.csv     #   NFL odds snapshots
│   ├── 2023-2025W3.csv         #   per-team NFL stats (NFL model input)
│   ├── nba/                    #   NBA odds snapshots, results, team stats
│   ├── models/  predictions/   #   saved NBA models + prediction history
│   └── schedules/
├── data_jobs/
│   ├── odds_api/               # MAINTAINED odds fetcher (used by CI)
│   │   ├── client.py           #   API client w/ quota tracking
│   │   ├── config.py           #   sports + team metadata
│   │   ├── processors.py       #   API JSON -> DataFrame -> CSV
│   │   └── fetch_odds.py       #   CLI entry point
│   ├── reports/
│   │   ├── simplified_daily_report.py   # NBA HTML report (used by CI)
│   │   └── generate_daily_report.py     # Hornets weekly report variant
│   ├── nbaodds.py                       # legacy standalone NBA puller
│   └── pull_in_odds_api_data.py         # legacy standalone NFL puller
├── NFL/model/                  # LightGBM NFL model (features + train)
│   └── artifacts/              #   saved model, metrics, feature importances
├── reports/                    # Generated HTML reports + PNG charts
└── .github/workflows/          # Scheduled odds pulls + daily report
```

## Requirements

- Python 3.11
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Running

### Fetch odds (requires an API key)

The odds fetcher needs a free key from The Odds API, supplied via the
`ODDS_API_KEY` environment variable:

```bash
export ODDS_API_KEY="your_key_here"
python -m data_jobs.odds_api.fetch_odds --sport all   # or nfl / nba
python -m data_jobs.odds_api.fetch_odds --report      # show quota usage
```

The client tracks usage in `data/api_usage.json` and warns as you approach
the free-tier monthly limit (500 requests).

### Generate the NBA daily report (works offline on committed data)

```bash
python -m data_jobs.reports.simplified_daily_report --days 30 --no-email
```

This writes `reports/daily_report_<date>.html` plus charts under
`reports/charts/`. Email delivery is optional and only attempted when
`SENDER_EMAIL`, `SENDER_PASSWORD`, and `RECIPIENT_EMAIL` are set (omit
`--no-email`). The Hornets-focused variant is
`data_jobs.reports.generate_daily_report`.

### Train the NFL model (works offline on committed data)

```bash
cd NFL/model
python3 train.py --target win   # straight-up winner
python3 train.py --target ats   # against-the-spread cover
```

Outputs the trained booster, metrics, and feature importances to
`NFL/model/artifacts/`. See `NFL/model/README.md` for the feature pipeline.

## Automation (GitHub Actions)

- **`unified-odds.yml`** — fetches NFL + NBA odds twice daily (preferred).
- **`odds.yml` / `nba-odds.yml`** — legacy single-sport pulls, kept for
  backwards compatibility.
- **`daily-report.yml`** — generates and commits the daily NBA report, and
  optionally emails it.

All odds workflows require an `ODDS_API_KEY` repository secret. The email
step additionally requires `SMTP_USERNAME` / `SMTP_PASSWORD` secrets and is
skipped automatically when they are absent.

## What works vs. what needs external access

| Component | Status | Notes |
|-----------|--------|-------|
| NBA daily report | Works offline | Runs against committed CSV snapshots. |
| NFL LightGBM model | Works offline | Trains on bundled `data/2023-2025W3.csv`. |
| Odds fetching | Needs `ODDS_API_KEY` | Live calls to The Odds API. |
| Email delivery | Needs SMTP secrets | Optional; skipped when unconfigured. |

## Disclaimer

This is a personal hobby/educational project for exploring sports-betting
data. It is not betting advice, and the models are not guaranteed to be
profitable (the name is a joke).
