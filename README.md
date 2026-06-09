# Can Tre Beat Vegas?

Sports-betting research repo: pulls live odds from
[The Odds API](https://the-odds-api.com/), enriches them with matchup
context (divisional games, travel distance, prime time), and stores
snapshots for model building.

## Layout

| Path | Purpose |
|------|---------|
| `data_jobs/pull_in_odds_api_data.py` | NFL odds snapshot → `data/odds_api_data_*.csv` |
| `data_jobs/nbaodds.py` | NBA odds snapshot → `data/nba/nba_odds_api_data_*.csv` |
| `data/` | Committed odds snapshots (timestamped + `_latest`) |
| `NFL/` | NFL modeling notebooks/scripts |
| `reports/` | Generated analysis reports |
| `.github/workflows/` | Scheduled data pulls |

## Setup

```bash
pip install -r requirements.txt
export ODDS_API_KEY=your_key_here   # never commit this
python data_jobs/pull_in_odds_api_data.py
python data_jobs/nbaodds.py
```

In CI the key comes from the `ODDS_API_KEY` GitHub Actions secret
(Settings → Secrets and variables → Actions). The scripts fail fast with a
clear error when it is missing — there is intentionally no hard-coded
fallback.

## Odds columns

Each snapshot row carries league, game time (ET), home/away teams,
divisional-matchup flag, arena-to-arena travel distance, prime-time flag,
cross-book average odds for spread / moneyline / totals, and per-book
detail columns.
