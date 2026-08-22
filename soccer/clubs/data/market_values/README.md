# Squad market values & wage bills — upload slot

Optional per-season squad-economics data consumed by
`soccer/clubs/model/features.py` (`value_diff_z` / `wage_diff_z`). Until
files land here the outcome model imputes zeros and degrades gracefully —
same contract as the international model's FIFA-ratings layer.

## Schema

One file per season, named `values_<season>.csv` (e.g. `values_2026-27.csv`):

```csv
league,club,squad_value_eur_m,wage_bill_eur_m
epl,Arsenal FC,1180.0,235.0
epl,Manchester City FC,1250.0,260.0
bundesliga,FC Bayern München,930.0,190.0
```

- `league` — our league key (`epl`, `bundesliga`, `la_liga`, `serie_a`, `ligue_1`)
- `club` — whatever spelling Transfermarkt shows for that season (e.g.
  "Manchester United", not "Manchester United FC" pre-2020). The loader
  canonicalizes through `leagues.py`'s alias maps on load, same as every
  other data source in this pipeline — no need to pre-canonicalize.
- `squad_value_eur_m` — total squad market value, EUR millions
- `wage_bill_eur_m` — optional; annual gross wage bill, EUR millions.
  Leave the column out (or blank) if you only have values.

Vintage matters the same way it does for FIFA ratings: a season's file
should hold the values as of (or close to) that season's start, not
today's. Features are z-scored within league-season, so absolute currency
level and inflation across seasons don't leak.

## Where to get the data

- **Market values** — Transfermarkt club pages, or the
  [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)
  project's `clubs`/`player_valuations` exports (data.world / Kaggle),
  aggregated to club level. The proxy in the hosted dev environment blocks
  transfermarkt.com itself, so populate these locally.
- **Wage bills** — Capology's league payroll tables are the usual manual
  source; FBref mirrors them per squad.
