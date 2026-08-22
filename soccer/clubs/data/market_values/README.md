# Squad market values & wage bills — upload slot

Optional per-season squad-economics data consumed by
`soccer/clubs/model/features.py` (`value_diff_z` / `wage_diff_z`). Until
files land here the outcome model imputes zeros and degrades gracefully —
same contract as the international model's FIFA-ratings layer.

## Schema

One file per season, named `values_<season>.csv`. The season string is
whatever the league itself uses — `values_2026-27.csv` for the
July-boundary European leagues, `values_2026.csv` (bare calendar year) for
MLS — the loader derives `season` straight from the filename, so the two
formats coexist without collision:

```csv
league,club,squad_value_eur_m,wage_bill_eur_m,squad_size,avg_age,foreigners,avg_value_eur_m
epl,Arsenal FC,1180.0,235.0,25,25.4,19,47.2
epl,Manchester City FC,1250.0,260.0,24,27.1,21,52.1
bundesliga,FC Bayern München,930.0,190.0,,,,
```

- `league` — our league key (`epl`, `bundesliga`, `bundesliga_2`, `la_liga`,
  `la_liga_2`, `serie_a`, `serie_b`, `ligue_1`, `ligue_2`, `championship`,
  `mls`)
- `club` — whatever spelling Transfermarkt shows for that season (e.g.
  "Manchester United", not "Manchester United FC" pre-2020). The loader
  canonicalizes through `leagues.py`'s alias maps on load, same as every
  other data source in this pipeline — no need to pre-canonicalize.
- `squad_value_eur_m` — total squad market value, EUR millions (Transfermarkt's
  "Total market value" column). Required — this is the only column the
  outcome model actually trains on (`value_diff_z`).
- `wage_bill_eur_m` — optional; annual gross wage bill, EUR millions
  (`wage_diff_z`). Leave the column out (or blank) if you only have values.
- `squad_size`, `avg_age`, `foreigners`, `avg_value_eur_m` — optional,
  transcribed straight from Transfermarkt's "Squad" / "ø age" /
  "Foreigners" / "ø market value" columns (the last converted to EUR
  millions, e.g. "€991k" -> `0.991`). Not model features — they feed the
  site's cross-league rankings page only (`daily/export_site.py`'s
  `league_rankings`), display-only, no effect on training. Leave blank for
  seasons transcribed before this column set existed; the rankings page
  reports each league's most recent season that actually has them,
  separately from the season used for value/wage.

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
