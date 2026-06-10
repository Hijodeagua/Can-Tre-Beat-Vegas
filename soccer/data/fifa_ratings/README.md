# FIFA ratings uploads

Drop one CSV per game edition here, named `fifa_<edition_year>.csv`
(e.g. `fifa_2007.csv` for FIFA 07), with columns:

    team,player,overall

- `team` must match national team names in `soccer/data/results.csv`
  (e.g. "Argentina", "South Korea", "United States").
- `overall` is the player's overall rating (0-100).
- Use the edition published closest to but before the matches you care
  about; matches older than the oldest edition are imputed from it.

The pipeline (`soccer/model/squad.py`) computes top-25 depth and top-5 star
averages per squad and z-scores them across squads within each edition.
National "squads" are proxied by player nationality (top players per country
by overall), since full-game dumps don't carry tournament rosters.

Raw dumps in other layouts (stefanoleone992 Kaggle files, "male2" files,
sofifa-web-scraper output) can be converted with:

    python -m soccer.model.convert_ratings <raw.csv> <release_year>

Name the edition by its RELEASE year: FIFA 21 released Oct 2020 → `2020`.

## Currently loaded

| File | Edition | Source |
|---|---|---|
| `fifa_2014.csv` … `fifa_2019.csv` | FIFA 15–20 | sofifa dumps mirrored on GitHub |
| `fifa_2020.csv` | FIFA 21 | sofifa dump mirrored on GitHub |

## Wanted (manual upload — sofifa/Kaggle/HF are blocked from this environment)

- **FC 24 / FC 25 / FC 26 (`2023`–`2025`)** — highest priority: without a
  current vintage, 2026 World Cup predictions impute squads from FIFA 21.
- **FIFA 22 / FIFA 23 (`2021`, `2022`)** — fills the recent gap.
- **FIFA 07–14 (`2006`–`2013`)** — backfills the fresh-2006 Elo era so α/β
  train on the full history instead of imputing pre-2014 from FIFA 15.
