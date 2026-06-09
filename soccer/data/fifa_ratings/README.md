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
