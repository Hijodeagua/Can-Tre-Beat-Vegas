# FIFA ratings uploads

## Pulling editions from the SoFIFA API (recommended)

`soccer/model/sofifa_client.py` pulls national-team squad ratings straight
from SoFIFA's free REST API and writes the `fifa_<year>.csv` files for you —
no manual download. **No API key needed** for the team/player endpoints (the
apiToken in SoFIFA's docs is only for their customizedPlayers endpoints).

Run it **locally** (api.sofifa.net is blocked from the Claude Code web
sandbox), then commit the CSVs:

    # Fill the editions we don't have yet (FIFA 07–14 and FC 22–26)
    python -m soccer.model.sofifa_client

    # Or re-pull all of 07–26 from one source for full consistency
    python -m soccer.model.sofifa_client --versions all

    # Sanity check first: one edition, 5 teams
    python -m soccer.model.sofifa_client --versions 26 --limit 5

It self-throttles under SoFIFA's 60 req/min limit and caches each squad to
`.cache/` (git-ignored) so an interrupted run resumes cheaply. A full
gap-fill is ~13 editions × ~180 national teams ≈ 40–60 min.

If an edition's launch roster (`{YY}0001`) is wrong, pass a roster override:
`--rosters rosters.json` with `{"26": "260012", ...}` (roster ids are on
SoFIFA's version-select dropdown).

**Attribution:** SoFIFA requires non-commercial use and a SoFIFA logo + link
on the consuming site's landing page — add that to `/vegas` if it surfaces
these ratings.

> Note: the API path builds each national side from its **actual in-game
> squad** (`/team/{id}/{roster}`), whereas the mirrored CSVs below use a
> **nationality pool**. Both feed the same top-25 depth / top-5 star metrics;
> for one consistent method across all editions, run `--versions all`.

## Manual upload (alternative)

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
