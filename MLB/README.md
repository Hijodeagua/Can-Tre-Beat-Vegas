# MLB — Stage 1: the data

Game-results cache for the MLB model, 2005–present, with a weekly refresh
path. Built the way `data_jobs/` works elsewhere in this repo: fetch jobs are
idempotent, incremental, and run on a CI schedule; everything lands in
version-controlled CSV so a re-run never re-downloads history.

## What's cached

`MLB/data/games.csv.gz` — **50,259 games, 2005–2025** (49,492 regular season
+ 767 postseason), one row per game, 170 columns. Per game:

- date, teams (Retrosheet codes + a stable franchise key), home/away, final
  score, **innings** (parsed from the line score, so extras and shortened
  games are explicit), park ID, day/night, attendance, duration;
- **both starting pitchers** (Retrosheet player IDs + names), winning/losing/
  save pitchers, full starting lineups with positions, both managers;
- **home-plate umpire** (and the full crew) — strike-zone tendencies are a
  planned feature and the ID is the join key;
- complete per-team offensive lines (AB H 2B 3B HR RBI SH SF HBP BB IBB SO SB
  CS GIDP LOB), pitching lines (pitchers used, ER, wild pitches, balks) and
  defensive lines (PO A E PB DP TP) — enough to compute run differential,
  Pythagorean expectation, OPS, FIP and rolling pitcher/team form without
  another source.

`MLB/data/reference/` — Retrosheet `ballparks.csv` and `teams.csv`.
`MLB/data/raw/gamelogs/` — the raw season files, gzipped verbatim, so the
parsed table can always be rebuilt offline (`refresh --no-network`).
`MLB/data/coverage_report.md` — per-season coverage table, regenerated (and
asserted) on every refresh.

## Sources, and why these

Reachability was probed before planning (2026-08-05): retrosheet.org,
statsapi.mlb.com, Baseball Savant and FanGraphs are all blocked from the dev
session's egress policy; GitHub and PyPI are open, and GitHub Actions runners
have open internet.

| source | role | status |
|---|---|---|
| Retrosheet game logs via the [Chadwick Bureau mirror](https://github.com/chadwickbureau/retrosheet) | historical spine 2005–2025, incl. postseason | **live, pulled** |
| MLB Stats API (`MLB/data_jobs/statsapi.py`) | in-progress season, probable pitchers, weather, umpires | runs in CI (blocked from dev session) |
| The Odds API (`data_jobs/odds_api`, sport `mlb`) | moneyline / runline / totals snapshots | runs in CI, same job as NFL/NBA |
| Baseball Savant / FanGraphs (Statcast era) | xwOBA, barrels, EV, xERA, wOBA/wRC+/xFIP | **deferred** — needs a CI-side fetch job; see below |

The mirror carries the same files as retrosheet.org (seasons 1871–2025:
game logs, event files, rosters, umpires, schedules), is refreshed by the
Chadwick Bureau, and supports conditional GETs — a weekly refresh costs one
ETag check per season.

## Refresh paths

```bash
# Weekly (also: .github/workflows/mlb-data.yml, Sundays)
python -m MLB.data_jobs.refresh            # incremental; picks up the next
                                           # Retrosheet annual release itself
python -m MLB.data_jobs.refresh --force    # re-download everything
python -m MLB.data_jobs.refresh --no-network  # re-parse the cache only

# Daily in CI (statsapi.mlb.com is not reachable from dev sessions)
python -m MLB.data_jobs.statsapi --finals-days 3 --probables-days 7
```

The refresh **asserts coverage** (house rule 7) and fails loudly: per-season
game counts inside a ±6 window of the expected schedule (2430; 898 in 2020),
zero null scores/teams/parks, zero missing starting-pitcher IDs, unique game
IDs, and every Retrosheet team code resolving through the franchise
crosswalk. `tests/test_mlb_pipeline.py` re-checks the invariants plus two
parse-sanity gates: home-win rate must sit in [.52, .56] (it is **.5375**)
and the extra-innings share in [5%, 12%] (it is ~8.6%).

## Odds capture

The existing unified odds job now fetches `baseball_mlb` (moneyline, runline
as `spreads`, totals) on the same 2×-daily schedule, plus a 16:00 UTC
MLB-only top-up. Snapshots are timestamped, so **opening lines are captured
from day one**: books post most MLB lines overnight, the 10:00 UTC snapshot
is the as-captured opener, and the earliest snapshot per game ID
reconstructs it. Closing lines come from the last snapshot before first
pitch. Quota: ~7 requests/day ≈ 210/month of the 500 free tier.

Historical odds (2005–2024) are **not** backfilled — they need a paid source.
Decision deferred until the Elo baseline says what it's worth.

## Two-era plan

- **2005–2014**: traditional metrics only, all computable from this cache.
- **2015+**: Statcast layer (xwOBA, barrel rate, EV, xERA) to be joined on
  later from a CI-side Savant job. The expected-vs-actual gap becomes its own
  feature class. Nothing in the schema assumes the layer exists — it joins on
  date + team / pitcher ID.

## Leakage discipline notes (house rule 2)

- Probable pitchers are legal pregame information, but only as of their
  announcement. The statsapi job snapshots probables with a `pulled_at_utc`
  column on every row and keeps every snapshot, so a backtest can use
  "probable as of the morning" rather than "the guy who actually started".
  For the Retrosheet era the actual starter is the only record — a pitcher
  feature built on it must treat scratched starts as irreducible noise.
- Bullpen usage in the prior 2–3 days is knowable in advance and is on the
  feature list, but per-reliever innings need the Retrosheet **event files**
  (same mirror, `seasons/{year}/*.EV[NA]`, not yet cached — they're ~10× the
  size of game logs). Planned alongside times-through-the-order.
- Historical weather is not backfilled (no reachable source); the statsapi
  job records condition/temp/wind for every final going forward.

## Stage 2 preview

The Elo engine builds on `games.csv.gz` alone: MOV-adjusted with baseball-
damped margins, home advantage **fit** rather than ported from NFL,
season-to-season regression, and a pitcher-aware variant (team rating +
rolling starting-pitcher adjustment, FiveThirtyEight-style) compared against
the plain version under walk-forward evaluation — fit before S−1, calibrate
on S−1, score S, always reported against "bet the favourite" and the closing
line once odds accumulate.
