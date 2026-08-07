# NFL 2025 season inventory

_Generated 2026-07-31 02:48 UTC by `python3 -m NFL.inventory.audit --season 2025 --write`._

## 1. Results

- **285 of 285** scheduled 2025 games have final scores.
- Regular season: 272 games; postseason: 13.
- Season ran 2025-09-04 to 2026-02-08.

## 2. Odds capture

- **589** snapshot files on disk, 589 readable, 23,846 total rows.
- Snapshots span 2025-10-04 to 2026-07-30, covering **477** distinct games across all seasons.
- Of the 285 2025 games, **202 (71%)** were captured, at a median of **40** snapshots each.
- Spread *points* are present for only **1** 2025 game(s): the legacy schema stored the juice on each side but never the number itself, so this season's line movement is **moneyline-only**.
- Weeks with nothing captured: **1, 2, 3, 4, 17, 21**. Weeks 1-4 predate the first snapshot; the later gaps are missed runs of the fetch job.
- 83 games have no odds record at all.

### Schema generations

| generation | files | first_pull | last_pull  | col_counts                 | game_id | spread_points |
|------------|-------|------------|------------|----------------------------|---------|---------------|
| legacy     | 332   | 2025-10-04 | 2026-01-15 | 55, 61, 67                 | False   | False         |
| current    | 256   | 2026-02-01 | 2026-07-30 | 28, 34, 40, 52, 64, 70, 76 | True    | True          |

### Capture by week

| week | games | captured | median_snapshots | captured_pct |
|------|-------|----------|------------------|--------------|
| 1    | 16    | 0        |                  | 0.0          |
| 2    | 16    | 0        |                  | 0.0          |
| 3    | 16    | 0        |                  | 0.0          |
| 4    | 16    | 0        |                  | 0.0          |
| 5    | 14    | 13       | 5.0              | 0.93         |
| 6    | 15    | 15       | 32.0             | 1.0          |
| 7    | 15    | 15       | 47.0             | 1.0          |
| 8    | 13    | 13       | 46.0             | 1.0          |
| 9    | 14    | 14       | 46.0             | 1.0          |
| 10   | 14    | 14       | 46.0             | 1.0          |
| 11   | 15    | 15       | 44.0             | 1.0          |
| 12   | 14    | 14       | 44.0             | 1.0          |
| 13   | 16    | 16       | 40.0             | 1.0          |
| 14   | 14    | 14       | 40.0             | 1.0          |
| 15   | 16    | 16       | 46.0             | 1.0          |
| 16   | 16    | 16       | 19.0             | 1.0          |
| 17   | 16    | 0        |                  | 0.0          |
| 18   | 16    | 16       | 12.5             | 1.0          |
| 19   | 6     | 6        | 27.5             | 1.0          |
| 20   | 4     | 4        | 13.0             | 1.0          |
| 21   | 2     | 0        |                  | 0.0          |
| 22   | 1     | 1        | 23.0             | 1.0          |

## 3. Vegas' report card

This is the bar. Every number below is the closing line's own performance.

| metric                             | value             | note                                                |
|------------------------------------|-------------------|-----------------------------------------------------|
| games graded                       | 285               |                                                     |
| home teams SU                      | 152-132-1 (53.5%) |                                                     |
| closing favourites SU              | 187-97-1 (65.8%)  |                                                     |
| home teams ATS                     | 144-140-1 (50.7%) |                                                     |
| closing favourites ATS             | 138-146-1 (48.6%) |                                                     |
| dogs of 7+ ATS                     | 44-40 (52.4%)     | n=84                                                |
| overs                              | 148-137 (51.9%)   |                                                     |
| spread MAE (pts)                   | 9.67              | average miss of the closing spread vs actual margin |
| spread bias (pts)                  | 0.61              | positive = home teams beat the number on average    |
| total MAE (pts)                    | 10.42             |                                                     |
| total bias (pts)                   | 1.08              |                                                     |
| median |spread|                    | 4.5               |                                                     |
| games with moneyline               | 285               |                                                     |
| average hold (vig)                 | 4.28%             | book's built-in margin on the two-way moneyline     |
| closing ML Brier                   | 0.2104            |                                                     |
| closing ML predicted home win rate | 54.2%             |                                                     |
| closing ML actual home win rate    | 53.3%             |                                                     |

### Closing moneyline calibration

| bucket         | n  | predicted | actual | gap    |
|----------------|----|-----------|--------|--------|
| (0.0, 0.167]   | 3  | 0.128     | 0.0    | -0.128 |
| (0.167, 0.333] | 44 | 0.255     | 0.227  | -0.028 |
| (0.333, 0.5]   | 74 | 0.414     | 0.432  | 0.018  |
| (0.5, 0.667]   | 81 | 0.587     | 0.58   | -0.007 |
| (0.667, 0.833] | 60 | 0.742     | 0.683  | -0.059 |
| (0.833, 1.0]   | 23 | 0.874     | 0.957  | 0.083  |

### Did the line movement mean anything?

Opening vs closing no-vig moneyline for the 202 captured games:

| direction   | games | avg_move | closing_implied | home_win_rate | home_cover_rate | beat_closing_by |
|-------------|-------|----------|-----------------|---------------|-----------------|-----------------|
| flat        | 53    | 0.001    | 0.568           | 0.491         | 0.415           | -0.077          |
| toward away | 63    | -0.083   | 0.43            | 0.46          | 0.508           | 0.03            |
| toward home | 86    | 0.057    | 0.622           | 0.593         | 0.541           | -0.029          |

## 4. Our assets

| asset                             | status     | detail                                                                                                                                                                                                              |
|-----------------------------------|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| data/2023-2025W3.csv              | STALE      | per-team box scores end 2025-10-02 — never saw the 2025 season (v2 no longer depends on it)                                                                                                                         |
| data/advanced_stats_25-26.csv     | MISFILED   | contains NBA team ratings, not NFL — not used by any NFL model                                                                                                                                                      |
| data/schedules/nflverse_games.csv | CURRENT    | 285 completed 2025 games; through 2026 schedule                                                                                                                                                                     |
| NFL weekly pick logs              | EMPTY      | 8 placeholder files with no content: NFL/Week_1/Week_1_Results.md, NFL/Week_2/Week_2.md, NFL/Week_3/Week_3.md, NFL/Week_4/Week_4.md, NFL/Week_5/Week_5.md, NFL/Week_6/Week_6.md, NFL/Week_7/Week_7.md, NFL/Weeks.md |
| NFL/model/artifacts (v1)          | SUPERSEDED | single split, 144-game test set, trained on the stale box-score file                                                                                                                                                |
| NFL/model/v2 (win)                | CURRENT    | walk-forward, 4350 out-of-sample games                                                                                                                                                                              |
| NFL/model/v2 (ats)                | CURRENT    | walk-forward, 4254 out-of-sample games                                                                                                                                                                              |
| NFL/model/v2 (total)              | CURRENT    | walk-forward, 4318 out-of-sample games                                                                                                                                                                              |

### v2 model on 2025, out of sample

| target                   | games | accuracy | brier  | break_even_at_-110 |
|--------------------------|-------|----------|--------|--------------------|
| win (straight up)        | 284   | 0.6549   | 0.2156 | 0.5238             |
| ats (against the spread) | 284   | 0.493    | 0.2504 | 0.5238             |
| total (over/under)       | 285   | 0.5193   | 0.2494 | 0.5238             |
