# Advanced metrics layer

Second-stage features on top of the Elo engines for club soccer, the NFL
and college football. The Elo engines are untouched; each sport gets a
raw fetch layer (cached, not committed), a processed table (committed),
a leakage-safe pregame feature builder, a chronological ablation, and a
freshness gate so a dead feed degrades the forecast to Elo instead of
feeding it stale numbers.

Rules that hold everywhere:

- **Pre-game only.** Every rolling stat is shifted so a match never sees
  its own result (`common/pregame.py`, tested in `tests/test_common_pregame.py`).
- **Chronological splits only.** Fit on earlier seasons, score later ones.
  Imputers, scalers and models are fit on training rows only
  (`common/evaluate.py` wraps them in one sklearn `Pipeline`).
- **Log loss and Brier decide.** Accuracy is printed because people ask.
  A feature group has to beat the baseline on the *same rows*; the number
  reported is the paired gain in standard errors (`paired_se`).
- **Freshness gate.** `common/freshness.py` reports each feed's newest
  row on the run date; a feed past its tolerance is logged STALE and the
  feature it drives is off for that run.
- **No paid APIs.** Everything below is a public file or an unauthenticated
  endpoint.

Regenerate any table here with the commands in each section. Shared code:
`common/pregame.py` (rolling / EWMA / shrink / rest), `common/freshness.py`,
`common/evaluate.py` (walk-forward, fixed split, paired test).

---

## Club soccer

### Sources

| Feed | Endpoint | Refresh | Processed table | Raw cache |
|---|---|---|---|---|
| Understat league data | `https://understat.com/getLeagueData/{league}/{season}` (headers: `User-Agent`, `X-Requested-With: XMLHttpRequest`, `Referer`), legacy `datesData`/`teamsData` HTML blobs as fallback | daily Actions job, current season only; backfill 2014-15 → on demand | `soccer/clubs/data/understat_matches.csv` (+ `xg_matches.csv` in the legacy 6-column shape) | `data/soccer_clubs/raw/understat/` |
| football-data.co.uk shots | unchanged (`fetch_shots.py`) | daily | `shots_matches.csv` | — |

The Understat host is blocked from the dev sandbox (proxy 403), so the
fetcher has only been exercised against fixtures. The daily workflow now
runs `python -m soccer.clubs.data.understat --probe epl <year>` before the
pipeline so the first Actions run prints the live response shape; the
parser accepts a JSON object, a JSON list, and the legacy HTML blob.

Per match the processed table keeps: xG, npxG, xPts, PPDA numerator and
denominator (attacking and defensive), deep completions, deep completions
allowed, goals. Shots and shots on target come from the football-data feed.

Commands:

```
python -m soccer.clubs.data.understat                  # last + current season refresh (Actions)
python -m soccer.clubs.data.understat --all            # backfill 2014-15 → current
python -m soccer.clubs.data.understat --probe epl 2025 # print the live response shape
python -m soccer.clubs.model.eval_advanced             # ablation, both scopes
```

### Pregame features (`soccer/clubs/model/advanced.py`)

Every value is a difference, home minus away, from each club's previous
league matches only. Rolling = mean of the last 10; EWMA = half-life 5
matches; both need 5 prior matches, and a club whose last match is more
than 130 days old reads NaN (imputed to the training median).

| Group | Features |
|---|---|
| xG form (EWMA) | `xg_for_ewm_diff`, `xg_against_ewm_diff`, `npxg_for_ewm_diff`, `npxg_against_ewm_diff`, `xg_net_ewm_diff` |
| xG form (rolling) | the rolling-10 versions of the above |
| Matchup | `home_att_vs_away_def` (home's home-only xG for vs away's away-only xG against), `away_att_vs_home_def`, `xg_per_shot_diff` |
| Territory | `deep_diff`, `deep_share_diff` (labelled a field-tilt *proxy*: deep completions / (deep for + deep against), not possession), `ppda_diff` (PPDA = opponent passes / defensive actions; lower = more pressing), `xpts_form_diff` |
| Rest | `rest_diff` (days since previous match), `congestion_diff` (matches in the previous 14 days), `uefa_diff` (European tie within 7 days), from the whole calendar including UEFA rows |

### Evaluation

Train < 2023-24, validation 2023-24 (picks C and the compact set), test
2024-25 → 2026-27, multinomial logistic on the shipping feature set plus
one group. Artifacts: `soccer/clubs/model/artifacts/advanced_eval_{all,top5}.json`.

Coverage on the test seasons (top-5 scope):

| season | rows | live xG form | npxG / territory | rest |
|---|---|---|---|---|
| 2023-24 | 1752 | 98.6% | 0% | 100% |
| 2024-25 | 1752 | 89.1% | 0% | 100% |
| 2025-26 | 1751 | 0% | 0% | 100% |
| 2026-27 | 146 | 0% | 0% | 98.6% |

npxG, xPts, PPDA and deep completions are 0% because the processed
Understat table cannot be fetched from the sandbox; the legacy xG file
only carries xG, and it ended on 2025-01-04.

Test, all leagues (n = 7,827) and top-5 only (n = 3,649):

| model | log loss (all) | SE vs base | log loss (top5) | SE vs base |
|---|---|---|---|---|
| base (Elo + economics + xG net + SoT net) | 1.01755 | — | 0.98353 | — |
| + xG EWMA | 1.01750 | +0.34 | 0.98360 | −0.18 |
| + xG rolling | 1.01762 | −0.47 | 0.98374 | −0.67 |
| + matchup | 1.01787 | −1.80 | 0.98439 | −2.07 |
| + territory | 1.01755 | +1.19 (all-NaN columns) | 0.98353 | −0.71 |
| + rest | 1.01739 | +0.59 | 0.98331 | +0.38 |
| combined | 1.01778 | −0.96 | = base | — |

Nothing clears +2 SE. That is the expected result of scoring a
chance-creation layer on seasons where the feed was dead for 57% of the
rows: the model learns coefficients on training seasons with live xG and
then meets imputed medians.

**Were the earlier xG results distorted by stale data?** Yes, in the
direction of hiding the signal, not inventing it. The staleness guard
zeroed `xg_net_diff` on 2,088 of 3,649 top-5 test matches. On all test
rows the feature is worth +0.35 SE; on the 1,561 matches where both clubs
had live form it is worth +1.78 SE (0.97941 vs 0.98013). The signal the
2023-24 validation found (+2.1 SE) is still there wherever the feed is.

### Production decision

- Promoted: nothing new. The shipping feature set is unchanged.
- The Understat fetcher is replaced (`fetch_xg.py` is a shim over
  `understat.py`), so once Actions lands one refresh `xg_net_diff` comes
  back on live slates through the existing 130-day guard.
- The daily run logs freshness for all three feeds and writes them to
  `web/public/data/soccer/latest.json` under `feeds`.
- Collected, not promoted: npxG, xPts, PPDA, deep completions, xG per
  shot, home/away attack splits, rest and congestion. Re-run
  `eval_advanced` after a season of live Understat rows before promoting
  any of them; the current numbers say nothing about them either way.

Leakage safeguards specific to this layer: form uses only rows dated
strictly before the match (`test_form_is_strictly_pre_match_and_warms_up`),
home splits only see home matches, rest and congestion come from the full
calendar (UEFA included) but only from dates before the match, and the
staleness guard is applied per row, per side.

---

## NFL

### Sources

| Feed | Endpoint | Refresh | Processed table | Raw cache |
|---|---|---|---|---|
| nflverse play-by-play | `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet` (nflfastR columns, no key) | daily Actions job, current season; nflverse rebuilds nightly in season | `data/nfl/team_games.csv` — one row per (game, team), 2002 → | `data/nfl/raw/pbp/` |

The schedule spine (`data/schedules/nflverse_games.csv`) is unchanged.
Team keys: the aggregates carry nflverse's current abbreviations (LA,
LAC, LV); the spine's historical ones (STL, SD, OAK) join through
`NFL.elo.teams.canonical`, tested in
`test_historical_abbreviations_join_through_the_franchise_map`.

Commands:

```
python -m NFL.data.pbp                    # backfill 2002 → current
python -m NFL.data.pbp --seasons 2020 2023
python -m NFL.data.pbp --current          # daily refresh (Actions)
python -m NFL.model.eval_advanced         # walk-forward ablation
```

### Team-game aggregates (`NFL/data/pbp.py`)

Play filter: `play_type in (pass, run)` with non-null EPA; special teams
plays are `special_teams_play == 1`. Dropbacks are `qb_dropback == 1`
(sacks and scrambles included); rushes are `rush == 1` and not a
dropback. "Neutral" = quarters 1-3 with the score within 8. Explosive =
run ≥ 12 yards or pass ≥ 20. Third-and-short ≤ 3, third-and-long ≥ 7.
Drives are `fixed_drive`; drive points are `posteam_score_post − posteam_score`
from first to last play. Every offence metric has a `def_` twin (what the
team allowed) and `st_epa_net` is special-teams EPA for minus against.

| Family | Columns (offence; `def_` twins exist) |
|---|---|
| Efficiency | `off_epa`, `off_success`, `off_plays` |
| Dropback | `off_dropbacks`, `off_db_epa`, `off_db_success`, `off_sack_rate` |
| Rush | `off_rushes`, `off_rush_epa`, `off_rush_success` |
| Early downs | `off_early_epa`, `off_early_success` |
| Tendencies | `off_pass_rate`, `off_proe` (mean `pass_oe`), `off_proe_neutral` |
| Explosive | `off_explosive_rush_rate`, `off_explosive_pass_rate`, `off_explosive_rate` |
| Third down | `off_third_att`, `off_third_conv`, `off_third_epa`, `off_third_dist`, `off_third_short_conv`, `off_third_long_conv` |
| Pace | `off_neutral_pace` (seconds between consecutive neutral plays of a drive) |
| Drives | `off_drives`, `off_epa_per_drive`, `off_points_per_drive`, `off_yards_per_drive`, `off_plays_per_drive`, `off_series_success` |
| Red zone | `off_rz_trips`, `off_rz_td_per_trip`, `off_rz_pts_per_trip`, `off_rz_epa` |
| Special teams | `st_epa_net` |

Definitions are asserted on a hand-built game in `tests/test_nfl_advanced.py`.
This is not DVOA and nothing here is labelled as such.

### Pregame features (`NFL/model/advanced.py`)

**Opponent-adjusted ratings** (`off_{m}_adj`, `def_{m}_adj`): per metric,
a weighted ridge over every offence-vs-defence observation before the
week being predicted, `y = off[o] + def[d] + hfa·home`. Recency
half-life 5 weeks; the previous season's games at half weight so Week 1
starts from a regressed prior; ridge 2.0 shrinks thin samples toward
league average. Recomputed per (season, week) from games with
`week < W` (`test_ratings_for_week_w_ignore_week_w_games`), so a Sunday
game never sees Thursday's result. Metrics: EPA, success, dropback EPA and
success, rush EPA and success, early-down EPA and success, explosive
rate, points and EPA per drive, red-zone TD per trip, third-down
conversion.

**Unadjusted form** (`{m}_ewm`): the team's own EWMA (half-life 5 games)
over previous games, shrunk toward league mean by games played this
season, shifted (`test_ewm_is_shifted_and_shrunk`).

**Matchups**: expected value of an offence against a defence is
`off + def` (def = allowed above average), `{m}_matchup_net` = home's minus
away's.

### Evaluation

Walk-forward, one refit per test season on 2002 → s−1, logistic on a
standardised, median-imputed design matrix. Ties dropped. Selection
window 2015-2023 (C, greedy group selection), clean window 2024-2025.
The Elo parameters were tuned on 2005-2023, so only the clean window is
honest for the Elo baseline; every candidate sits on the same Elo.
Artifact: `NFL/model/artifacts/advanced_eval.json`. Coverage of every
feature is 100% from 2003 (94% in 2002, the first season with no prior).

| model | selection 2015-23 (n=2449) | SE | clean 2024-25 (n=569) | SE |
|---|---|---|---|---|
| 1. Elo | 0.63714 | — | 0.62411 | — |
| 2. + core EPA (adjusted off/def EPA, matchup) | 0.63428 | +1.55 | 0.61988 | +1.05 |
| 3. + core EPA + success | 0.63351 | +1.81 | 0.61798 | +1.39 |
| 4. + core EPA + splits (dropback/rush/early-down, PROE, explosive) | 0.63532 | +0.84 | 0.61899 | +1.14 |
| 5. + core EPA + drive | 0.63514 | +0.95 | 0.61863 | +1.22 |
| 6. + core EPA + red zone / third down | 0.63516 | +1.01 | 0.62284 | +0.28 |
| x. + core EPA + pace / ST / sacks | 0.63384 | +1.54 | 0.61989 | +0.86 |
| x. + unadjusted EPA EWMA (no opponent adjustment) | 0.63463 | +1.87 | 0.62177 | +0.73 |
| 7. combined (greedy: success + pace/ST) | 0.63210 | +2.58 | 0.61859 | +1.23 |
| 7 as LightGBM | 0.64022 | −1.06 | 0.62843 | −0.74 |
| **Elo + success only** | **0.63275** | **+2.59** | **0.61853** | **+1.45** |

Pooled 2015-2025 (n=3018): Elo 0.63468; Elo + success 0.63007 (+2.96 SE,
Brier 0.22209 → 0.22002); Elo + core EPA + success 0.63058 (+2.24 SE);
Elo + success + pace/ST 0.62955 (+2.85 SE).

Standardised coefficients of the combined logistic (fit 2002-2023):
`elo_logit` +0.49, `success_matchup_net` +0.14, `home_off_success_adj`
+0.13, `away_off_success_adj` −0.11; everything else under 0.08.

### Production decision

- Promoted: **Elo + adjusted success rate** (`PRODUCTION_FEATURES`: `elo_logit`,
  home/away `off_success_adj`, home/away `def_success_adj`,
  `success_matchup_net`), C = 0.03. Smallest set that clears the bar
  pooled and is best on the clean window. Success rate beats EPA per
  play here because it is the less noisy of the two at 17 games.
- Fit in-run (`NFL/daily/state.py::build_second_stage`) on the replay
  history joined to the aggregates, 2002 → last completed week, ties
  excluded; backdated runs only see aggregates dated before the run date.
- Fallback: `data/nfl/team_games.csv` missing, or its newest row more
  than 10 days behind the spine's newest final, turns the stage off for
  the run and every forecast is Elo. A fixture whose side has no rating
  for (season, week) is Elo too. The slate carries `p_home` (shipped),
  `p_home_elo` and `model`; `latest.json` carries `second_stage` and
  `feeds`.
- The rest-of-season simulation and the expected score are still Elo
  only.
- Collected, not promoted: adjusted EPA, dropback/rush/early-down splits,
  PROE, explosive rates, drive metrics, red-zone and third-down metrics,
  pace, sack rates, special-teams EPA, the unadjusted EWMA twins. All are
  in the processed table and in `build_game_table`; none beat Elo +
  success on 2024-25. LightGBM on the same features loses to the
  logistic at this sample size.
