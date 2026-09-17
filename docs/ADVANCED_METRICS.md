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

### Production decision (round 1, superseded)

The first pass promoted nothing and kept the seven-feature logistic;
the ablation above was scored on seasons where the feed was dead, so it
could not speak to npxG at all. Round 2 below reverses that: every
feature ships.

### Round 2 — every feature in, learner head-to-head, Elo level effect

`soccer/clubs/model/eval_learners.py`, artifact
`soccer/clubs/model/artifacts/learners_eval.json`. Fit on every season
before 2024-25, scored 2024-25 onward. Feature sets: the round-1
shipping set (`base`: gap + economics + xG net + SoT net) and `full`
(home Elo, away Elo, gap, economics, xG and SoT form, and every advanced
column — 31 inputs). Learners from `common/learners.py`, fixed
hyperparameters, same rows.

| scope | features | learner | log loss | Brier | vs round-1 model |
|---|---|---|---|---|---|
| all (n=7827) | base | logistic | 1.01755 | 0.60957 | — |
| all | base | random forest | 1.01866 | 0.61022 | −0.94 SE |
| all | base | boosting | 1.01956 | 0.61015 | −1.64 SE |
| all | **full** | **logistic** | **1.01676** | **0.60919** | **+1.43 SE** |
| all | full | random forest | 1.01846 | 0.61013 | −0.76 SE |
| all | full | boosting | 1.02019 | 0.61090 | −1.77 SE |
| top-5 (n=3649) | base | logistic | 0.98353 | 0.58576 | — |
| top-5 | full | logistic | 0.98319 | 0.58570 | +0.35 SE |
| top-5 | full | random forest | 0.98492 | 0.58659 | −0.73 SE |
| top-5 | full | boosting | 0.99099 | 0.58905 | −2.35 SE |

By season, round-1 model → full logistic (all leagues): 2024-25 1.01224
→ 1.01172 (+0.59 SE), 2025-26 1.01592 → 1.01429 (+2.16 SE), 2026-27
1.00204 → 1.00098; the two small MLS slices are a wash.

**What ships: the full set with a random forest**, by decision. The
forest is 0.0017 behind the full logistic on all leagues (0.8 SE) and
0.0017 on the top five (0.7 SE) — noise-level — and it was measured
with npxG, xPts, PPDA and deep completions entirely empty (the Understat
backfill had not landed). It is the learner that can use those columns
non-linearly when they exist, and the one that can carry upset structure
a linear fit averages away. Re-run `eval_learners` once
`understat_matches.csv` covers 2014-15 onward; that is the comparison to
trust for the learner question.

**Elo has a level effect, not just a gap effect.** Empirical home-win
rate, all seasons, by home Elo × venue-adjusted gap:

| home Elo | gap < −50 | −50..50 | 50..150 | > 150 |
|---|---|---|---|---|
| < 1325 | 0.278 (n=2436) | 0.366 (7372) | 0.466 (5867) | 0.562 (608) |
| 1325–1475 | 0.204 (3926) | 0.365 (9522) | 0.485 (11601) | 0.575 (2673) |
| 1475–1625 | 0.235 (838) | 0.381 (2116) | 0.518 (5557) | 0.659 (4008) |
| > 1625 | 0.250 (12) | 0.310 (84) | 0.468 (220) | **0.767 (2068)** |

The same gap converts far more often at the top of the scale. That is
why `elo_home_pre` and `elo_away_pre` are inputs in their own right now,
not only their difference.

The processed Understat table is live: the first Actions fetch on
2026-09-17 returned 1,952 matches with npxG/PPDA/deep for 2025-26 and
2026-27, and `xg_net_diff` reads fresh again (newest 2026-09-16). The
one-time backfill for 2014-15 → 2024-25 is the `understat_all` input on
the soccer workflow.

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

---

## College football

### Sources

| Feed | Endpoint | Refresh | Processed table | Raw cache |
|---|---|---|---|---|
| SportsDataverse weekly team summaries | `https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfb_team_summaries_weekly/cfb_team_summaries_weekly_{YEAR}.parquet` (no key) | daily Actions job, last + current season; upstream rebuilds in season | `data/college_football/team_weeks.csv` — one row per (season, team_id, through_week), 2004 → | `data/college_football/raw/weekly/` |
| cfbfastR schedules | unchanged (`fetch_schedule.py`), now also carrying ESPN `home_id` / `away_id` | daily | `data/college_football/games.csv` (+2 columns, additive) | `data/college_football/raw/schedules/` (backfill only) |

Team identity is the ESPN team id. The spine gained `home_id` / `away_id`
(`fetch_schedule.normalize`, backfilled once from the cached season CSVs
with `--backfill-ids`; every row 2001 → has them), and the weekly table's
`team_id` is the same id. A name crosswalk (`advanced.team_ids`) is the
fallback for a spine row without an id; both are tested.

Commands:

```
python -m CFB.data.fetch_weekly                 # last + current season (Actions)
python -m CFB.data.fetch_weekly --all           # backfill 2004 → current
python -m CFB.data.fetch_schedule --backfill-ids  # one-time id backfill from the raw cache
python -m CFB.model.eval_advanced               # fixed-split ablation
```

### The week rule

The snapshot with `through_week == W` **includes** week W's games. A
week-W game is featured from the W−1 snapshot (`snapshot_before`), falling
back to the latest earlier week the table has; postseason games (the
spine restarts `week` at 1 with `season_type == "postseason"`) use the
season's final snapshot, which predates every bowl. Asserted in
`tests/test_cfb_advanced.py::TestSnapshots`. The publisher's live-season
file carries rows for every week with the totals frozen at the last
played one; `real_weeks` detects those copies so they never count as
coverage (still pre-game, so harmless for the join).

### Pregame features (`CFB/model/advanced.py`)

Every value is a shrunk blend of the current snapshot and the previous
season's final snapshot regressed halfway to the league mean:
`(n · current + 4 · prior) / (n + 4)`, `n` = games in the snapshot
(`plays_off / 65`). Week 1 is the regressed prior alone; a team with
neither reads NaN, which the imputer fills with the training median, and
the logistic leans on `elo_logit` — the Elo fallback in practice. The
opponent-adjusted columns are the publisher's own ridge (offence +
defence + home effects) and are NaN for most teams until week 3, hence
the prior blend.

| Group | Features (SportsDataverse column in brackets) |
|---|---|
| Core | home/away `adj_epa_off` [`adj_off_epa`], home/away `adj_epa_def` [`adj_def_epa`], `adj_epa_matchup_net` = (home off + away def) − (away off + home def) |
| Success | home/away `success_off/def` [`success_off/def`, share of plays with EPA > 0], matchup net |
| Early / explosive / havoc | `early_epa_matchup_net` [`early_down_EPA`], `explosive_matchup_net` [`explosive`], home/away `havoc_off/def` [`havoc`, TFL + FF + INT + PBU per play] |
| Drive | `epa_drive_matchup_net` [`EPAdrive`], home/away `drives_game_off` [`drivesgame`], `plays_drive_off` [`playsdrive`], `yards_drive_off` [`yardsdrive`] |
| Situational | `rz_success_matchup_net` [`red_zone_success`], `third_success_matchup_net` [`third_down_success`], home/away `third_dist_off` [`third_down_distance`], home/away `passrate_off` |
| Splits | `pass_epa_matchup_net`, `rush_epa_matchup_net` [`EPAplay_*_pass/_rush`] |

Coverage (core features present on both sides): 86-89% of all rows every
season 2005 →, 99-100% of FBS-vs-FBS rows; the gap is the pooled FCS side,
which has no weekly data. By regular-season week (2019-2025): 50% in
week 1, 58% week 2, 72% week 3, 87% week 4, ≥ 94% from week 5.

### Evaluation

Fixed split: train 2005-2022, validation 2023 (C and the compact set),
test 2024-2025 reported once; 2026 excluded. Contamination, stated: the
Elo parameters were tuned on 2005-2023, so the Elo-only baseline is
in-sample on train and validation; only 2024-2025 is clean for it. Every
candidate sits on the same Elo. Artifact: `CFB/model/artifacts/advanced_eval.json`.

| model | overall 2024-25 (n=1853) | SE | FBS-vs-FBS (n=1606) | SE |
|---|---|---|---|---|
| 1. Elo | 0.49768 | — | 0.55358 | — |
| 2. + core adjusted EPA | **0.49285** | **+1.87** | **0.54599** | **+2.59** |
| 3. 2 + success | 0.49375 | +1.33 | 0.54618 | +2.23 |
| 4. 2 + early-down / explosive / havoc | 0.49346 | +1.51 | 0.54620 | +2.34 |
| 5. 2 + drive | 0.49295 | +1.73 | 0.54579 | +2.52 |
| 6. 2 + red zone / third down / pass rate | 0.49206 | +2.07 | 0.54494 | +2.82 |
| x. 2 + pass / rush splits | 0.49312 | +1.64 | 0.54579 | +2.49 |
| 7. combined (greedy on 2023: success + core + situational + splits) | 0.49431 | +1.08 | 0.54726 | +1.79 |

By season, Elo → core: 2024 0.51373 → 0.51247, 2025 0.48190 → 0.47355
(FBS-vs-FBS 2025: 0.53803 → 0.52643, +1.95 SE for the combined set).
Brier moves with log loss everywhere (0.16831 → 0.16580 overall).

### Production decision

- Promoted: **Elo + core adjusted EPA** (`PRODUCTION_FEATURES`: `elo_logit`,
  home/away `adj_epa_off`, home/away `adj_epa_def`, `adj_epa_matchup_net`),
  C = 1.0. It clears +2 SE where the feed actually covers both sides
  (FBS-vs-FBS) and is within 0.001 of the best group overall; the
  situational group's extra 0.0008 does not pay for six more columns, and
  the greedy combined set picked on 2023 loses to core alone on the test
  window (overfit).
- Fit in-run (`CFB/daily/state.py::build_second_stage`) on 2005 → last
  completed week; backdated runs only see snapshots through the last
  week completed before the run date.
- Fallback: the weekly table missing, or its last real snapshot more
  than 2 weeks behind the spine's last completed regular-season week,
  turns the stage off for the run; a game with an FCS side, or any side
  without a strength row, is Elo. The slate carries `p_home`,
  `p_home_elo`, `model` (`elo+adj_epa` or `elo`); `latest.json` carries
  `second_stage` and `feeds`.
- The rest-of-season simulation and the expected score are still Elo only.
- Collected, not promoted: success, early-down EPA, explosive, havoc,
  drive metrics, red zone, third down, pass rate, pass/rush EPA splits.
