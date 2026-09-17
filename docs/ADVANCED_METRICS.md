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
shipping set (`base`: gap + economics + xG net + SoT net) and `full` =
exactly `train.FEATURES`: home Elo, away Elo, gap, economics, xG and SoT
form, every advanced Understat column, and the league / tier / season
context (44 inputs). Learners from `common/learners.py`, fixed
hyperparameters, same rows. Measured **after** the Understat backfill,
so npxG, xPts, PPDA and deep completions are real numbers on every row
2014-15 onward.

| scope | features | learner | log loss | Brier | vs round-1 model |
|---|---|---|---|---|---|
| all (n=7827) | base | logistic | 1.01749 | 0.60949 | — |
| all | base | random forest | 1.01744 | 0.60932 | +0.04 SE |
| all | base | boosting | 1.01990 | 0.61021 | −1.83 SE |
| all | full | logistic | 1.01634 | 0.60896 | +1.17 SE |
| all | **full** | **random forest** | **1.01453** | **0.60747** | **+2.36 SE** |
| all | full | boosting | 1.02764 | 0.61579 | −4.59 SE |
| top-5 (n=3649) | base | logistic | 0.98317 | 0.58545 | — |
| top-5 | full | logistic | 0.98072 | 0.58412 | +1.44 SE |
| top-5 | **full** | **random forest** | **0.97917** | **0.58281** | **+2.08 SE** |
| top-5 | full | boosting | 0.99033 | 0.58932 | −2.21 SE |

Top-5 by season, round-1 model → full forest: 2024-25 0.98042 → 0.97727
(+1.14 SE), 2025-26 0.98701 → 0.98209 (+1.76 SE), 2026-27 0.97006 →
0.96701.

By season, all leagues, round-1 model → full forest: 2024-25 1.01207 → 1.01111
(+0.53 SE), **2025-26 1.01603 → 1.00924 (+3.20 SE)**, 2026-27 1.00132 →
0.99497 (+0.93); the two small MLS slices are a wash. 2025-26 is the
first full season with the whole advanced layer live on both sides, and
it is where the gain is.

**What ships: the full set with a random forest** — now the measured
winner, not only the chosen one. Before the backfill (advanced columns
empty) the forest sat 0.0017 behind the logistic; with the columns
filled it is 0.0018 ahead of the logistic and 0.0030 ahead of the old
model. Boosting at the shared hyperparameters overfits and is not
shipped. The forest's own permutation importance on this window
(`/models`): `value_diff_z` +0.0137, `elo_gap` +0.0120, `elo_home_pre`
+0.0021, `elo_away_pre` +0.0016, `deep_share_diff` +0.0011,
`xpts_ewm_diff` +0.0011, `sot_net_diff` +0.0008, `npxg_for_ewm_diff`
+0.0003.

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

**One pooled model, not per-league sub-models.** Same split, full set,
logistic (7,827 test matches):

| model | log loss | vs pooled |
|---|---|---|
| pooled, full set | 1.01676 | — |
| pooled + league one-hots | 1.01705 | −0.57 SE |
| pooled + league + tier + season | **1.01651** | **+0.34 SE** |
| one sub-model per league (11 fits, scored on their own rows) | 1.02006 | worse in 9 of 11 leagues |

Per league, sub-model vs the pooled model on the same rows: only Serie B
(+2.36 SE) and La Liga 2 (+0.74) prefer their own model; La Liga (−2.21),
MLS (−1.94), EPL (−1.68), Ligue 1 (−1.58) and the rest are better served
by the pooled fit, because each sub-model sees a tenth of the data. So the
league, its tier and the season ride along as columns
(`train.CONTEXT_FEATURES`: `lg_<league>` one-hots, `tier`, `season_idx`)
that the forest can split on, and there is one model.

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

### Production decision (round 1, superseded)

Round 1 shipped Elo + adjusted success only. Round 2 below ships every
feature.

### Round 2 — every feature in, raw Elos, learner head-to-head

`PRODUCTION_FEATURES` is now every group above plus `elo_home_pre` and
`elo_away_pre` (41 inputs). Walk-forward as before, the three learners
from `common/learners.py` at fixed hyperparameters:

| window | model | log loss | Brier | vs Elo |
|---|---|---|---|---|
| clean 2024-25 (n=569) | Elo alone | 0.62411 | 0.21746 | — |
| clean | full set, logistic | 0.61691 | 0.21433 | +1.17 SE |
| clean | **full set, random forest** | **0.61687** | **0.21432** | **+1.16 SE** |
| clean | full set, boosting | 0.63012 | 0.21960 | −0.60 SE |
| clean | Elo + raw Elos + success, logistic | 0.61824 | 0.21491 | +1.54 SE |
| selection 2015-23 (n=2449) | Elo alone | 0.63714 | 0.22316 | — |
| selection | full set, logistic | 0.63600 | 0.22257 | +0.41 SE |
| selection | full set, random forest | 0.63723 | 0.22325 | −0.03 SE |
| selection | full set, boosting | 0.65656 | 0.23061 | −3.93 SE |
| selection | Elo + raw Elos + success, logistic | 0.63297 | 0.22126 | +2.49 SE |

**What ships: the full set with a random forest.** It ties the logistic
on the clean window (0.61687 vs 0.61691 — a gap one twentieth of the
seed noise floor, so read them as indistinguishable) and beats Elo by
+1.16 SE; the
compact success-only set is still the best *pooled* number, but the
brief is every feature in, and the forest gives that set away nothing on
the honest window. Boosting at the shared hyperparameters overfits 6k
rows badly (worse than Elo alone on 2015-23) and is not shipped.

**Elo level effect, NFL.** Empirical home-win rate 2002-2025 by home Elo
× rating difference:

| home Elo | diff < −75 | −75..0 | 0..75 | diff > 75 |
|---|---|---|---|---|
| < 1425 | 0.334 (n=917) | 0.477 (277) | 0.612 (129) | 0.591 (22) |
| 1425–1500 | 0.389 (578) | 0.512 (572) | 0.614 (422) | 0.721 (240) |
| 1500–1575 | 0.396 (235) | 0.506 (427) | 0.611 (550) | 0.743 (553) |
| > 1575 | 0.400 (35) | 0.584 (197) | 0.612 (366) | **0.789 (980)** |

The same edge converts more often the better the favourite is; the Elo
curve alone gives every row of a column the same number.

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

### Production decision (round 1, superseded)

Round 1 shipped Elo + core adjusted EPA only. Round 2 below ships every
feature.

### Round 2 — every feature in, raw Elos, learner head-to-head

`PRODUCTION_FEATURES` is now every group above plus `elo_home_pre` and
`elo_away_pre` (33 inputs). Same fixed split (fit 2005-2023, test
2024-2025):

| scope | model | log loss | Brier | vs Elo |
|---|---|---|---|---|
| overall (n=1853) | Elo alone | 0.49768 | 0.16831 | — |
| overall | full set, logistic | 0.49426 | 0.16628 | +1.05 SE |
| overall | **full set, random forest** | **0.49347** | **0.16523** | **+1.12 SE** |
| overall | full set, boosting | 0.49595 | 0.16691 | +0.41 SE |
| overall | Elo + raw Elos + core, random forest | 0.49159 | 0.16515 | +1.73 SE |
| FBS-vs-FBS (n=1606) | Elo alone | 0.55358 | 0.18894 | — |
| FBS-vs-FBS | full set, logistic | 0.54766 | 0.18639 | +1.59 SE |
| FBS-vs-FBS | **full set, random forest** | **0.54657** | **0.18509** | **+1.62 SE** |
| FBS-vs-FBS | full set, boosting | 0.54967 | 0.18708 | +0.82 SE |
| FBS-vs-FBS | Elo + raw Elos + core, random forest | 0.54469 | 0.18519 | +2.22 SE |

**What ships: the full set with a random forest** — the best of the three
learners on the full set in both scopes, though its margin over the
logistic (0.0008) is under two seed standard deviations, so only the win
over boosting is a real separation; see the seed spread below. The compact core set is still a
little better in absolute terms (its extra columns cost about 0.002), but
the brief is every feature in, and the forest is the learner that loses
least to them.

**Elo level effect, CFB** (FBS-vs-FBS, empirical home-win rate by home
Elo × rating difference):

| home Elo | < −100 | −100..0 | 0..100 | 100..250 | > 250 |
|---|---|---|---|---|---|
| < 1350 | 0.242 (n=2767) | 0.532 (930) | 0.630 (776) | 0.751 (586) | 0.886 (114) |
| 1350–1500 | 0.298 (1210) | 0.520 (793) | 0.629 (726) | 0.781 (904) | 0.909 (525) |
| 1500–1650 | 0.326 (540) | 0.493 (635) | 0.663 (763) | 0.766 (1036) | 0.941 (922) |
| > 1650 | 0.330 (106) | 0.488 (217) | 0.687 (355) | 0.779 (616) | **0.947 (1074)** |

Smaller than the NFL and soccer effects but in the same direction at the
top of the scale; the raw ratings are inputs so the forest can use it.

- Fallback rules are unchanged: the weekly table stale by more than two
  weeks, or an FCS side, or a side without a strength row, is Elo.
- The rest-of-season simulation and the expected score are still Elo only.

---

## Research harness: tuning, calibration, seed stability

`research/learner_lab.py` answers two questions the development sandbox
is too small to answer well. Both cache each sport's feature table under
`research/.cache/` (gitignored).

```
python -m research.learner_lab tune      [--sport soccer|nfl|cfb|all]
python -m research.learner_lab calibrate [--sport ...] [--seeds 20]
```

**tune** grid-searches each learner family on a validation window that
ends before the test window (soccer 2023-24, CFB 2023, NFL 2020-2023),
then scores only the per-family winners on the test window, once. Every
published learner comparison above ran on one shared, unsearched set of
hyperparameters, so "boosting overfits" is really "boosting at those
defaults overfits" until this has run.

**calibrate** reports, for the shipped learner: the spread of test log
loss across seeds, a reliability table, and whether isotonic or sigmoid
calibration fit on the validation window helps. The calibrated rows are
compared against the *same* inner model (fit on the pre-validation
window), not against the shipped one, so calibration is not charged for
the season it has to hold out.

### Results — all three sports

**Seed spread is the scale to read learner gaps against.** Refitting each
shipped forest across seeds, on its own test window:

| sport | seeds | test log loss range | sd |
|---|---|---|---|
| soccer | 8 | 1.01411 – 1.01470 | 0.00018 |
| CFB | 20 | 0.49260 – 0.49438 | 0.00050 |
| NFL | 20 | 0.61642 – 0.61947 | 0.00087 |

Against that scale, the round-2 learner claims come out differently by
sport:

- **Soccer holds.** Forest 1.01453 vs logistic 1.01634 is 0.0018 — about
  **ten seed standard deviations**. The forest genuinely beats the
  logistic there, and the +2.36 SE gain over the round-1 model stands.
- **NFL was a tie and is a tie.** Forest 0.61687 vs logistic 0.61691 is
  0.00004, one twentieth of the noise floor. Report them as
  indistinguishable. The gain over Elo alone (0.62411 → 0.61687) is
  about eight seed sd and is real.
- **CFB overstated it.** Forest 0.49347 vs logistic 0.49426 is 0.0008 —
  under two seed sd. On CFB the two learners are indistinguishable and
  only the win over boosting stands. The forest-vs-Elo gain (+1.1 SE
  overall, +1.6 FBS-vs-FBS) is a paired test over 1,853 games and is
  unaffected.

**Reliability.** Predicted vs actual by bucket, shipped model, test
window:

| bucket | soccer (n, gap) | CFB (n, gap) | NFL (n, gap) |
|---|---|---|---|
| 0.2-0.3 | 1027, +0.016 | 134, +0.014 | 29, −0.043 |
| 0.3-0.4 | 1945, +0.001 | 143, +0.006 | 87, +0.067 |
| 0.4-0.5 | 1942, +0.007 | 176, +0.002 | 132, +0.019 |
| 0.5-0.6 | 1479, +0.003 | 248, +0.014 | 87, +0.032 |
| 0.6-0.7 | 686, +0.006 | 250, +0.013 | 104, −0.004 |
| 0.7-0.8 | 328, −0.035 | 232, −0.013 | 75, +0.029 |
| 0.8-0.9 | 79, +0.032 | 263, **−0.051** | 55, −0.067 |
| 0.9-1.0 | 8, −0.094 | 309, **−0.029** | — | 

(positive = the model predicted higher than it happened)

Soccer is well calibrated where its mass is: every bucket from 0.3 to
0.7 holds 7,500 of its 7,827 matches and is within 0.007. **CFB is the
one real miscalibration**: 3-5 points under-confident on favourites over
572 games in the top two buckets — leaf frequencies cannot reach 1.0,
the usual forest ceiling. NFL's buckets hold 29-132 games each, where a
±0.06 gap is inside binomial noise; nothing to conclude there.

**Post-hoc calibration does not help anywhere.** Against the same inner
model (so calibration is not charged for the season it holds out):

| sport | inner | isotonic | sigmoid |
|---|---|---|---|
| soccer | 1.01625 | 1.02064 (−2.07 SE) | 1.01699 (−1.70 SE) |
| CFB | 0.49561 | 0.53033 (−2.31 SE) | 0.50032 (−2.05 SE) |
| NFL | 0.61558 | 0.63579 (−0.98 SE) | 0.61626 (−0.23 SE) |

Every cell is worse or neutral. One validation season is not enough to
fit a calibrator. If CFB's tail is worth correcting it needs a
calibrator fit across several seasons by cross-validation, or a learner
with better tail probabilities — which is what `tune` may find.


