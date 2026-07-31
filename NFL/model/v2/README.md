# NFL model, v2

A rebuild of `NFL/model/` after the 2025 season inventory
(`NFL/inventory/INVENTORY_2025.md`) showed the v1 setup could not answer the
question the repo is named after.

## Why a rebuild

v1 had three problems, all of them structural rather than tuning:

1. **Stale spine.** It trained on `data/2023-2025W3.csv`, per-team box scores
   that stop on 2025-10-02. The model never saw the season we now have results
   for.
2. **One split, tiny test.** Train/val/test was a single date cut with a
   144-game test set. At that size a 57.9% ATS result is about one standard
   error from a coin flip — it read as a finding but wasn't one.
3. **No out-of-sample record.** Nothing scored unplayed games, so there was no
   ledger to check the model against.

v2 fixes the spine and the evaluation. It does **not** find an edge — see
Results.

## Design

**Spine:** `data/schedules/nflverse_games.csv` — every game from 1999 on, with
scores, closing spread, closing total, moneylines, rest days, roof, surface,
weather, and starting QBs. One row per game, home perspective. Seasons from
2002 (the current 32-team alignment).

**Features (45)** — `dataset.py`:

| group | what |
|---|---|
| market | closing spread, total, no-vig moneyline probability, book hold |
| ratings | MOV-adjusted Elo (`elo.py`), Elo-implied spread, Elo minus the market spread |
| form | 5-game rolling margin / PF / PA / ATS margin / total-vs-line, margin volatility |
| season | win %, point differential per game (expanding, resets each season) |
| schedule | rest, short week, off bye, division game, neutral site, primetime, playoff |
| context | travel miles between stadiums, indoor flag, temp, wind, QB change |

Every rolling and expanding feature is shifted one game. Elo is updated
strictly in chronological order, so a game's rating is always its pregame
rating.

**Targets:** `win` (home wins outright), `ats` (home covers, pushes dropped),
`total` (over, pushes dropped).

**Evaluation** — `train.py`: walk-forward, one retrain per season. For season
S the model fits on everything before S-1, Platt-calibrates on S-1, and scores
S. Nothing reported below is in-sample.

## Results

Out of sample, 2010-2025.

### Straight up (4,350 games)

| model | accuracy | log loss | Brier | AUC |
|---|---|---|---|---|
| **market (closing moneyline)** | **66.4%** | **0.610** | **0.211** | **0.720** |
| market + model, stacked | 66.6% | 0.613 | 0.213 | 0.716 |
| lgbm v2 | 64.3% | 0.628 | 0.219 | 0.694 |
| Elo alone | 64.4% | 0.633 | 0.221 | 0.688 |
| always home | 55.7% | 0.687 | 0.247 | 0.500 |

The stacked row is the one that settles it. Blend the model's logit with the
market's and refit each season: out of sample the blend is *worse* than the
market alone (0.613 vs 0.610). The weight the stacker puts on the model column
swings between +0.57 and -0.16 across recent seasons with no stability. After
the closing line has spoken, this model has nothing to add.

### Spread and total

| target | games | accuracy | break-even |
|---|---|---|---|
| ATS | 4,254 | 50.5% | 52.4% |
| total | 4,318 | 49.5% | 52.4% |

Coin flips, as the efficient-market null predicts.

### Flat-stake backtest

Bet whenever the model's edge over the market clears 2 points; -110 on spreads
and totals, actual closing price on moneylines.

| market | bets | win % | units | ROI |
|---|---|---|---|---|
| straight up | 3,570 | 45.2% | +6.5 | +0.2% |
| spread | 2,210 | 49.8% | -108.1 | -4.9% |
| total | 3,011 | 50.0% | -135.9 | -4.5% |

The spread and total ROIs sit right about where the vig says they should. The
moneyline's +0.2% over 3,570 bets is noise, not an edge — the standard error on
that ROI is 2.2%, so the result is a tenth of a standard error from zero.

### Top features (gain, SU model)

`spread_line` and `market_home_prob` are first and second by a factor of four
over everything else. The market is doing the work; the football features are
decoration on top of it.

## Running

```bash
# from the repo root
python3 -m NFL.model.v2.train --target all --start-season 2010 --save
python3 -m NFL.model.v2.predict --season 2026 --week 1 --write
```

Artifacts land in `NFL/model/v2/artifacts/`: boosters, Platt coefficients,
per-season metrics, backtests, feature importances, and the full out-of-sample
prediction files that `NFL/inventory/grade_season.py` grades.

## Known limitations

- **Closing lines only.** nflverse carries the closing number. Every result
  here is "can the model beat the *closing* line", which is the hardest version
  of the question. Beating the *opening* line is a different and more winnable
  test, and it needs opening numbers we have not captured yet (see below).
- **Future weeks share one rating.** Elo and rolling form advance only on
  played games, so scoring all of 2026 at once prices week 17 off
  end-of-2025 ratings. Re-run weekly.
- **Week 1 skews under-confident on home teams.** Season-to-date features are
  empty in week 1, and the model's probabilities sit systematically below the
  market's. Treat week 1 edges as untrustworthy.
- **No play-by-play.** EPA and success rate are the obvious missing inputs.
  They are also unlikely to change the conclusion, since the market prices them
  too.

## Where an edge would actually come from

Not from a better rating system — the closing line already contains one. The
two directions with a real prior:

1. **Beat the close, not the closer.** Snapshot openers and bet numbers that
   move your way. The 2025 capture starts a median of 11 days before kickoff,
   which is well after the real opener — and it captured no spread *numbers* at
   all under the legacy schema, only the juice. Both are fixable in the fetch
   job.
2. **Shop the books.** The snapshots already carry 8-10 books per game. Taking
   the best available number instead of the consensus is a known, mechanical
   edge and it does not require the model to be right about anything.

## Production scorecard

The rating engine is the **QB + talent adjusted Elo** from `elo_variants.py`
(`dataset.DEFAULT_ELO = "qb_talent"`), with a graceful fallback to plain Elo
when the roster/awards caches are absent. Picks model is a calibrated logistic
on ten features, recency half-life 6 seasons.

`python3 -m NFL.model.v2.scorecard --save` regenerates everything below.
Walk-forward out of sample, 3,018 games, 2015-2025:

| model | acc | AUC | log loss | Brier | McFadden R² | Efron R² | cal. slope | ECE | top-3 |
|---|---|---|---|---|---|---|---|---|---|
| **production (qb+talent Elo)** | **66.5%** | 0.7172 | 0.6140 | 0.2127 | 0.1079 | 0.1406 | 0.977 | 0.0131 | 77.9% |
| previous (base Elo) | 66.5% | 0.7168 | 0.6140 | 0.2128 | 0.1079 | 0.1404 | 0.981 | 0.0152 | 78.2% |
| market (closing line) | 66.1% | **0.7179** | **0.6121** | **0.2122** | **0.1106** | **0.1429** | 1.056 | 0.0181 | **79.6%** |
| qb+talent Elo alone | 65.2% | 0.6976 | 0.6316 | 0.2201 | 0.0823 | 0.1108 | 0.808 | 0.0336 | 76.2% |
| always base rate | 55.0% | 0.500 | 0.6882 | 0.2475 | 0.000 | 0.000 | 0.038 | — | 53.8% |

### Reading these

- **R² on a 0/1 outcome is not meaningful**, so two pseudo-R² are reported
  instead. *McFadden* = 1 − LL/LL_null, the log-likelihood improvement over an
  intercept-only model. *Efron* = 1 − Brier/Brier_null, the squared-error
  version (identical to the Brier skill score). Both say the model explains
  roughly 11-14% of what there is to explain, and the market explains slightly
  more.
- **Calibration slope** comes from refitting a logistic on the model's own
  logit. 1.0 is perfect; **0.977 means the model is very slightly
  overconfident**, the market at 1.056 is slightly *under*confident, and raw
  Elo at 0.808 is materially overconfident — which is why it gets Platt-scaled
  before use.
- **ECE** (expected calibration error) is the average gap between predicted and
  actual within probability bins. The model's 0.0131 is better than the
  market's 0.0181 — the one metric where we beat the books, and it reflects
  calibration, not discrimination.
- **AUC vs accuracy**: the model edges the market on accuracy (66.5% vs 66.1%)
  while losing on AUC (0.7172 vs 0.7179). It sorts games marginally worse but
  places the 0.5 cut marginally better. Neither gap is significant.

### The honest summary

Beating the closing line on log loss remains out of reach: 0.6140 vs 0.6121.
The flat-stake moneyline backtest returns +0.8% ROI over 1,813 bets with a
standard error of 2.6% — indistinguishable from zero.

**Wiring in the better Elo changed nothing downstream.** Standalone it is worth
0.0055 log loss (0.6316 vs 0.6374 for plain Elo); inside the model both land on
0.6140. The market features absorb it. It is kept because it is free, strictly
better on its own, and would matter if the pipeline ever prices against opening
lines.

### Why the feature list did not change

Regenerating the uniform ranking on the new Elo promoted `market_vig` and
`travel_miles` into the top ten, displacing `home_roll_margin` and
`away_ppg_diff_std`. Adopting that swap **costs 0.0014 of log loss**
(0.6154 vs 0.6140), so it was not adopted. Importance rank measures what each
feature contributes on its own; it does not identify the best-performing
*combination*. `PICKS_FEATURES` in `weekly_nfl_report.py` stays pinned, and
`scorecard.py` deliberately scores that pinned list rather than a freshly
ranked one.

### How much of that 66.5% is just "pick the favourite"?

Almost all of it. This is the check that matters most on a 66% headline.

| rule | accuracy | 95% CI | record |
|---|---|---|---|
| model (production) | **66.50%** | [64.81, 68.16] | 2007-1011 |
| pick the spread favourite | 66.07% | [64.38, 67.76] | 1994-1024 |
| pick the moneyline favourite | 66.14% | [64.41, 67.83] | 1996-1022 |
| always pick home | 54.97% | [53.18, 56.73] | 1659-1359 |

The model beats the one-line rule "take whoever Vegas favours" by **0.43
percentage points — 13 games out of 3,018 across eleven seasons.** The paired
bootstrap puts that gap at 95% CI [-0.10, +0.96], so it does not clear
significance (P(model better) = 0.94).

The reason is visible in the agreement rate: **the model picks the same side as
the spread favourite in 98.0% of games.** It disagrees 61 times in eleven
seasons. On those 61 it goes 37-24 (60.7%), which sounds good until you see the
interval — [47.5, 72.1] on 61 games.

Accuracy by spread size shows where the difference lives, and where it does not:

| \|spread\| | games | model | favourite rule | gap |
|---|---|---|---|---|
| 0-1.5 | 312 | 52.9% | 49.7% | +3.2 |
| 2-3.5 | 1,126 | 59.1% | 58.9% | +0.2 |
| 4-7 | 891 | 69.0% | 69.0% | 0.0 |
| 7.5+ | 689 | 81.4% | 81.4% | 0.0 |

On any game with a spread of four points or more the model is **identical** to
just taking the favourite — same pick, same record, every time. Its entire
contribution is in near-pick'em games, which is exactly where the market has
least to say and where a rating system should help most.

So the honest framing of the headline: **66.5% is the spread's accuracy, not
the model's skill.** The model's own contribution is a third of a percentage
point, concentrated in coin-flip games, and not statistically separable from
zero. That is not a failure of this particular model — it is the same
efficient-market result every other measurement in this repo has produced,
stated in the most direct way available.
