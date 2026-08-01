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
| **production (qb+talent Elo)** | **66.5%** | 0.7171 | 0.6140 | 0.2127 | 0.1078 | 0.1405 | 0.978 | **0.0126** | 78.1% |
| previous (base Elo) | 66.5% | 0.7168 | 0.6140 | 0.2128 | 0.1079 | 0.1404 | 0.981 | 0.0152 | 78.2% |
| market (closing line) | 66.1% | **0.7179** | **0.6121** | **0.2122** | **0.1106** | **0.1429** | 1.056 | 0.0181 | **79.6%** |
| qb+talent Elo alone | 65.2% | 0.6972 | 0.6319 | 0.2202 | 0.0818 | 0.1102 | 0.806 | 0.0339 | 76.2% |
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
  actual within probability bins. The model's 0.0126 is better than the
  market's 0.0181 — the one metric where we beat the books, and it reflects
  calibration, not discrimination.
- **AUC vs accuracy**: the model edges the market on accuracy (66.5% vs 66.1%)
  while losing on AUC (0.7171 vs 0.7179). It sorts games marginally worse but
  places the 0.5 cut marginally better. Neither gap is significant.

### The honest summary

Beating the closing line on log loss remains out of reach: 0.6140 vs 0.6121.
The flat-stake moneyline backtest returns +0.5% ROI over 1,810 bets with a
standard error of 2.6% — indistinguishable from zero.

**Wiring in the better Elo changed nothing downstream.** Standalone it is worth
0.0052 log loss (0.6322 vs 0.6374 for plain Elo); inside the model both land on
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
bootstrap puts that gap at 95% CI [-0.07, +0.93], so it does not clear
significance (P(model better) = 0.95).

The reason is visible in the agreement rate: **the model picks the same side as
the spread favourite in 98.0% of games.** It disagrees 59 times in eleven
seasons, going 36-23 (61.0%) on them — which sounds good until you see the
interval on 59 games.

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

## Can we make a better spread? (`margin.py`)

Everything else here predicts *whether* the home team wins. `margin.py`
predicts *by how much* — the same question a bookmaker answers when it opens a
number, and the only framing where "we made a better line" is testable.

The headline model is **blind**: every market feature is removed (spread,
total, moneyline probability, vig, elo-vs-spread), so it has to construct a
line from Elo, form, rest, travel, weather and roster quality alone.

Walk-forward 2015-2025, 3,028 games, predicting the actual margin:

| line | MAE | RMSE | R² | bias |
|---|---|---|---|---|
| **closing spread (Vegas)** | **9.81** | **12.72** | **0.193** | +0.05 |
| model, sees the market | 9.90 | 12.77 | 0.186 | -0.24 |
| model, blind to the market | 10.06 | 12.97 | 0.161 | -0.39 |
| always pick'em (0) | 11.12 | 14.29 | -0.018 | +1.90 |

R² *is* meaningful here, unlike on the binary win target. The honest read:
**a line built with no knowledge of the market lands within 0.26 points of
Vegas's MAE** — closer than expected, and it explains 16.1% of margin variance
against the closing line's 19.3%. It correlates 0.886 with the closing spread
and disagrees by 3+ points on 28.4% of games.

So: no, we cannot make a better spread. But we can make a *nearly as good* one
from scratch, which is a more interesting result than it sounds — it means the
gap to Vegas is small and specific rather than diffuse.

### Betting the disagreement

| min disagreement | bets | ATS | vs break-even | z |
|---|---|---|---|---|
| 0 pts | 2,953 | 51.9% | -0.5 | -0.51 |
| 2 pts | 1,363 | 51.1% | -1.3 | -0.97 |
| 4 pts | 457 | 52.1% | -0.3 | -0.13 |
| 6 pts | 124 | 58.9% | +6.5 | 1.45 |

Nothing clears break-even except the 6+ bucket, which is 124 games and z=1.45.
Treat it as a hypothesis, not a strategy.

### Is the closing spread shaded?

The premise that a spread is set to split money evenly is the textbook story,
not the practice — books knowingly run unbalanced positions because bettors
overbet favourites and popular teams, and shading into that flow earns more
than balancing would. If that happens here it should be visible as a non-zero
average of (actual margin − spread) within a bucket:

| side | spread size | games | avg margin vs spread | z | fav cover % |
|---|---|---|---|---|---|
| away favourite | 0-3 | 526 | +1.24 | **2.07** | 48.3% |
| home favourite | 10.5+ | 226 | +1.32 | 1.60 | 54.4% |
| home favourite | 0-3 | 610 | -0.86 | -1.71 | 44.8% |
| away favourite | 3.5-7 | 439 | -0.99 | -1.65 | 49.7% |

The suggestive cell is small away favourites: they underperform the number by
1.24 points on average (z=2.07), i.e. the home dog has been the side. **But
eight buckets were tested, and the largest |z| among eight independent draws
from noise is typically 1.9-2.3 — so this is exactly what chance produces.**
The corresponding cover rate (backing the dog at 51.7%) does not clear the
52.4% break-even anyway.

### Where this goes next

The blind line is the useful artefact, not the bet signal. Its value is that it
is *market-independent*, which makes it the right tool for the one direction
with a real prior: pricing against **opening** lines rather than closing ones.
That needs opening numbers the fetch job does not yet capture early enough.

## The market-blind spread model (`spread_model.py`)

`margin.py` asked whether a line built without the market could match Vegas.
This is the full treatment: six model families on one harness, walk-forward
with a calibration step, and KPIs built around the only question that matters —
**is our number closer to the final margin than the bookmaker's?**

All models see **54 features**: the 40 non-market columns plus the 14
squad-quality ones (draft capital, All-Pro / Pro Bowl / Top 100 counts, prior
season QB EPA, interim-coach flags). `spread_line`, `total_line`,
`market_home_prob`, `market_vig` and `elo_vs_spread` are removed, so nothing
can copy the answer.

> The first run of this bake-off used 40 features, not 54. `blind_features()`
> was derived from `FEATURE_COLS` alone, so `build_dataset(with_squad=True)`
> merged the roster columns into the frame and `feature_matrix` then dropped
> every one of them. Fixed, re-run, and pinned by `TestBlindFeatureSet`. The
> conclusions below did not change — see "what the squad features bought".

### Bake-off, walk-forward 2015-2025, 3,028 games

| model | MAE | RMSE | R² | bias | cal. slope | closer than Vegas |
|---|---|---|---|---|---|---|
| **closing spread (Vegas)** | **9.81** | **12.72** | **0.193** | +0.05 | 1.038 | — |
| extra trees (uncalibrated) | 10.08 | 13.02 | 0.155 | -0.46 | 1.129 | 45.4% |
| extra trees | 10.09 | 13.01 | 0.156 | **-0.03** | 0.960 | 45.2% |
| ridge | 10.10 | 13.01 | 0.156 | -0.07 | 0.968 | 45.4% |
| random forest | 10.10 | 13.06 | 0.150 | -0.04 | 0.951 | 45.0% |
| ridge (uncalibrated) | 10.11 | 13.01 | 0.156 | -0.45 | 0.916 | 45.8% |
| xgboost | 10.12 | 13.11 | 0.143 | -0.13 | 0.960 | 46.3% |
| huber | 10.18 | 13.23 | 0.127 | -0.09 | 0.872 | 45.7% |
| lightgbm | 10.22 | 13.20 | 0.131 | -0.11 | 0.937 | 45.5% |

Extra trees now edges ridge by **0.009 points of MAE** — three thousandths of a
percent, and the two swap places depending on which features are in the set.
Read that as a tie, not a win. The honest statement is the one the classifier
already made: **every model family lands within 0.13 points of every other**,
LightGBM is reliably last, and nothing an ensemble finds is worth the variance
it adds. With ~3,000 games and features this collinear there is no structure
for a boosted tree to exploit that a linear fit misses.

Our best line misses by **10.09 points against Vegas's 9.81** — a quarter of a
point — and explains 15.6% of margin variance against their 19.3%.

### What calibration bought

Every uncalibrated model carries a bias near −0.5 points (systematically too
low on home teams) and a slope away from 1. The linear recalibration on the
prior season fixes both — extra trees goes from bias −0.46 / slope 1.129 to
−0.03 / 0.960. It does **not** improve MAE (10.083 → 10.092, marginally worse).
Worth knowing: calibration buys you an unbiased line, not a more accurate one.

### The result that settles it

| disagreement | games | our MAE | Vegas MAE | we're closer | ATS on our side |
|---|---|---|---|---|---|
| 0-1 pts | 832 | 9.72 | 9.69 | 47.1% | 48.8% |
| 1-2 pts | 692 | 9.92 | 9.79 | 44.8% | 48.2% |
| 2-3 pts | 514 | 9.38 | 9.27 | 47.7% | 53.4% |
| 3-5 pts | 699 | 10.33 | 9.87 | 42.9% | 50.7% |
| 5+ pts | 291 | 12.23 | 10.98 | **42.3%** | 51.8% |

**Confidence is anti-correlated with correctness.** Vegas is closer in every
single bucket, and the gap widens as our line strays further from theirs: our
MAE climbs from 9.72 to 12.23 while Vegas holds between 9.3 and 11.0.

That is the opposite of a tradable signal. A useful model would be *no better*
than Vegas where they agree and *better* where they disagree; ours is worse
exactly where it is loudest. Its disagreements are noise, and the 49% of games
where we differ by 2+ points are where that noise lives.

No ATS bucket clears the 52.4% break-even by anything close to its standard
error (~3.0 points on 291-514 games), the largest reading sits in the middle of
the range rather than at the confident end, and the buckets do not order
consistently. Not a finding.

### What actually predicts a football game

This is the first ranking in the project not swamped by the market. Full charts
in `artifacts/spread/importance/`; the top of the uniform ranking across all six
models:

| rank | feature | note |
|---|---|---|
| 1 | `roll_margin_diff` | 5-game scoring-margin differential — first for both linear models |
| 2 | `elo_diff` | first for both boosted models |
| 3 | `elo_home_prob` | first for extra trees |
| 4 | `away_roll_pf` | |
| 5 | `away_ppg_diff_std` | season-to-date point differential |
| 6 | `home_roll_margin` | |
| 7 | `home_ppg_diff_std` | |
| 8 | `elo_spread` | first for random forest |
| 14 | `n_probowlers_diff` | **top squad feature** — 8th for RF, 7.5th for extra trees |
| 15 | `away_pct_drafted` | |

Recent scoring margin and Elo are the whole story, and they are near-duplicates
of each other. Every model family picks one of the two as its top feature and
the other close behind. Everything below rank 8 is worth less than a fifth of
the leader.

### What the squad features bought

Nothing measurable, but the *way* they fail is informative. MAE moved from
10.089 to 10.092 for ridge and 10.107 to 10.092 for extra trees — noise in both
directions. Where they land in the ranking splits hard by model family:

| feature | ridge | huber | RF | extra trees | XGB | LGBM |
|---|---|---|---|---|---|---|
| `n_probowlers_diff` | 30.5 | 43.0 | **8.0** | **7.5** | 17.5 | 20.0 |
| `allpro_score_diff` | 40.5 | 38.0 | 27.0 | **10.5** | 16.5 | **9.0** |
| `n_top100_diff` | 33.0 | 26.0 | 14.5 | **11.0** | 38.0 | 36.0 |
| `n_top2_rounders_diff` | 37.5 | 36.5 | 13.0 | 24.5 | 14.0 | 14.5 |

The linear models bury all four in the bottom third; the tree models rank them
in the top 15, and Pro Bowl count reaches the top 8 for both bagged families.
Roster talent is not additively separable from form and Elo — a stacked roster
matters *more* when the team is already good — and only the trees can express
that interaction. It still buys no accuracy, because Elo has already absorbed
the same information through results.

The QB features are the clearest casualty: `home_qb_epa_prior` ranks 45th of 54
and `qb_epa_prior_diff` 33rd. Prior-season QB EPA is stale by the time a season
starts, and current-season QB quality is exactly the thing the leakage rules
forbid us from using.

Which is the honest explanation for the quarter-point gap to Vegas: the
bookmaker knows about injuries, personnel and game-specific news that a
schedule-scores-and-accolades dataset does not contain, and that knowledge is
worth about 0.28 points of MAE. Adding better *historical* roster data does not
close it; only fresher information would.

## Is the predicted spread *efficient*? (`spread_efficiency.py`)

MAE says how wrong a forecast is. Efficiency says whether what remains is
**exploitable** — a forecast can be less accurate than a rival and still carry
information the rival lacks. Standard errors are clustered on `season-week`
throughout (236 clusters over 3,028 games); treating games in a week as
independent would overstate every t-stat below by roughly 3.5x.

### 1. Both forecasts are individually efficient

`margin = a + b * forecast`, efficient iff `(a, b) = (0, 1)`:

| forecast | a | b | t(b=1) | joint chi2 | p | efficient |
|---|---|---|---|---|---|---|
| ours (extra trees) | 0.048 | 0.960 | −1.02 | 1.06 | 0.587 | **yes** |
| closing spread | −0.024 | 1.038 | 0.97 | 0.99 | 0.609 | **yes** |

Our calibrated spread is a properly scaled, unbiased forecast. It is not too
timid, not too aggressive, and needs no shrinkage. That is a real result: the
number is usable as a number, whatever its accuracy.

### 2. The market encompasses us

`margin = a + b1 * ours + b2 * vegas`:

| | coefficient | SE | t | p |
|---|---|---|---|---|
| ours | 0.148 | 0.085 | 1.74 | 0.084 |
| vegas | 0.914 | 0.086 | 10.63 | <0.001 |

The optimal combination puts ~0.15 on us and ~0.91 on the close. Our weight is
**not significant at 5%**, though it is not comfortably zero either — this is
the one number in the project that has ever hinted at independent information,
and it is a hint, not a finding. Ridge gives the same answer (0.150, p=0.084),
so it is a property of the feature set rather than of one learner.

### 3. Diebold-Mariano: the market is decisively more accurate

| loss | mean difference | DM stat | p | verdict |
|---|---|---|---|---|
| absolute | +0.284 | 5.18 | <0.001 | **vegas better** |
| squared | +7.49 | 4.55 | <0.001 | **vegas better** |

The quarter-point MAE gap is not sampling noise. It survives week clustering at
five standard errors.

### 4. The edge is real-ish, and economically useless

Regressing the book's error on our disagreement gives **b = 0.110** (SE 0.084,
p = 0.19) — each point we differ from the close is worth about a ninth of a
point of expected edge. Converting that to a cover probability with a residual
SD of 12.72:

> break-even at −110 needs `Phi(b*x/sigma) = 52.38%`, which requires a
> **6.9-point disagreement**. Only **88 of 3,028 games** are ever that far off
> the line — and in the 7+ bucket we actually covered **42.7%**.

| disagreement | n | implied cover | actual cover | SE |
|---|---|---|---|---|
| 0-1 | 804 | 50.2% | 48.8% | 1.8% |
| 1-2 | 679 | 50.5% | 48.2% | 1.9% |
| 2-3 | 502 | 50.9% | 53.4% | 2.2% |
| 3-5 | 682 | 51.3% | 50.7% | 1.9% |
| 5-7 | 204 | 52.0% | 55.4% | 3.5% |
| 7+ | 82 | 53.0% | **42.7%** | 5.5% |

The edge model and reality agree on the shape until the tail, where they invert.
Nothing here clears 52.38% with the sample to back it.

### The bias that wasn't

Slicing residuals by **Vegas's** favourite size appeared to show a large,
significant defect: on games the book lined at 10+, the favourite beat our
number by 2.6 points (p = 0.003) while beating theirs by only 1.3.

That was an artifact of conditioning on the other forecast. Re-bucketing by our
**own** prediction dissolves it:

| our |forecast| | games | our number | vegas number | favourite residual |
|---|---|---|---|---|---|
| 0-3 | 1,055 | 1.49 | 3.04 | −0.12 |
| 3-7 | 1,162 | 4.82 | 4.76 | +0.22 |
| 7-10 | 479 | 8.35 | 7.58 | −0.26 |
| 10+ | 332 | 12.02 | 10.59 | −1.02 |

Our forecast is unbiased inside its own distribution, and the two distributions
are near-identical in spread (SD 3.55 vs 3.45). Selecting on `|vegas| >= 10`
picks games where the *book* is extreme; ours is lower there because we disagree
about **which** games are mismatches — only 192 of 370 overlap — not because we
compress the scale. Mincer-Zarnowitz, which conditions on nothing, said so
correctly all along.

Confirmed by construction: re-fitting the per-season calibration with tail-
expanding forms (quadratic, hinge at 7, hinges at 5 and 10) moves MAE from
10.093 to 10.080 and leaves the 10+ residual at +2.79 instead of +2.99. There
was no scale defect to fix.

**Conditioning on a rival forecast to diagnose your own is a trap.** Bucket by
your own prediction, or use a test that conditions on neither.

## Early-season ATS (`early_season.py`)

The one place in this project where the market looks beatable — and a good case
study in why "looks beatable" and "is beatable" differ.

### The effect

Backing **every underdog** in weeks 1-4, 2010-2025:

| slice | games | dogs cover | week-block 95% CI | ROI at -110 |
|---|---|---|---|---|
| weeks 1-4 | 990 | **54.6%** | [52.8%, 56.5%] | **+4.33%** |
| weeks 5+ | 3,079 | 50.0% | [48.7%, 51.3%] | −4.51% |

Break-even is 52.38%. The pooled bootstrap puts P(rate <= break-even) at 0.009.

### It survives the obvious robustness checks

**Not cherry-picked on the window.** Every cutoff from 2 to 6 weeks pays:

| weeks 1..k | 1 | 2 | 3 | **4** | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| dogs cover | 53.8% | 55.2% | 55.2% | **54.6%** | 54.1% | 54.1% | 53.6% | 52.8% | 52.4% |
| ROI | +2.7% | +5.4% | +5.4% | **+4.3%** | +3.2% | +3.3% | +2.2% | +0.7% | −0.0% |

**The decay has the right shape.** Per week, uncumulated: 53.8%, 56.6%, 55.2%,
52.9%, then noise around 50% and gone by week 9. That is what you would expect
if the mechanism is real — opening numbers lean on preseason priors and the
market needs results to correct them — rather than a spike in one arbitrary
bucket.

**It held out of sample.** Discovered on 2010-2017 (54.9%), it returned 54.4%
on 2018-2025 — through legalisation and a much sharper market.

### Why it is still not a green light

The pooled test treats 990 games as ~236 week-clusters. But **you do not deploy
this 990 times; you deploy it once a year.** At the season level:

| seasons | mean ROI | median | SD | t | p (one-sided) | winning | worst |
|---|---|---|---|---|---|---|---|
| 16 | +4.3% | +4.1% | 10.3% | **1.67** | **0.058** | 11 | 2025, −16.5% |

`t = 1.67` misses the one-sided 5% bar. Five losing seasons, a 10.3% ROI
standard deviation, and **the worst season in the sample is the most recent
one** — 2025 went 43.8%. The gap between p = 0.009 and p = 0.058 *is* the
finding: week-clustered resampling still understates the risk, because a whole
season can be a regime.

The breakdown by spread size is the other warning. If this were the classic
favourite-longshot bias it should strengthen with the number; instead it is
non-monotone (0-3: +10.3%, 3-6: −3.4%, 6-10: +7.9%, 10+: −6.2%), which is what
noise looks like.

### Neither model captures it

The ATS classifier goes **49.6%** in weeks 1-4 — no better than its 50.2%
full-season rate. The signal is sitting in the data unexploited.

The market-blind spread model *appears* to capture it at 53.3%, but it doesn't:

| slice | our line on the dog side | model ATS | naive dog rule |
|---|---|---|---|
| weeks 1-4 | 58.6% | 53.3% | **54.2%** |
| weeks 5-9 | 59.5% | 49.9% | 50.9% |
| weeks 10+ | 59.3% | 48.4% | 50.7% |

The blind line sits on the underdog side of the close ~59% of the time in
*every* part of the season. It has a constant dog tilt that pays only in the
weeks when the market's dog bias exists — and the naive rule beats it in all
three slices. That is attribution, not discovery.

### Verdict

A real, documented, decaying market bias that our models do not use, worth about
+4% ROI in a window of ~60 games a year, at t = 1.67. That is a hypothesis to
track forward with the opening-line capture, not a strategy to fund. If it is
genuine it should be **larger against openers than closers**, which is the
cleanest forward test available and needs no new modelling.
