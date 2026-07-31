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
