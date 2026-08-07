# NFL LightGBM Model (v1 — superseded)

> **Superseded by [`v2/`](v2/README.md).** This model trains on
> `data/2023-2025W3.csv`, which stops on 2025-10-02 and therefore never saw the
> 2025 season. Its headline "57.9% ATS" came from a 140-game test set — about
> one standard error from a coin flip. The v2 rebuild walks forward one retrain
> per season over 4,300+ out-of-sample games and finds no edge over the closing
> line. Kept here for provenance; `line_movement.py` is still live and shared.

Predictive model for the upcoming season. Two targets: straight-up
winner (`win`) and against-the-spread cover (`ats`).

## Layout

- `features.py` — rolling feature engineering + schedule merge
- `schedule.py` — nflverse games.csv loader (home/away, rest, weather, QB)
- `line_movement.py` — opening vs closing spread aggregator from
  `data/odds_api_data_*.csv` (not yet a model feature — see below)
- `train.py` — temporal split + LightGBM + baseline comparison
- `artifacts/` — saved model files, metrics CSVs, feature importances

## Running

```bash
cd NFL/model
python3 schedule.py --refresh   # one-time: pull latest schedule
python3 train.py --target win   # SU winner
python3 train.py --target ats   # cover the spread (drops pushes)
```

## Data flow

1. **Per-team stats** from `data/2023-2025W3.csv` — sorted by team+date,
   then 4-game rolling averages of off/def stats (shifted to avoid leakage).
2. **Opponent join** — opponent's rolling stats merged as `opp_roll_*`.
3. **Schedule merge** from cached nflverse `games.csv`:
   `is_home`, `rest_days`, `opp_rest_days`, `rest_diff`, `total_line`,
   `temp`, `wind`, `div_game`, `roof_*`, `qb_change`, `opp_qb_change`.
4. **QB change flag** — set when starter differs from the team's previous
   game.

Total: **79 features**.

## Temporal split

- Train: `< 2024-09-01`
- Val:   `2024-09-01` to `2025-01-15`
- Test:  `>= 2025-01-15`

## Latest results (with schedule features)

### Straight-up winner

| model               | n   | acc       | log_loss | brier     | auc       |
|---------------------|-----|-----------|----------|-----------|-----------|
| lgbm (val)          | 556 | **0.727** | 0.610    | 0.209     | 0.724     |
| market_logistic     | 556 | 0.714     | **0.593**| **0.202** | **0.757** |
| pick_favorite       | 556 | 0.714     | 1.324    | 0.280     | 0.714     |
| lgbm (test)         | 144 | 0.653     | 0.614    | 0.212     | 0.731     |
| market_logistic     | 144 | 0.688     | 0.599    | 0.205     | 0.753     |
| pick_favorite       | 144 | 0.688     | 1.446    | 0.306     | 0.687     |

LightGBM val accuracy improved from 71.2% → 72.7%, test AUC from 0.711 →
0.731 vs. the baseline (no-schedule) model. Still trails the 1-feature
market logistic on calibration — meaning the spread is doing most of the
work and the model isn't yet adding enough signal to beat it.

### ATS (cover)

| model               | n   | acc       | log_loss | brier | auc       |
|---------------------|-----|-----------|----------|-------|-----------|
| lgbm (val)          | 542 | 0.506     | 0.693    | 0.250 | 0.534     |
| lgbm (test)         | 140 | **0.579** | 0.691    | 0.249 | **0.612** |
| market_logistic     | 140 | 0.514     | 0.695    | 0.251 | 0.503     |

Test accuracy 57.9% (vs 57.1% baseline) and AUC 0.612 (vs 0.586). Still
small-sample on 140 games, so treat as suggestive not conclusive.

## Top features (gain) — SU model

1. `Spread_vg` — dominant
2. `roll_QBKD_ad` / `opp_roll_QBKD_ad` — QB knockdowns (pressure proxy)
3. `opp_roll_3D%_dn` / `roll_3D%_dn` — 3rd-down efficiency
4. `roll_TeamScore` / `opp_roll_TeamScore` — recent scoring
5. `temp`, `wind`, `total_line` — schedule features now ranking

`is_home` doesn't rank in top 20 because the spread already encodes
home-field advantage (~2-3 points).

## What's not yet a feature (and why)

- **Line movement** (open vs close spread/total). This section used to claim
  we had **zero historical games with line movement**. That was wrong, and the
  cause was a bug rather than a data gap: the aggregator keyed on `Game ID`,
  a column the legacy schema doesn't have, so it silently dropped all 332
  legacy files. Keying on `home|away|date` instead recovers **202 games of the
  2025 season** with full moneyline movement. The narrower true statement is
  that the legacy schema stored the *juice* on each side but never the spread
  *number*, so 2025 movement is moneyline-only.

  `line_movement.py` also now drops in-play snapshots. The fetch job sometimes
  caught a game after kickoff and stored live prices — a 3-point favourite at
  -2490 because it was already up two scores. Treating those as the closing
  line made "the line moved toward the winner" look 60% predictive when it was
  reading the scoreboard. Post-fix, movement does not beat the closing price;
  see `NFL/inventory/INVENTORY_2025.md`.

## Next steps

Items 3 and 4 below are done in [`v2/`](v2/README.md), which also adds the
walk-forward backtest item 6 asked for. The remaining ideas apply to v2:

1. **EPA / DVOA** from nflverse play-by-play (`load_pbp_data`) —
   drive-level efficiency >> box-score stats.
2. **QB on/off magnitude** — currently `qb_change` is just a flag; add
   the starter's career EPA and the backup's EPA to capture the size of
   the dropoff.
3. ~~**Travel + altitude**~~ — travel distance is a v2 feature.
4. ~~**Calibration**~~ — v2 Platt-calibrates on the prior season.
5. **Hyperparameter search** — still premature. The v2 result says the
   ceiling is set by the market, not by the tuning.
6. ~~**Backtest a betting strategy**~~ — v2 runs a flat-stake backtest over
   2010-2025. It loses at roughly the rate the vig predicts.
