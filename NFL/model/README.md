# NFL LightGBM Model

First-pass predictive model for the upcoming season. Two targets supported:
straight-up winner (`win`) and against-the-spread cover (`ats`).

## Layout

- `features.py` — rolling feature engineering from `data/2023-2025W3.csv`
- `train.py` — temporal split + LightGBM training + baseline comparison
- `artifacts/` — saved model files, metrics CSVs, feature importances

## Running

```bash
cd NFL/model
python3 train.py --target win   # SU winner
python3 train.py --target ats   # cover the spread (drops pushes)
```

## Data flow

Raw rows are per-team per-game with post-game box-score stats (leakage if
used directly). `features.py`:

1. Sorts by team + date.
2. Builds rolling 4-game averages of offensive/defensive stats, shifted so
   the current game is excluded.
3. Merges the opponent's rolling stats onto each row (`opp_roll_*`).
4. Adds rest days, day-of-week, spread, total.
5. Drops a team's first two games of each season (insufficient history).

## Temporal split

- Train: `< 2024-09-01`
- Val:   `2024-09-01` to `2025-01-15`
- Test:  `>= 2025-01-15`

## First-run results

### Straight-up winner

| model               | n   | acc   | log_loss | brier | auc   |
|---------------------|-----|-------|----------|-------|-------|
| lgbm (val)          | 556 | 0.712 | 0.613    | 0.211 | 0.723 |
| market_logistic     | 556 | 0.714 | **0.593**| 0.202 | **0.757** |
| pick_favorite       | 556 | 0.714 | 1.324    | 0.280 | 0.714 |
| lgbm (test)         | 144 | 0.639 | 0.624    | 0.217 | 0.711 |
| market_logistic     | 144 | 0.688 | 0.599    | 0.205 | 0.753 |
| pick_favorite       | 144 | 0.688 | 1.446    | 0.306 | 0.687 |

**Read**: LightGBM matches market accuracy in-sample but a 1-feature
logistic on the spread beats it on calibration (log-loss, Brier, AUC).
The model isn't yet adding signal beyond what the spread already encodes.

### ATS (cover)

| model               | n   | acc   | log_loss | brier | auc   |
|---------------------|-----|-------|----------|-------|-------|
| lgbm (val)          | 542 | 0.506 | 0.693    | 0.250 | 0.530 |
| lgbm (test)         | 140 | 0.571 | 0.692    | 0.249 | 0.586 |
| market_logistic     | 140 | 0.514 | 0.695    | 0.251 | 0.503 |

`best_iter=1` on ATS — early stopping fires immediately, meaning the
current feature set has essentially zero signal for cover at training time.
The 57% test accuracy is on 140 games and within noise. **Treat as
break-even until additional features are added.**

## Next steps (in priority order)

1. **Add home/away** — not in the current CSV; need to merge a schedule
   source. This is the single biggest missing feature.
2. **Line movement** — diff opening vs. closing spread from `data/odds_api_data_*.csv`.
3. **QB on/off** — starter status (injury reports) is huge for NFL.
4. **Weather** — wind/precip for outdoor games.
5. **EPA / DVOA** — richer rolling drive-level stats from nflverse.
6. **Hyperparameter search** — once features are stronger; tuning before
   then just overfits noise.
7. **Calibration** — Platt or isotonic on validation predictions before
   computing edge vs. market.
