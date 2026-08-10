# Matchup-specific expected totals - backtest

Walk-forward 2021-2025 (12148 games); every prediction uses only games completed before it. The baseline is the walk-forward league mean total - i.e. exactly what the score sim used before this change. Win probability and expected margin come from Elo in both variants and are untouched.

| | matchup total | league-constant baseline |
|---|---|---|
| MAE (runs) | **3.5343** | 3.5452 |
| RMSE (runs) | **4.4713** | 4.4889 |

- Improvement: +0.0109 MAE (+0.31%).
- Calibration slope of actual on predicted: 0.949 (1.0 = perfectly calibrated variance; below 1 means the spread is slightly optimistic, as expected with shrinkage tuned for MAE).
- Correlation with actual totals: 0.083.
- Predicted-total spread: 5th pct 8.3, median 8.9, 95th pct 9.6 - games now differ from each other instead of sharing one number.

## By season

| season | matchup MAE | baseline MAE |
|---|---|---|
| 2021 | 3.556 | 3.572 |
| 2022 | 3.465 | 3.484 |
| 2023 | 3.634 | 3.626 |
| 2024 | 3.413 | 3.426 |
| 2025 | 3.604 | 3.617 |

Model: EWMA runs scored/allowed per team (half-life 20 team-games, shrunk to league mean with a 15-game prior), T = L*(att_h*def_a + att_a*def_h), clipped to [5.5, 13.5]. Recent form is the weighting, so scoring streaks move totals without an explicit streak feature.
