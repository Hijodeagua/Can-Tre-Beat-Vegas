# Starting-pitcher Elo adjustment - walk-forward backtest

Method, leakage contract, and design choices: see the module
docstring of `mlb/backtest_sp.py`. Team Elo updates are untouched;
adjustments act on the pregame probability only, so variant (a) is
bit-identical to production. Realized starters proxy for announced
probables. Seasons >= 2012 evaluated (2009-2011 burn-in), tuned on
2012-2021, holdout 2022-2025 reported once.

Baseline reproduction: variant (a) log-loss on >= 2012 = 0.67962 vs published 0.67961 - reproduced.

## Tune window (2012-2021)

| variant | n | log-loss | Brier | BSS vs 0.5 | accuracy | P(pick)>62% | Δlog-loss vs always-home (paired ± SE) | Δaccuracy vs always-home (paired ± SE) |
|---|---|---|---|---|---|---|---|---|
| (a) current model | 22765 | 0.67959 | 0.24331 | +0.0268 | 0.5662 | 11.71% | -0.01108 ± 0.00090 | +0.0311 ± 0.0037 |
| (b) + pitcher (C=4.7, hl=10) | 22765 | 0.67898 | 0.24301 | +0.0280 | 0.5658 | 16.24% | -0.01169 ± 0.00104 | +0.0307 ± 0.0038 |
| (c) + pitcher + rest/travel | 22765 | 0.67896 | 0.24300 | +0.0280 | 0.5668 | 16.43% | -0.01172 ± 0.00104 | +0.0317 ± 0.0038 |
| (d) tuned: C=3.0, hl=20, + rest/travel | 22765 | 0.67813 | 0.24259 | +0.0296 | 0.5708 | 12.35% | -0.01255 ± 0.00092 | +0.0356 ± 0.0037 |

Always-pick-home on the same games: log-loss 0.69067 (home win rate 0.5352).

## Fallback-ladder coverage (half-life 10, eval games, both sides)

| mode | share |
|---|---|
| pitcher | 87.66% |
| thin | 7.02% |
| staff | 5.32% |

## Grid (tuned on tune-window log-loss only)

| half-life | C | tune log-loss |
|---|---|---|
| 20 | 3.0 | 0.67813 **<- selected** |
| 10 | 3.0 | 0.67818 |
| 20 | 4.0 | 0.67822 |
| 20 | 4.7 | 0.67845 |
| 10 | 4.0 | 0.67850 |
| 5 | 3.0 | 0.67863 |
| 20 | 5.5 | 0.67889 |
| 10 | 4.7 | 0.67896 |
| 5 | 4.0 | 0.67957 |
| 20 | 6.5 | 0.67968 |
| 10 | 5.5 | 0.67970 |
| 5 | 4.7 | 0.68058 |
| 10 | 6.5 | 0.68098 |
| 5 | 5.5 | 0.68211 |
| 5 | 6.5 | 0.68456 |

## Holdout (2022-2025) - reported once

| variant | n | log-loss | Brier | BSS vs 0.5 | accuracy | P(pick)>62% | Δlog-loss vs always-home (paired ± SE) | Δaccuracy vs always-home (paired ± SE) |
|---|---|---|---|---|---|---|---|---|
| (a) current model | 9719 | 0.67832 | 0.24270 | +0.0292 | 0.5700 | 16.66% | -0.01306 ± 0.00159 | +0.0403 ± 0.0058 |
| (b) + pitcher (C=4.7, hl=10) | 9719 | 0.67812 | 0.24259 | +0.0296 | 0.5717 | 19.93% | -0.01326 ± 0.00175 | +0.0420 ± 0.0060 |
| (c) + pitcher + rest/travel | 9719 | 0.67818 | 0.24262 | +0.0295 | 0.5708 | 20.10% | -0.01320 ± 0.00175 | +0.0412 ± 0.0060 |
| (d) tuned: C=3.0, hl=20, + rest/travel | 9719 | 0.67706 | 0.24208 | +0.0317 | 0.5732 | 17.33% | -0.01433 ± 0.00161 | +0.0435 ± 0.0058 |

Always-pick-home on the holdout: log-loss 0.69138 (home win rate 0.5297).

### Paired variant-vs-current differences (the decision numbers)

Per-game paired differences of each variant against (a) on the same
games, mean ± SE:

| window | comparison | Δlog-loss ± SE | z | Δaccuracy ± SE |
|---|---|---|---|---|
| tune 2012-2021 | (b) + pitcher (C=4.7, hl=10) vs (a) | -0.00061 ± 0.00061 | -0.99 | -0.0004 ± 0.0028 |
| tune 2012-2021 | (c) + pitcher + rest/travel vs (a) | -0.00063 ± 0.00061 | -1.03 | +0.0006 ± 0.0028 |
| tune 2012-2021 | (d) tuned: C=3.0, hl=20, + rest/travel vs (a) | -0.00146 ± 0.00034 | -4.35 | +0.0046 ± 0.0021 |
| holdout 2022-2025 | (b) + pitcher (C=4.7, hl=10) vs (a) | -0.00020 ± 0.00090 | -0.22 | +0.0016 ± 0.0039 |
| holdout 2022-2025 | (c) + pitcher + rest/travel vs (a) | -0.00014 ± 0.00091 | -0.15 | +0.0008 ± 0.0039 |
| holdout 2022-2025 | (d) tuned: C=3.0, hl=20, + rest/travel vs (a) | -0.00126 ± 0.00050 | -2.54 | +0.0032 ± 0.0029 |

## Interpretation

- On the untouched 2022-2025 holdout, the tuned variant (d) improves log-loss by 0.00126 ± 0.00050 per game over the current model (paired, z = -2.54) - distinguishable from zero at the ~2σ level.
- Accuracy moves +0.32% ± 0.29% on the holdout. 538 reported roughly +1pp of games called correctly from their pitcher adjustment; we are below that, not above it, which is the right side to land on for leakage suspicion (their pitcher model also carried more machinery than a single rolling game score).
- Resolution: the share of picks above 62% rises versus the current model (see tables) - the adjustment widens the probability range rather than shuffling picks near the coin-flip line.
- The grid prefers a LONGER memory (half-life 20 starts) and a SMALLER C (3.0) than 538's published 4.7: single-season game-score noise wants more smoothing, and with the smoothed rating the published 4.7 scaling overshoots.
- Rest/travel is approximately free on top of the pitcher adjustment (compare (b) vs (c)): most of (d)'s gain over (b) comes from the retuned (C, half-life), not from rest/travel.

## Calibration, tune window 2012-2021 (5-point bins of p_home)

### (a) current model

| p_home bin | n | mean predicted | observed |
|---|---|---|---|
| [0.00, 0.30) | 12 | 0.2870 | 0.4167 |
| [0.30, 0.35) | 87 | 0.3306 | 0.2759 |
| [0.35, 0.40) | 484 | 0.3778 | 0.3306 |
| [0.40, 0.45) | 1931 | 0.4301 | 0.4179 |
| [0.45, 0.50) | 4487 | 0.4769 | 0.4794 |
| [0.50, 0.55) | 6358 | 0.5251 | 0.5296 |
| [0.55, 0.60) | 5508 | 0.5732 | 0.5741 |
| [0.60, 0.65) | 2956 | 0.6205 | 0.6313 |
| [0.65, 0.70) | 797 | 0.6697 | 0.6725 |
| [0.70, 0.75) | 129 | 0.7151 | 0.7132 |
| [0.75, 1.00) | 16 | 0.7614 | 0.8125 |

### (d) tuned: C=3.0, hl=20, + rest/travel

| p_home bin | n | mean predicted | observed |
|---|---|---|---|
| [0.00, 0.30) | 20 | 0.2883 | 0.1500 |
| [0.30, 0.35) | 110 | 0.3331 | 0.2364 |
| [0.35, 0.40) | 509 | 0.3800 | 0.3694 |
| [0.40, 0.45) | 1962 | 0.4300 | 0.4134 |
| [0.45, 0.50) | 4464 | 0.4772 | 0.4702 |
| [0.50, 0.55) | 6177 | 0.5257 | 0.5304 |
| [0.55, 0.60) | 5524 | 0.5737 | 0.5798 |
| [0.60, 0.65) | 2958 | 0.6211 | 0.6285 |
| [0.65, 0.70) | 866 | 0.6690 | 0.6824 |
| [0.70, 0.75) | 162 | 0.7171 | 0.7284 |
| [0.75, 1.00) | 13 | 0.7720 | 0.6923 |


## Calibration, holdout 2022-2025 (5-point bins of p_home)

### (a) current model

| p_home bin | n | mean predicted | observed |
|---|---|---|---|
| [0.30, 0.35) | 120 | 0.3318 | 0.3083 |
| [0.35, 0.40) | 334 | 0.3788 | 0.3832 |
| [0.40, 0.45) | 963 | 0.4291 | 0.4496 |
| [0.45, 0.50) | 1803 | 0.4764 | 0.4526 |
| [0.50, 0.55) | 2369 | 0.5254 | 0.5201 |
| [0.55, 0.60) | 2134 | 0.5728 | 0.5717 |
| [0.60, 0.65) | 1327 | 0.6215 | 0.6134 |
| [0.65, 0.70) | 523 | 0.6707 | 0.6979 |
| [0.70, 0.75) | 133 | 0.7192 | 0.7218 |
| [0.75, 1.00) | 13 | 0.7571 | 0.5385 |

### (d) tuned: C=3.0, hl=20, + rest/travel

| p_home bin | n | mean predicted | observed |
|---|---|---|---|
| [0.00, 0.30) | 9 | 0.2932 | 0.4444 |
| [0.30, 0.35) | 94 | 0.3318 | 0.3617 |
| [0.35, 0.40) | 349 | 0.3771 | 0.3897 |
| [0.40, 0.45) | 972 | 0.4279 | 0.4126 |
| [0.45, 0.50) | 1817 | 0.4765 | 0.4590 |
| [0.50, 0.55) | 2306 | 0.5254 | 0.5186 |
| [0.55, 0.60) | 2120 | 0.5736 | 0.5679 |
| [0.60, 0.65) | 1354 | 0.6222 | 0.6174 |
| [0.65, 0.70) | 548 | 0.6701 | 0.7281 |
| [0.70, 0.75) | 132 | 0.7205 | 0.7045 |
| [0.75, 1.00) | 18 | 0.7611 | 0.6111 |

