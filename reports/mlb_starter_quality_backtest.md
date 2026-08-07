# Starter quality vs. Elo - backtest

Question: with the Elo win probability controlled for, does the starting pitchers' prior-completed-season quality (ERA+/FIP, IP>=125 qualifying seasons only) improve game-level prediction?

## Lookback = 1 prior season(s)

- Games 2010-2025: 37343; both starters matched to a qualifying prior season: 9314 (24.9%). Unmatched games are excluded, so this sample is biased toward established starters.
- Coverage by season (min/median/max): 16% / 26% / 38%
- Most-started unmatched pitchers (start count) - the IP>=125 filter's visible gap, not an ID-join failure: Nathan Eovaldi (178), Martin Perez (164), Yu Darvish (153), Eduardo Rodriguez (151), Charlie Morton (150), Brett Anderson (148), Drew Smyly (145), Matt Harvey (143), Anthony DeSclafani (141), Adam Wainwright (136), Clay Buchholz (134), Robbie Ray (133), Carlos Carrasco (132), Michael Wacha (128), Sonny Gray (128)

### ERA+ delta (home - away)
- coefficient +0.00255, LR chi2 17.54, p = 2.811e-05 (n = 9314)
- OOS 2021-2025 (n=1720): log-loss 0.67885 (Elo) -> 0.67680 (+SP), delta -0.00205

### FIP delta (away - home)
- coefficient +0.11722, LR chi2 25.34, p = 4.801e-07 (n = 9314)
- OOS 2021-2025 (n=1720): log-loss 0.67885 (Elo) -> 0.67709 (+SP), delta -0.00176

## Lookback = 2 prior season(s)

- Games 2010-2025: 37343; both starters matched to a qualifying prior season: 12289 (32.9%). Unmatched games are excluded, so this sample is biased toward established starters.
- Coverage by season (min/median/max): 13% / 33% / 45%
- Most-started unmatched pitchers (start count) - the IP>=125 filter's visible gap, not an ID-join failure: Nathan Eovaldi (153), Luis Severino (121), Aaron Civale (117), David Peterson (115), James Paxton (114), Martin Perez (113), Homer Bailey (110), Tyler Glasnow (110), Jose Urena (109), Matt Boyd (109), Mike Clevinger (109), Alex Cobb (107), Michael Pineda (105), Kevin Gausman (105), Ross Stripling (104)

### ERA+ delta (home - away)
- coefficient +0.00254, LR chi2 22.80, p = 1.795e-06 (n = 12289)
- OOS 2021-2025 (n=2666): log-loss 0.68014 (Elo) -> 0.67922 (+SP), delta -0.00092

### FIP delta (away - home)
- coefficient +0.11218, LR chi2 31.03, p = 2.543e-08 (n = 12289)
- OOS 2021-2025 (n=2666): log-loss 0.68014 (Elo) -> 0.67935 (+SP), delta -0.00079

## Interpretation
- The signal is real and stable: ERA+ coefficient ~ +0.0025 per point at both lookbacks (p < 1e-4 with Elo controlled), i.e. a 30-point prior-season ERA+ edge (an ace vs. a league-average starter) is worth roughly +2 percentage points of win probability on top of Elo. Out-of-sample it recovers about 0.002 of log-loss on the matched sample - roughly a fifth of the entire Elo model's edge over always-pick-home (0.011).
- The binding constraint is coverage, not signal: the IP>=125 table matches both starters in only ~25-33% of games. Extending the pitcher table downward (lower-IP seasons) is the highest-value next data ask.

## Notes
- 2020 contributes no qualifying seasons (60-game year), so 2021 games at lookback 1 must reach the 2019 season via lookback 2 or drop out; both variants are shown above.
- Identity matching is exact (Retrosheet ID -> Chadwick register -> bbref ID); no name-based joins.
- Quality features are strictly prior-completed-season - the same no-look-ahead rule the pipeline applies to run differential.
