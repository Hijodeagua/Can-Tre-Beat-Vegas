"""Prediction-market snapshotters (Kalshi, Polymarket).

Same shape as ``data_jobs/odds_api``: fetch, normalise to a tidy CSV snapshot
in ``data/prediction_markets/``, one timestamped file per pull. Both APIs are
public and keyless for read-only market data, so unlike The Odds API there is
no quota to conserve.

Once snapshots accumulate, ``NFL/inventory/book_performance.py``'s leaderboard
can grade the prediction markets alongside the sportsbooks — a market price
is just another no-vig probability.
"""
