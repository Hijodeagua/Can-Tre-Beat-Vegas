"""Cross-sport plumbing for the advanced-metrics layers.

Three things every sport's second-stage model needs and none should
re-implement:

- `pregame` — leakage-safe rolling and exponentially weighted team form,
  shifted so a row only ever sees games completed before it.
- `freshness` — source-freshness checks, so a feed that stops updating
  turns into a loud failure and an Elo-only fallback rather than a column
  of silent zeros.
- `evaluate` — the chronological evaluation harness (log loss, Brier,
  accuracy, n) and the paired comparison every candidate feature group
  has to pass.

Nothing here knows what a sport is. Anything that does lives with that
sport.
"""
