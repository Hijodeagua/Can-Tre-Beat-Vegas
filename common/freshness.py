"""Source-freshness checks.

A feed that stops updating is the quiet failure mode of every feature
layer here: the fetcher exits 0 with the committed file untouched, the
rolling form goes stale, the staleness guard zeroes the feature, and the
model silently degrades to Elo-only while every dashboard still shows a
feature name. Understat did exactly this from 2025-01-04.

So every advanced feed carries a `FreshnessReport`: the newest date in
the processed table, how old that is on the run date, and whether it is
inside the feed's tolerance. The daily pipelines log it; the tests assert
on it; and a feature layer whose feed is stale is *disabled for the run*
rather than fed zeros — the model falls back to the feature set that is
actually current.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class FreshnessReport:
    source: str
    newest: str | None       # ISO date of the newest row, None if empty
    age_days: int | None
    tolerance_days: int
    rows: int

    @property
    def fresh(self) -> bool:
        return self.age_days is not None and self.age_days <= self.tolerance_days

    def __str__(self) -> str:
        state = "fresh" if self.fresh else "STALE"
        return (f"{self.source}: {self.rows} rows, newest {self.newest}, "
                f"{self.age_days} days old ({state}, tolerance {self.tolerance_days})")


def check(source: str, frame: pd.DataFrame | None, date_col: str,
          run_date: str | date, tolerance_days: int) -> FreshnessReport:
    """Freshness of a processed table on `run_date`.

    `tolerance_days` is the feed's own cadence plus the longest gap it is
    allowed to span (an off-season, a bye week). Past it, the report reads
    stale and the caller must not feature against this table.
    """
    if frame is None or len(frame) == 0:
        return FreshnessReport(source, None, None, tolerance_days, 0)
    newest = pd.to_datetime(frame[date_col]).max()
    run = pd.Timestamp(run_date)
    age = int((run - newest).days)
    return FreshnessReport(source, newest.date().isoformat(), age, tolerance_days, int(len(frame)))


def gate(report: FreshnessReport, log=print) -> bool:
    """Log the report and return whether the feed may be used. One line
    per feed per run so a stale feed is visible in the Actions log the
    day it goes stale."""
    log(f"   freshness: {report}")
    return report.fresh
