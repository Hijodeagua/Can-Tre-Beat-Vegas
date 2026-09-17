"""Feed freshness for the soccer daily run: a feed whose newest row is
older than its staleness guard reads STALE, and the export carries it."""

import pandas as pd

from common import freshness
from soccer.clubs.daily import export_site, state


def test_feed_status_reads_the_committed_feeds():
    feeds = state.feed_status("2026-09-17")
    assert set(feeds) == {"xg", "shots", "understat_advanced"}
    # The committed xG file ended on 2025-01-04: stale on any 2026 run date.
    assert feeds["xg"].rows > 0 and not feeds["xg"].fresh
    # The processed Understat table is not committed until Actions fetches it.
    assert feeds["understat_advanced"].rows == 0 and not feeds["understat_advanced"].fresh


def test_feeds_payload_serialises_reports():
    r = freshness.check("x", pd.DataFrame({"date": ["2026-09-01"]}), "date", "2026-09-17", 130)
    out = export_site.feeds_payload({"x": r})
    assert out["x"] == {"source": "x", "newest": "2026-09-01", "age_days": 16,
                        "tolerance_days": 130, "rows": 1, "fresh": True}
    assert export_site.feeds_payload(None) == {}
