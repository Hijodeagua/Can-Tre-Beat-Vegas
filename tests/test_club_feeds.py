"""Feed freshness for the soccer daily run: a feed whose newest row is
older than its staleness guard reads STALE, and the export carries it."""

import pandas as pd

from common import freshness
from soccer.clubs.daily import export_site, state


def test_feed_status_reads_the_committed_feeds():
    feeds = state.feed_status("2026-09-17")
    assert set(feeds) == {"xg", "shots", "understat_advanced"}
    assert feeds["xg"].rows > 0 and feeds["xg"].newest is not None
    # Whatever the committed files hold today, the verdict follows the
    # guard: fresh iff the newest row is within the tolerance.
    for r in feeds.values():
        if r.rows:
            assert r.fresh == (r.age_days <= r.tolerance_days)
    # A run date far past every feed reads STALE across the board.
    far = state.feed_status("2099-01-01")
    assert not any(r.fresh for r in far.values())


def test_feeds_payload_serialises_reports():
    r = freshness.check("x", pd.DataFrame({"date": ["2026-09-01"]}), "date", "2026-09-17", 130)
    out = export_site.feeds_payload({"x": r})
    assert out["x"] == {"source": "x", "newest": "2026-09-01", "age_days": 16,
                        "tolerance_days": 130, "rows": 1, "fresh": True}
    assert export_site.feeds_payload(None) == {}
