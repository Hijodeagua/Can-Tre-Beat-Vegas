"""
Compatibility entry point: `python -m soccer.clubs.data.fetch_xg` is what
the daily runner and the Actions job have always called to refresh
per-match xG. The scraper that used to live here — a regex over the
league page for a `datesData` blob — stopped working when Understat moved
the data behind its JSON league endpoint, which is why the committed xG
ended on 2025-01-04.

The fetch and parse now live in `soccer/clubs/data/understat.py`, which
writes the full processed table (xG, npxG, xPts, PPDA, deep completions)
*and* keeps `xg_matches.csv` — the file `model/xg.py` reads — in step.
This module just forwards, so nothing that calls it has to change.
"""

from soccer.clubs.data.understat import (  # noqa: F401  (re-exported for callers)
    LEAGUES,
    OUT_CSV as PROCESSED_CSV,
    LEGACY_XG_CSV as OUT_CSV,
    UNDERSTAT_ALIASES,
    main,
)

if __name__ == "__main__":
    main()
