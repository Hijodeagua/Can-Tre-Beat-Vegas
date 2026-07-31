"""Roster and squad-quality data ingestion.

Two very different sources live here:

- ``fetch_nflverse`` — free, keyless CSV releases (weekly rosters, the player
  master with draft position, and weekly player stats). Fully automatable and
  safe to run in CI.
- ``scrape_awards`` — Pro Football Reference and Wikipedia, for All-Pro /
  Pro Bowl / NFL Top 100 honors. **Must be run locally**; see that module's
  docstring for why.
"""
