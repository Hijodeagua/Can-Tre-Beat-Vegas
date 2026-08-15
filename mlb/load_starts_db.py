"""Bulk-upsert data/mlb/pitcher_starts.csv into Neon Postgres.

    DATABASE_URL=... python -m mlb.load_starts_db

Idempotent: keyed on (game_id, team), so reruns update in place. Skips
cleanly (exit 0, one log line) when DATABASE_URL is unset - the committed
CSV remains the source of truth, same contract as mlb/daily/db.py. The
per-start table complements (does not duplicate) the existing game table:
mlb_slate_predictions stores probables by name at slate time; this stores
the realized starting pitcher and his line per (game, team).
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from mlb.daily.db import DDL, MIGRATIONS, _upsert
from mlb.daily.config import REPO

STARTS_CSV = REPO / "data" / "mlb" / "pitcher_starts.csv"


def main() -> int:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("load_starts_db: DATABASE_URL not set - skipping")
        return 0

    import psycopg2

    df = pd.read_csv(STARTS_CSV, dtype={"mlbam_id": str, "retro_id": str})
    conn = psycopg2.connect(url)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(DDL)
            cur.execute(MIGRATIONS)
            _upsert(cur, "mlb_pitcher_starts", df, ["game_id", "team"])
        print(f"load_starts_db: upserted {len(df)} rows")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
