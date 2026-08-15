"""Optional Postgres (Neon) writes, gated on DATABASE_URL.

When the DATABASE_URL env var is present (in CI it comes from the repo
secret), each day's slate predictions, grade row, and futures snapshot are
upserted so the FastAPI backend / site can serve the same source of truth as
the emails. When it's absent the pipeline logs one line and moves on - the
committed CSV/JSON artifacts remain the fallback source.
"""

from __future__ import annotations

import os

import pandas as pd

DDL = """
CREATE TABLE IF NOT EXISTS mlb_slate_predictions (
    date DATE NOT NULL,
    away TEXT NOT NULL,
    home TEXT NOT NULL,
    game_num INT NOT NULL,
    away_sp TEXT,
    home_sp TEXT,
    pred_total DOUBLE PRECISION,
    p_home DOUBLE PRECISION,
    pick TEXT,
    pick_prob DOUBLE PRECISION,
    pred_home_score INT,
    pred_away_score INT,
    elo_home DOUBLE PRECISION,
    elo_away DOUBLE PRECISION,
    PRIMARY KEY (date, away, home, game_num)
);
CREATE TABLE IF NOT EXISTS mlb_daily_grades (
    date DATE PRIMARY KEY,
    games INT,
    correct INT,
    accuracy DOUBLE PRECISION,
    log_loss DOUBLE PRECISION,
    brier DOUBLE PRECISION,
    avg_margin_err DOUBLE PRECISION,
    avg_total_err DOUBLE PRECISION,
    skipped INT,
    cum_games INT,
    cum_correct INT,
    cum_accuracy DOUBLE PRECISION,
    cum_log_loss DOUBLE PRECISION,
    cum_brier DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS mlb_pitcher_starts (
    game_id TEXT NOT NULL,
    source TEXT NOT NULL,
    date DATE NOT NULL,
    season INT NOT NULL,
    game_num INT NOT NULL,
    team TEXT NOT NULL,
    opponent TEXT NOT NULL,
    home INT NOT NULL,
    retro_id TEXT,
    mlbam_id TEXT,
    name TEXT,
    outs INT,
    h INT,
    r INT,
    er INT,
    bb INT,
    so INT,
    PRIMARY KEY (game_id, team)
);
CREATE TABLE IF NOT EXISTS mlb_futures (
    date DATE NOT NULL,
    team TEXT NOT NULL,
    elo DOUBLE PRECISION,
    mean_wins DOUBLE PRECISION,
    mean_losses DOUBLE PRECISION,
    division_pct DOUBLE PRECISION,
    playoff_pct DOUBLE PRECISION,
    top_seed_pct DOUBLE PRECISION,
    PRIMARY KEY (date, team)
);
"""

# Idempotent migrations for tables created before a column existed - the
# CREATE TABLE IF NOT EXISTS above is a no-op on an existing database, so
# every column added after first release must also appear here.
MIGRATIONS = """
ALTER TABLE mlb_slate_predictions ADD COLUMN IF NOT EXISTS away_sp TEXT;
ALTER TABLE mlb_slate_predictions ADD COLUMN IF NOT EXISTS home_sp TEXT;
ALTER TABLE mlb_slate_predictions ADD COLUMN IF NOT EXISTS pred_total DOUBLE PRECISION;
ALTER TABLE mlb_slate_predictions ADD COLUMN IF NOT EXISTS away_sp_id TEXT;
ALTER TABLE mlb_slate_predictions ADD COLUMN IF NOT EXISTS home_sp_id TEXT;
ALTER TABLE mlb_slate_predictions ADD COLUMN IF NOT EXISTS model_version TEXT;
ALTER TABLE mlb_slate_predictions ADD COLUMN IF NOT EXISTS home_sp_adj DOUBLE PRECISION;
ALTER TABLE mlb_slate_predictions ADD COLUMN IF NOT EXISTS away_sp_adj DOUBLE PRECISION;
ALTER TABLE mlb_slate_predictions ADD COLUMN IF NOT EXISTS home_sp_mode TEXT;
ALTER TABLE mlb_slate_predictions ADD COLUMN IF NOT EXISTS away_sp_mode TEXT;
ALTER TABLE mlb_slate_predictions ADD COLUMN IF NOT EXISTS home_rt_adj DOUBLE PRECISION;
ALTER TABLE mlb_slate_predictions ADD COLUMN IF NOT EXISTS away_rt_adj DOUBLE PRECISION;
ALTER TABLE mlb_daily_grades ADD COLUMN IF NOT EXISTS home_log_loss DOUBLE PRECISION;
ALTER TABLE mlb_daily_grades ADD COLUMN IF NOT EXISTS home_correct INT;
ALTER TABLE mlb_daily_grades ADD COLUMN IF NOT EXISTS d_ll_sum DOUBLE PRECISION;
ALTER TABLE mlb_daily_grades ADD COLUMN IF NOT EXISTS d_ll_sq_sum DOUBLE PRECISION;
ALTER TABLE mlb_daily_grades ADD COLUMN IF NOT EXISTS cum_d_ll_mean DOUBLE PRECISION;
ALTER TABLE mlb_daily_grades ADD COLUMN IF NOT EXISTS cum_d_ll_se DOUBLE PRECISION;
"""


def _upsert(cur, table: str, df: pd.DataFrame, key_cols: list[str]) -> None:
    cols = list(df.columns)
    updates = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in cols if c not in key_cols
    )
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join(['%s'] * len(cols))}) "
        f"ON CONFLICT ({', '.join(key_cols)}) DO UPDATE SET {updates}"
    )
    rows = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.itertuples(index=False)
    ]
    cur.executemany(sql, rows)


def write_postgres(run_date: str, slate: pd.DataFrame,
                   futures: pd.DataFrame,
                   ledger_row: pd.Series | None) -> bool:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("db: DATABASE_URL not set - skipping Postgres write")
        return False

    import psycopg2  # deferred so local runs don't require the driver

    conn = psycopg2.connect(url)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(DDL)
            cur.execute(MIGRATIONS)
            if not slate.empty:
                _upsert(cur, "mlb_slate_predictions", slate,
                        ["date", "away", "home", "game_num"])
            f = futures.copy()
            f.insert(0, "date", run_date)
            _upsert(cur, "mlb_futures", f, ["date", "team"])
            if ledger_row is not None:
                _upsert(cur, "mlb_daily_grades",
                        pd.DataFrame([ledger_row]), ["date"])
        print(f"db: wrote slate/futures/grade for {run_date}")
        return True
    finally:
        conn.close()
