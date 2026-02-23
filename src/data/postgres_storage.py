"""
PostgreSQL / TimescaleDB storage backend.

Drop-in replacement for SQLite Storage that supports:
  - TimescaleDB hypertables for time-series data
  - Concurrent read/write access
  - Automatic data compression
  - Connection pooling via asyncpg
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    logger.info("asyncpg not installed — PostgreSQL backend unavailable")


class PostgresStorage:
    """
    Async PostgreSQL/TimescaleDB storage backend.

    Connection configured via DATABASE_URL env var or constructor params.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        min_connections: int = 2,
        max_connections: int = 10,
    ):
        self._dsn = dsn or os.getenv(
            "DATABASE_URL",
            "postgresql://pabot:pabot@localhost:5432/pabot"
        )
        self._min_conn = min_connections
        self._max_conn = max_connections
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Initialize connection pool."""
        if not HAS_ASYNCPG:
            raise RuntimeError("asyncpg not installed — pip install asyncpg")

        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_conn,
            max_size=self._max_conn,
        )
        logger.info("PostgreSQL connection pool created")

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("PostgreSQL connection pool closed")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def initialize_schema(self) -> None:
        """Create tables and convert to hypertables if TimescaleDB is available."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    ts          TIMESTAMPTZ NOT NULL,
                    symbol      TEXT NOT NULL,
                    interval    TEXT NOT NULL DEFAULT '60',
                    open        DOUBLE PRECISION,
                    high        DOUBLE PRECISION,
                    low         DOUBLE PRECISION,
                    close       DOUBLE PRECISION,
                    volume      DOUBLE PRECISION,
                    UNIQUE(symbol, ts, interval)
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id              SERIAL PRIMARY KEY,
                    ts              TIMESTAMPTZ NOT NULL,
                    symbol          TEXT NOT NULL,
                    interval        TEXT DEFAULT '60',
                    direction       TEXT,
                    prob_up         DOUBLE PRECISION,
                    prob_flat       DOUBLE PRECISION,
                    prob_down       DOUBLE PRECISION,
                    magnitude       DOUBLE PRECISION,
                    conviction      DOUBLE PRECISION,
                    signal_score    DOUBLE PRECISION,
                    actual_direction TEXT,
                    actual_magnitude DOUBLE PRECISION,
                    was_correct     BOOLEAN,
                    scored_at       TIMESTAMPTZ,
                    created_at      TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS order_book_snapshots (
                    ts              TIMESTAMPTZ NOT NULL,
                    symbol          TEXT NOT NULL,
                    bid_volume      DOUBLE PRECISION,
                    ask_volume      DOUBLE PRECISION,
                    spread          DOUBLE PRECISION,
                    mid_price       DOUBLE PRECISION,
                    imbalance       DOUBLE PRECISION,
                    best_bid        DOUBLE PRECISION,
                    best_ask        DOUBLE PRECISION,
                    bid_levels      JSONB,
                    ask_levels      JSONB,
                    UNIQUE(symbol, ts)
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS accuracy_log (
                    run_date            DATE PRIMARY KEY,
                    total_scored        INTEGER,
                    direction_accuracy  DOUBLE PRECISION,
                    magnitude_mae       DOUBLE PRECISION,
                    up_precision        DOUBLE PRECISION,
                    up_recall           DOUBLE PRECISION,
                    down_precision      DOUBLE PRECISION,
                    down_recall         DOUBLE PRECISION,
                    flat_precision      DOUBLE PRECISION,
                    flat_recall         DOUBLE PRECISION,
                    flat_threshold      DOUBLE PRECISION
                );
            """)

            try:
                await conn.execute(
                    "SELECT create_hypertable('ohlcv', 'ts', if_not_exists => TRUE);"
                )
                await conn.execute(
                    "SELECT create_hypertable('predictions', 'ts', if_not_exists => TRUE);"
                )
                await conn.execute(
                    "SELECT create_hypertable('order_book_snapshots', 'ts', if_not_exists => TRUE);"
                )
                logger.info("TimescaleDB hypertables created")
            except Exception:
                logger.info("TimescaleDB not available — using standard PostgreSQL tables")

            try:
                await conn.execute("""
                    ALTER TABLE ohlcv SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'symbol,interval'
                    );
                """)
                await conn.execute("""
                    SELECT add_compression_policy('ohlcv',
                        INTERVAL '30 days', if_not_exists => TRUE);
                """)
                logger.info("TimescaleDB compression policies set")
            except Exception:
                pass

    async def insert_candles(
        self,
        candles: List[Tuple],
        interval: str = "60",
    ) -> int:
        """Insert OHLCV candles. Returns number inserted."""
        if not candles:
            return 0

        async with self._pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO ohlcv (ts, symbol, interval, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (symbol, ts, interval) DO NOTHING
            """, candles)

        return len(candles)

    async def get_ohlcv(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Get OHLCV data for a symbol."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT ts, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = $1 AND interval = $2
                ORDER BY ts DESC
                LIMIT $3
            """, symbol, interval, limit)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame([dict(r) for r in rows])

    async def insert_prediction(self, prediction: Dict[str, Any]) -> int:
        """Insert a prediction record."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO predictions (
                    ts, symbol, interval, direction,
                    prob_up, prob_flat, prob_down,
                    magnitude, conviction, signal_score
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
            """,
                prediction["ts"], prediction["symbol"],
                prediction.get("interval", "60"),
                prediction["direction"],
                prediction.get("prob_up", 0),
                prediction.get("prob_flat", 0),
                prediction.get("prob_down", 0),
                prediction.get("magnitude", 0),
                prediction.get("conviction", 0),
                prediction.get("signal_score", 0),
            )
        return row["id"]
