#!/usr/bin/env python3
"""
Database migration: SQLite -> PostgreSQL/TimescaleDB.

Reads all data from the SQLite database and inserts it into PostgreSQL.
Designed to be run once when transitioning to production infrastructure.

Usage:
    python scripts/migrate_to_postgres.py --sqlite data/ohlcv.db
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv()

import aiosqlite

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def migrate(sqlite_path: str, pg_dsn: str) -> None:
    """Migrate all tables from SQLite to PostgreSQL."""
    try:
        import asyncpg  # noqa: F401
    except ImportError:
        logger.error("asyncpg not installed — pip install asyncpg")
        return

    from src.data.postgres_storage import PostgresStorage

    storage = PostgresStorage(dsn=pg_dsn)
    await storage.connect()
    await storage.initialize_schema()

    async with aiosqlite.connect(sqlite_path) as db:
        db.row_factory = aiosqlite.Row

        logger.info("Migrating ohlcv...")
        cursor = await db.execute("SELECT COUNT(*) FROM ohlcv")
        total = (await cursor.fetchone())[0]
        logger.info("  %d rows to migrate", total)

        batch_size = 5000
        offset = 0
        migrated = 0

        while offset < total:
            cursor = await db.execute(
                "SELECT ts, symbol, interval, open, high, low, close, volume "
                "FROM ohlcv ORDER BY ts LIMIT ? OFFSET ?",
                (batch_size, offset),
            )
            rows = await cursor.fetchall()
            if not rows:
                break

            async with storage._pool.acquire() as conn:
                await conn.executemany("""
                    INSERT INTO ohlcv (ts, symbol, interval, open, high, low, close, volume)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (symbol, ts, interval) DO NOTHING
                """, [tuple(r) for r in rows])

            migrated += len(rows)
            offset += batch_size
            if migrated % 50000 == 0:
                logger.info("  migrated %d/%d ohlcv rows", migrated, total)

        logger.info("  ohlcv migration complete: %d rows", migrated)

        logger.info("Migrating predictions...")
        cursor = await db.execute("SELECT COUNT(*) FROM predictions")
        total = (await cursor.fetchone())[0]

        cursor = await db.execute(
            "SELECT ts, symbol, interval, direction, prob_up, prob_flat, prob_down, "
            "magnitude, conviction, signal_score, actual_direction, actual_magnitude, "
            "was_correct, scored_at, created_at FROM predictions ORDER BY ts"
        )
        rows = await cursor.fetchall()

        async with storage._pool.acquire() as conn:
            for row in rows:
                await conn.execute("""
                    INSERT INTO predictions (
                        ts, symbol, interval, direction, prob_up, prob_flat, prob_down,
                        magnitude, conviction, signal_score, actual_direction,
                        actual_magnitude, was_correct, scored_at, created_at
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
                """, *[tuple(row)])

        logger.info("  predictions migration complete: %d rows", len(rows))

        try:
            logger.info("Migrating order_book_snapshots...")
            cursor = await db.execute(
                "SELECT ts, symbol, bid_volume, ask_volume, spread, mid_price, "
                "imbalance, best_bid, best_ask, bid_levels, ask_levels "
                "FROM order_book_snapshots ORDER BY ts"
            )
            rows = await cursor.fetchall()
            async with storage._pool.acquire() as conn:
                for row in rows:
                    await conn.execute("""
                        INSERT INTO order_book_snapshots (
                            ts, symbol, bid_volume, ask_volume, spread, mid_price,
                            imbalance, best_bid, best_ask, bid_levels, ask_levels
                        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                        ON CONFLICT (symbol, ts) DO NOTHING
                    """, *[tuple(row)])
            logger.info("  order_book migration complete: %d rows", len(rows))
        except Exception as e:
            logger.warning("  order_book migration skipped: %s", e)

    await storage.close()
    logger.info("Migration complete!")


def main():
    parser = argparse.ArgumentParser(description="Migrate SQLite to PostgreSQL")
    parser.add_argument("--sqlite", default="data/ohlcv.db", help="SQLite DB path")
    parser.add_argument(
        "--pg-dsn",
        default=os.getenv("DATABASE_URL", "postgresql://pabot:pabot@localhost:5432/pabot"),
        help="PostgreSQL DSN",
    )
    args = parser.parse_args()

    if not os.path.exists(args.sqlite):
        logger.error("SQLite database not found: %s", args.sqlite)
        sys.exit(1)

    asyncio.run(migrate(args.sqlite, args.pg_dsn))


if __name__ == "__main__":
    main()
