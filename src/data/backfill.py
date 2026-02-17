"""
Convenience wrapper for backfill operations.

Can be imported or run directly via `scripts/backfill_data.py`.
"""

from __future__ import annotations

import asyncio
import logging

from src.api.bitunix_client import BitunixClient
from src.data.collector import DataCollector
from src.data.storage import Storage

logger = logging.getLogger(__name__)


async def run_backfill(
    db_path: str | None = None,
    symbols: list[str] | None = None,
    interval: str = "60",
    total_candles: int = 2000,
    concurrency: int = 5,
) -> None:
    """
    Backfill historical kline data from BitUnix.

    If *symbols* is None, we first discover all futures pairs from the
    tickers endpoint and backfill each one.
    """
    async with BitunixClient() as client, Storage(db_path) as storage:
        collector = DataCollector(client, storage)

        if symbols is None:
            symbols = await collector.discover_futures_symbols()
            logger.info("Will backfill %d discovered symbols", len(symbols))

        total = await collector.backfill_all(
            symbols,
            interval=interval,
            total_candles=total_candles,
            concurrency=concurrency,
        )
        count = await storage.candle_count()
        logger.info("Backfill done. %d new candles. DB total: %d rows", total, count)
