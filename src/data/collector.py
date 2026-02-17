"""
Data collector: orchestrates fetching candles from BitUnix and persisting to storage.

Used both for incremental hourly fetches and one-time backfills.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from src.api.bitunix_client import BitunixClient, Candle, Ticker
from src.data.storage import Storage

logger = logging.getLogger(__name__)


class DataCollector:
    """Coordinates data retrieval from BitUnix and storage."""

    def __init__(self, client: BitunixClient, storage: Storage):
        self.client = client
        self.storage = storage

    # ------------------------------------------------------------------
    # Symbol discovery
    # ------------------------------------------------------------------
    async def discover_futures_symbols(self) -> List[str]:
        """Fetch all available futures symbols from the tickers endpoint."""
        tickers = await self.client.get_all_tickers()
        symbols = [t.symbol for t in tickers]
        logger.info("Discovered %d futures symbols", len(symbols))
        return sorted(symbols)

    # ------------------------------------------------------------------
    # Incremental fetch (hourly run)
    # ------------------------------------------------------------------
    async def fetch_latest_candles(
        self,
        symbols: List[str],
        interval: str = "60",
        lookback: int = 5,
        concurrency: int = 8,
    ) -> int:
        """
        For each symbol, fetch the last *lookback* candles to fill gaps
        since the previous run.  Deduplication happens at the storage layer.

        Uses bounded concurrency via semaphore for faster throughput
        (rate limiting is still enforced at the HTTP client level).

        Returns total number of new candles inserted.
        """
        semaphore = asyncio.Semaphore(concurrency)
        total_inserted = 0

        async def _fetch_one(symbol: str) -> int:
            async with semaphore:
                try:
                    candles = await self.client.get_kline_history(
                        symbol=symbol, interval=interval, limit=lookback
                    )
                    if candles:
                        rows = self._candles_to_rows(candles)
                        inserted = await self.storage.insert_candles(rows, interval=interval)
                        if inserted:
                            logger.debug("%s: +%d candles", symbol, inserted)
                        return inserted
                except Exception as exc:
                    logger.warning("Failed to fetch candles for %s: %s", symbol, exc)
                return 0

        results = await asyncio.gather(
            *[_fetch_one(s) for s in symbols], return_exceptions=True
        )

        for sym, res in zip(symbols, results):
            if isinstance(res, Exception):
                logger.error("Fetch failed for %s: %s", sym, res)
            else:
                total_inserted += res

        logger.info(
            "Incremental fetch complete: %d new candles across %d symbols",
            total_inserted,
            len(symbols),
        )
        return total_inserted

    # ------------------------------------------------------------------
    # Backfill (one-time historical download)
    # ------------------------------------------------------------------
    async def backfill_symbol(
        self,
        symbol: str,
        interval: str = "60",
        total_candles: int = 2000,
        batch_size: int = 500,
    ) -> int:
        """
        Download up to *total_candles* historical candles for a single symbol,
        paging backwards from the most recent data.

        The Spot kline/history endpoint returns candles in chronological order
        ending just before *endTime* (exclusive upper bound, in seconds).
        """
        inserted_total = 0
        end_time: Optional[int] = None
        remaining = total_candles

        while remaining > 0:
            fetch_count = min(remaining, batch_size)
            try:
                candles = await self.client.get_kline_history(
                    symbol=symbol,
                    interval=interval,
                    limit=fetch_count,
                    end_time=end_time,
                )
            except Exception as exc:
                logger.error("Backfill request failed for %s: %s", symbol, exc)
                break

            if not candles:
                logger.debug("%s: no more candles returned", symbol)
                break

            rows = self._candles_to_rows(candles)
            inserted = await self.storage.insert_candles(rows, interval=interval)
            inserted_total += inserted

            earliest_ts = candles[0].ts
            try:
                from datetime import datetime, timezone

                dt = datetime.fromisoformat(earliest_ts.replace("Z", "+00:00"))
                end_time = int(dt.timestamp())
            except Exception:
                logger.warning(
                    "%s: could not parse earliest ts '%s', stopping backfill",
                    symbol,
                    earliest_ts,
                )
                break

            remaining -= len(candles)
            if len(candles) < fetch_count:
                break

        logger.info("%s: backfilled %d candles", symbol, inserted_total)
        return inserted_total

    async def backfill_all(
        self,
        symbols: List[str],
        interval: str = "60",
        total_candles: int = 2000,
        concurrency: int = 5,
    ) -> int:
        """Backfill multiple symbols with bounded concurrency."""
        semaphore = asyncio.Semaphore(concurrency)
        total = 0

        async def _backfill_one(sym: str) -> int:
            async with semaphore:
                return await self.backfill_symbol(
                    sym, interval=interval, total_candles=total_candles
                )

        results = await asyncio.gather(
            *[_backfill_one(s) for s in symbols], return_exceptions=True
        )
        for sym, res in zip(symbols, results):
            if isinstance(res, Exception):
                logger.error("Backfill failed for %s: %s", sym, res)
            else:
                total += res

        logger.info(
            "Backfill complete: %d total candles for %d symbols", total, len(symbols)
        )
        return total

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _candles_to_rows(candles: List[Candle]) -> List[tuple]:
        return [
            (c.symbol, c.ts, c.open, c.high, c.low, c.close, c.volume)
            for c in candles
        ]
