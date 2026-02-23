"""
Convenience wrapper for backfill operations.

Can be imported or run directly via `scripts/backfill_data.py`.
"""

from __future__ import annotations

import logging

from src.api.bitunix_client import BitunixClient
from src.data.collector import DataCollector
from src.data.storage import Storage

logger = logging.getLogger(__name__)


async def _get_spot_symbols(client: BitunixClient) -> set[str]:
    """Fetch all open spot trading pair symbols (lowercase)."""
    pairs = await client.get_coin_pairs()
    spot = set()
    for p in pairs:
        sym = p.get("symbol", "")
        if p.get("isOpen") == 1 and sym:
            spot.add(sym.lower())
    logger.info("Found %d open spot pairs", len(spot))
    return spot


async def run_backfill(
    db_path: str | None = None,
    symbols: list[str] | None = None,
    interval: str = "60",
    total_candles: int = 2000,
    concurrency: int = 5,
) -> None:
    """
    Backfill historical kline data from BitUnix.

    Supports **resume**: checks how many candles already exist per symbol
    and only fetches the remaining. Fully backfilled symbols are skipped.

    If *symbols* is None, we first discover all futures pairs from the
    tickers endpoint, then filter to only those with a matching spot pair
    (since kline history comes from the spot API).
    """
    async with BitunixClient() as client, Storage(db_path) as storage:
        collector = DataCollector(client, storage)

        # Get the set of valid spot symbols for cross-referencing
        spot_symbols = await _get_spot_symbols(client)

        if symbols is None:
            futures_symbols = await collector.discover_futures_symbols()
            symbols = [
                s for s in futures_symbols
                if s.lower() in spot_symbols
            ]
            skipped = len(futures_symbols) - len(symbols)
            logger.info(
                "Will backfill %d symbols (%d futures-only symbols skipped — no spot kline data)",
                len(symbols), skipped,
            )
        else:
            missing = [s for s in symbols if s.lower() not in spot_symbols]
            if missing:
                logger.warning(
                    "These symbols don't exist on the spot market and will likely fail: %s",
                    ", ".join(missing),
                )

        # Resume logic: check existing candle counts per symbol
        symbols_to_fetch: list[str] = []
        candles_needed: dict[str, int] = {}
        already_complete = 0

        for sym in symbols:
            existing = await storage.candle_count_for_symbol(sym, interval=interval)
            if existing >= total_candles:
                already_complete += 1
                continue
            remaining = total_candles - existing
            symbols_to_fetch.append(sym)
            candles_needed[sym] = remaining

        if already_complete:
            logger.info(
                "Backfill resume: %d symbols already complete, %d need more data",
                already_complete, len(symbols_to_fetch),
            )

        if not symbols_to_fetch:
            count = await storage.candle_count()
            logger.info("Backfill: all symbols complete. DB total: %d rows", count)
            return

        total = await collector.backfill_all(
            symbols_to_fetch,
            interval=interval,
            total_candles=total_candles,
            concurrency=concurrency,
        )
        count = await storage.candle_count()
        logger.info("Backfill done. %d new candles. DB total: %d rows", total, count)
