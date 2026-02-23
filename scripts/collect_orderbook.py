#!/usr/bin/env python3
"""
Collect order book depth snapshots for all tradeable symbols.

Designed to run on a cron schedule (every 15 minutes) to accumulate
order book data that will later be used as features for the LSTM model.

Usage:
    python scripts/collect_orderbook.py [--db path/to/ohlcv.db]

Cron (every 15 minutes):
    */15 * * * * cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/collect_orderbook.py >> logs/orderbook.log 2>&1
"""

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.bitunix_client import BitunixClient
from src.data.collector import DataCollector
from src.data.storage import Storage


async def collect(db_path: str | None = None) -> int:
    """Fetch order book snapshots for all symbols and store them."""
    async with BitunixClient() as client, Storage(db_path) as storage:
        collector = DataCollector(client, storage)
        symbols = await collector.discover_tradeable_symbols()
        logging.info("Collecting order book snapshots for %d symbols...", len(symbols))

        rows = await collector.fetch_order_book_snapshots(symbols)
        if rows:
            inserted = await storage.insert_order_book_snapshots(rows)
            logging.info("Inserted %d order book snapshots", inserted)
            return inserted

        logging.warning("No order book snapshots collected")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect order book depth snapshots"
    )
    parser.add_argument("--db", type=str, default=None, help="Override SQLite DB path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    total = asyncio.run(collect(args.db))
    logging.info("Order book collection complete: %d snapshots", total)


if __name__ == "__main__":
    main()
