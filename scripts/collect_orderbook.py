#!/usr/bin/env python3
"""
Collect order book depth snapshots AND funding rate snapshots for all
tradeable futures symbols.

Designed to run on a cron schedule (every 15 minutes) to accumulate
market microstructure data that will later be used as features for the
LSTM model.

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


async def collect(db_path: str | None = None) -> dict:
    """Fetch order book and funding rate snapshots for all symbols."""
    stats = {"order_book": 0, "funding_rate": 0}

    async with BitunixClient() as client, Storage(db_path) as storage:
        collector = DataCollector(client, storage)

        symbols = await collector.discover_tradeable_symbols()
        logging.info(
            "Collecting snapshots for %d symbols...", len(symbols)
        )

        ob_rows, fr_rows = await asyncio.gather(
            collector.fetch_order_book_snapshots(symbols),
            collector.fetch_funding_rate_snapshots(symbols),
        )

        if ob_rows:
            inserted = await storage.insert_order_book_snapshots(ob_rows)
            stats["order_book"] = inserted
            logging.info("Inserted %d order book snapshots", inserted)
        else:
            logging.warning("No order book snapshots collected")

        if fr_rows:
            inserted = await storage.insert_funding_rate_snapshots(fr_rows)
            stats["funding_rate"] = inserted
            logging.info("Inserted %d funding rate snapshots", inserted)
        else:
            logging.warning("No funding rate snapshots collected")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect order book depth and funding rate snapshots"
    )
    parser.add_argument("--db", type=str, default=None, help="Override SQLite DB path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    stats = asyncio.run(collect(args.db))
    logging.info(
        "Collection complete — order book: %d, funding rate: %d",
        stats["order_book"],
        stats["funding_rate"],
    )


if __name__ == "__main__":
    main()
