#!/usr/bin/env python3
"""
Collect open interest, liquidation, and long/short ratio data from
Coinalyze for all tradeable symbols.

Runs hourly on cron. Fetches the last 1 hour of data at 15-minute
granularity (4 data points per symbol per run).

Rate limit: 40 API calls/min. ~250 symbols x 3 data types = 750 calls
= ~19 minutes. Well within the 1-hour window.

Usage:
    python scripts/collect_coinalyze.py [--db path/to/ohlcv.db]

Cron (hourly at minute 10, after predictions at :05):
    10 * * * * cd /opt/pa_bot && /opt/pa_bot/venv/bin/python scripts/collect_coinalyze.py >> logs/coinalyze.log 2>&1
"""

import argparse
import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.bitunix_client import BitunixClient
from src.api.coinalyze_client import CoinalyzeClient
from src.data.collector import DataCollector
from src.data.storage import Storage


async def collect(db_path: str | None = None) -> dict:
    """Fetch Coinalyze data for the last hour at 15-min granularity."""
    stats = {"oi": 0, "liquidations": 0, "long_short": 0, "symbols_mapped": 0}

    async with CoinalyzeClient() as ca_client:
        if not ca_client.is_configured:
            logging.warning(
                "COINALYZE_API_KEY not set — skipping Coinalyze collection. "
                "Set it in .env or as an environment variable."
            )
            return stats

        async with BitunixClient() as bx_client, Storage(db_path) as storage:
            collector = DataCollector(bx_client, storage)

            symbols = await collector.discover_tradeable_symbols()
            symbol_map = await ca_client.build_symbol_map(symbols)
            stats["symbols_mapped"] = len(symbol_map)

            if not symbol_map:
                logging.warning("No symbols matched on Coinalyze — nothing to collect")
                return stats

            reverse_map = {v: k for k, v in symbol_map.items()}

            now = int(time.time())
            one_hour_ago = now - 3600

            raw = await ca_client.fetch_all_history(
                symbol_map,
                interval="15min",
                from_ts=one_hour_ago,
                to_ts=now,
            )

            oi_rows, liq_rows, ls_rows = DataCollector.parse_coinalyze_results(
                raw, reverse_map
            )

            if oi_rows:
                inserted = await storage.insert_coinalyze_oi(oi_rows)
                stats["oi"] = inserted
                logging.info("Inserted %d OI rows", inserted)

            if liq_rows:
                inserted = await storage.insert_coinalyze_liquidations(liq_rows)
                stats["liquidations"] = inserted
                logging.info("Inserted %d liquidation rows", inserted)

            if ls_rows:
                inserted = await storage.insert_coinalyze_long_short(ls_rows)
                stats["long_short"] = inserted
                logging.info("Inserted %d long/short rows", inserted)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Coinalyze derivatives data (OI, liquidations, L/S ratio)"
    )
    parser.add_argument("--db", type=str, default=None, help="Override SQLite DB path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    stats = asyncio.run(collect(args.db))
    logging.info(
        "Coinalyze collection complete — OI: %d, liquidations: %d, L/S: %d "
        "(from %d mapped symbols)",
        stats["oi"], stats["liquidations"], stats["long_short"],
        stats["symbols_mapped"],
    )


if __name__ == "__main__":
    main()
