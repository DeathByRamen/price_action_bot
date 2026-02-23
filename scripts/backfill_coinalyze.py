#!/usr/bin/env python3
"""
One-time backfill of historical Coinalyze data.

Fetches:
  - Daily OI, liquidation, L/S ratio data (kept indefinitely by Coinalyze)
  - 15-min OI, liquidation, L/S ratio data (~15 days retained by Coinalyze)

Rate limit: 40 API calls/min (each symbol = 1 call).
Strategy: batch 20 symbols per request, one data type at a time.

Usage:
    python scripts/backfill_coinalyze.py [--db path] [--daily-days 365] [--intraday-days 14]
"""

import argparse
import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from src.api.bitunix_client import BitunixClient
from src.api.coinalyze_client import CoinalyzeClient
from src.data.collector import DataCollector
from src.data.storage import Storage


async def backfill(
    db_path: str | None = None,
    daily_days: int = 365,
    intraday_days: int = 14,
) -> dict:
    """
    Backfill historical Coinalyze data.

    Parameters
    ----------
    daily_days : int
        How many days of daily data to backfill (default 365).
    intraday_days : int
        How many days of 15-min data to backfill (default 14, max ~15).
    """
    stats = {
        "daily_oi": 0, "daily_liq": 0, "daily_ls": 0,
        "intraday_oi": 0, "intraday_liq": 0, "intraday_ls": 0,
        "symbols_mapped": 0,
    }

    async with CoinalyzeClient() as ca_client:
        if not ca_client.is_configured:
            logging.error(
                "COINALYZE_API_KEY not set. Get a free key at "
                "https://coinalyze.net/account/api-key/ and add it to .env"
            )
            return stats

        async with BitunixClient() as bx_client, Storage(db_path) as storage:
            collector = DataCollector(bx_client, storage)
            symbols = await collector.discover_tradeable_symbols()
            symbol_map = await ca_client.build_symbol_map(symbols)
            stats["symbols_mapped"] = len(symbol_map)

            if not symbol_map:
                logging.error("No symbols matched on Coinalyze")
                return stats

            reverse_map = {v: k for k, v in symbol_map.items()}
            now = int(time.time())

            # --- Daily backfill ---
            logging.info(
                "=== Backfilling %d days of DAILY data for %d symbols ===",
                daily_days, len(symbol_map),
            )
            daily_from = now - (daily_days * 86400)
            raw_daily = await ca_client.fetch_all_history(
                symbol_map,
                interval="daily",
                from_ts=daily_from,
                to_ts=now,
            )
            oi_rows, liq_rows, ls_rows = DataCollector.parse_coinalyze_results(
                raw_daily, reverse_map
            )
            if oi_rows:
                stats["daily_oi"] = await storage.insert_coinalyze_oi(oi_rows)
            if liq_rows:
                stats["daily_liq"] = await storage.insert_coinalyze_liquidations(liq_rows)
            if ls_rows:
                stats["daily_ls"] = await storage.insert_coinalyze_long_short(ls_rows)

            logging.info(
                "Daily backfill: OI=%d, Liq=%d, L/S=%d rows inserted",
                stats["daily_oi"], stats["daily_liq"], stats["daily_ls"],
            )

            # --- Intraday (15-min) backfill ---
            logging.info(
                "=== Backfilling %d days of 15-MIN data for %d symbols ===",
                intraday_days, len(symbol_map),
            )
            intraday_from = now - (intraday_days * 86400)
            raw_intraday = await ca_client.fetch_all_history(
                symbol_map,
                interval="15min",
                from_ts=intraday_from,
                to_ts=now,
            )
            oi_rows, liq_rows, ls_rows = DataCollector.parse_coinalyze_results(
                raw_intraday, reverse_map
            )
            if oi_rows:
                stats["intraday_oi"] = await storage.insert_coinalyze_oi(oi_rows)
            if liq_rows:
                stats["intraday_liq"] = await storage.insert_coinalyze_liquidations(liq_rows)
            if ls_rows:
                stats["intraday_ls"] = await storage.insert_coinalyze_long_short(ls_rows)

            logging.info(
                "Intraday backfill: OI=%d, Liq=%d, L/S=%d rows inserted",
                stats["intraday_oi"], stats["intraday_liq"], stats["intraday_ls"],
            )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill historical Coinalyze data (OI, liquidations, L/S)"
    )
    parser.add_argument("--db", type=str, default=None, help="Override SQLite DB path")
    parser.add_argument(
        "--daily-days", type=int, default=365,
        help="Days of daily data to backfill (default: 365)",
    )
    parser.add_argument(
        "--intraday-days", type=int, default=14,
        help="Days of 15-min data to backfill (default: 14, max ~15)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    stats = asyncio.run(backfill(args.db, args.daily_days, args.intraday_days))

    total = sum(stats[k] for k in stats if k != "symbols_mapped")
    logging.info(
        "Backfill complete — %d total rows inserted across %d symbols",
        total, stats["symbols_mapped"],
    )


if __name__ == "__main__":
    main()
