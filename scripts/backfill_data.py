#!/usr/bin/env python3
"""
One-time (or periodic) historical data backfill script.

Usage:
    python scripts/backfill_data.py [--candles 2000] [--concurrency 5] [--symbols BTCUSDT,ETHUSDT]
"""

import argparse
import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.backfill import run_backfill


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill BitUnix kline data")
    parser.add_argument(
        "--candles",
        type=int,
        default=2000,
        help="Number of hourly candles to backfill per symbol (default: 2000 ≈ 83 days)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent API requests (default: 5)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols to backfill (default: all futures pairs)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to SQLite database file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symbols = args.symbols.split(",") if args.symbols else None

    asyncio.run(
        run_backfill(
            db_path=args.db,
            symbols=symbols,
            total_candles=args.candles,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
