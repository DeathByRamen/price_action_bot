#!/usr/bin/env python3
"""
One-time (or periodic) historical data backfill script.

Usage:
    python scripts/backfill_data.py [--candles 2000] [--concurrency 5] [--symbols BTCUSDT,ETHUSDT]
"""

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.backfill import run_backfill


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill BitUnix kline data")
    parser.add_argument(
        "--candles",
        type=int,
        default=2000,
        help="Number of candles to backfill per symbol (default: 2000 ≈ 83 days of 1h)",
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
    parser.add_argument(
        "--interval",
        type=str,
        default="60",
        help="Candle interval: 1,3,5,15,30,60,120,240,360,720,D,W,M (default: 60)",
    )
    parser.add_argument(
        "--all-timeframes",
        action="store_true",
        help="Backfill both 1h and 15m candles (auto-scales candle count for 15m)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symbols = args.symbols.split(",") if args.symbols else None

    if args.all_timeframes:
        logging.info("Backfilling all timeframes: 60m + 15m")
        # 1h backfill
        logging.info("--- Backfilling 1h candles (%d per symbol) ---", args.candles)
        asyncio.run(
            run_backfill(
                db_path=args.db,
                symbols=symbols,
                total_candles=args.candles,
                concurrency=args.concurrency,
                interval="60",
            )
        )
        # 15m backfill (4x the candles to cover the same time window)
        candles_15m = args.candles * 4
        logging.info("--- Backfilling 15m candles (%d per symbol) ---", candles_15m)
        asyncio.run(
            run_backfill(
                db_path=args.db,
                symbols=symbols,
                total_candles=candles_15m,
                concurrency=args.concurrency,
                interval="15",
            )
        )
    else:
        asyncio.run(
            run_backfill(
                db_path=args.db,
                symbols=symbols,
                total_candles=args.candles,
                concurrency=args.concurrency,
                interval=args.interval,
            )
        )


if __name__ == "__main__":
    main()
