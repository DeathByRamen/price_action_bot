#!/usr/bin/env python3
"""
Binance cross-exchange data collection — cron target.

Collects funding rates and open interest from Binance public API
for cross-exchange feature computation (funding rate spread, OI divergence).

Cron (hourly):
    5 * * * * cd /path/to/pa_bot && python scripts/collect_binance.py
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml

from src.api.binance_client import BinanceClient
from src.data.storage import Storage


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
    config = load_config(config_path)

    if not config.get("binance", {}).get("enabled", False):
        logging.info("Binance collection disabled in config")
        return

    db_path = config.get("storage", {}).get("db_path")

    async with BinanceClient() as client, Storage(db_path) as storage:
        fr_data = await client.get_funding_rates()
        if fr_data:
            rows = [(d["symbol"], d["ts"], d["funding_rate"]) for d in fr_data]
            n = await storage.insert_binance_funding_rate(rows)
            logging.info("Inserted %d Binance funding rate records", n)

        top_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]
        oi_data = await client.get_open_interest_batch(top_symbols)
        if oi_data:
            rows = [(d["symbol"], d["ts"], d["oi_value"]) for d in oi_data]
            n = await storage.insert_binance_oi(rows)
            logging.info("Inserted %d Binance OI records", n)


if __name__ == "__main__":
    asyncio.run(main())
