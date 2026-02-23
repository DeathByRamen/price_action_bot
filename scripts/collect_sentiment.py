#!/usr/bin/env python3
"""
Sentiment data collection — cron target.

Collects Fear & Greed Index and CryptoPanic news sentiment hourly.

Cron:
    0 * * * * cd /path/to/pa_bot && python scripts/collect_sentiment.py
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml

from src.api.sentiment_client import SentimentClient
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
    sentiment_cfg = config.get("sentiment", {})

    if not sentiment_cfg.get("enabled", False):
        logging.info("Sentiment collection disabled in config")
        return

    cp_key = sentiment_cfg.get("cryptopanic_api_key") or os.getenv("CRYPTOPANIC_API_KEY", "")
    db_path = config.get("storage", {}).get("db_path")

    async with SentimentClient(cryptopanic_api_key=cp_key) as client, Storage(db_path) as storage:
        if sentiment_cfg.get("fear_greed_enabled", True):
            fg_data = await client.get_fear_greed(limit=10)
            if fg_data:
                rows = [(d["ts"], d["value"], d["label"]) for d in fg_data]
                n = await storage.insert_fear_greed(rows)
                logging.info("Inserted %d Fear & Greed records", n)

        if cp_key:
            news = await client.get_crypto_news()
            if news:
                rows = [
                    (d["symbol"], d["ts"], d["positive"], d["negative"], d["neutral"],
                     d["positive"] + d["negative"] + d["neutral"])
                    for d in news
                ]
                n = await storage.insert_news_sentiment(rows)
                logging.info("Inserted %d news sentiment records", n)


if __name__ == "__main__":
    asyncio.run(main())
