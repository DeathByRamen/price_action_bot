"""
Sentiment data clients for Fear & Greed Index and CryptoPanic news.

Fear & Greed: https://api.alternative.me/fng/ (no key required)
CryptoPanic: https://cryptopanic.com/api/v1/posts/ (free tier, 200 req/day)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

FEAR_GREED_URL = "https://api.alternative.me/fng/"
CRYPTOPANIC_URL = "https://cryptopanic.com/api/developer/posts/"


class SentimentClient:
    """Async client for sentiment data collection."""

    def __init__(
        self,
        cryptopanic_api_key: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        self._cp_key = cryptopanic_api_key or os.getenv("CRYPTOPANIC_API_KEY", "")
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self):
        if self._owns_session:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self._owns_session and self._session:
            await self._session.close()

    async def get_fear_greed(self, limit: int = 30) -> List[Dict]:
        """
        Fetch Fear & Greed Index history.

        Returns list of {ts, value, label} dicts.
        """
        try:
            params = {"limit": limit, "format": "json"}
            async with self._session.get(FEAR_GREED_URL, params=params) as resp:
                if resp.status != 200:
                    logger.warning("Fear & Greed API returned %d", resp.status)
                    return []
                data = await resp.json()

            results = []
            for entry in data.get("data", []):
                ts_epoch = int(entry.get("timestamp", 0))
                ts = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S"
                )
                results.append({
                    "ts": ts,
                    "value": float(entry.get("value", 50)),
                    "label": entry.get("value_classification", ""),
                })
            return results

        except Exception as exc:
            logger.warning("Fear & Greed fetch failed: %s", exc)
            return []

    async def get_crypto_news(
        self,
        currencies: Optional[List[str]] = None,
        kind: str = "news",
    ) -> List[Dict]:
        """
        Fetch recent crypto news with sentiment labels.

        Returns list of {ts, symbol, positive, negative, neutral, title} dicts.
        Requires a CryptoPanic API key for the free tier.
        """
        if not self._cp_key:
            return []

        try:
            params = {
                "auth_token": self._cp_key,
                "kind": kind,
                "filter": "hot",
                "public": "true",
            }
            if currencies:
                params["currencies"] = ",".join(currencies[:5])

            async with self._session.get(CRYPTOPANIC_URL, params=params) as resp:
                if resp.status != 200:
                    logger.warning("CryptoPanic API returned %d", resp.status)
                    return []
                data = await resp.json()

            results = []
            for post in data.get("results", []):
                votes = post.get("votes", {})
                currencies_list = post.get("currencies", [])
                if not currencies_list:
                    continue

                ts = post.get("published_at", "")[:19]
                for cur in currencies_list:
                    code = cur.get("code", "").upper()
                    if not code:
                        continue
                    results.append({
                        "ts": ts,
                        "symbol": f"{code}USDT",
                        "positive": votes.get("positive", 0),
                        "negative": votes.get("negative", 0),
                        "neutral": votes.get("important", 0),
                        "title": post.get("title", ""),
                    })

            return results

        except Exception as exc:
            logger.warning("CryptoPanic fetch failed: %s", exc)
            return []
