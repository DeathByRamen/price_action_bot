"""
Binance public API client for cross-exchange data.

No API key required. Used for:
  - Funding rate comparison (BitUnix vs Binance spread)
  - Open interest cross-validation
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

BINANCE_FAPI_BASE = "https://fapi.binance.com"


class BinanceClient:
    """Async client for Binance public futures endpoints."""

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self):
        if self._owns_session:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self._owns_session and self._session:
            await self._session.close()

    async def get_funding_rates(self, limit: int = 100) -> List[Dict]:
        """
        Fetch latest funding rates for all perpetual contracts.

        Returns list of {symbol, ts, funding_rate}.
        """
        try:
            url = f"{BINANCE_FAPI_BASE}/fapi/v1/premiumIndex"
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    logger.warning("Binance funding rate API returned %d", resp.status)
                    return []
                data = await resp.json()

            results = []
            for item in data[:limit]:
                symbol = item.get("symbol", "")
                if not symbol.endswith("USDT"):
                    continue
                ts = datetime.fromtimestamp(
                    item.get("time", 0) / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%S")
                results.append({
                    "symbol": symbol,
                    "ts": ts,
                    "funding_rate": float(item.get("lastFundingRate", 0)),
                })
            return results

        except Exception as exc:
            logger.warning("Binance funding rates failed: %s", exc)
            return []

    async def get_open_interest(self, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """
        Fetch current open interest for a single symbol.

        Returns {symbol, ts, oi_value} or None.
        """
        try:
            url = f"{BINANCE_FAPI_BASE}/fapi/v1/openInterest"
            async with self._session.get(url, params={"symbol": symbol}) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

            ts = datetime.fromtimestamp(
                data.get("time", 0) / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S")
            return {
                "symbol": data.get("symbol", symbol),
                "ts": ts,
                "oi_value": float(data.get("openInterest", 0)),
            }

        except Exception as exc:
            logger.warning("Binance OI for %s failed: %s", symbol, exc)
            return None

    async def get_open_interest_batch(
        self, symbols: List[str]
    ) -> List[Dict]:
        """Fetch OI for multiple symbols sequentially."""
        results = []
        for sym in symbols:
            oi = await self.get_open_interest(sym)
            if oi:
                results.append(oi)
        return results
