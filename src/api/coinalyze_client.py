"""
Async client for Coinalyze public market-data API.

Provides open interest, liquidation, and long/short ratio data
aggregated across major exchanges (Binance, Bybit, OKX, etc.).

API docs: https://api.coinalyze.net/v1/doc/
Rate limit: 40 API calls per minute (each symbol in a batch = 1 call).
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp

from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

COINALYZE_BASE = "https://api.coinalyze.net/v1"

# 40 calls/min = 0.667 calls/sec — use 0.45 for conservative safety margin
_RATE_LIMIT = 0.45
_BURST = 1


@dataclass
class CoinalyzeMarket:
    """Represents a supported futures market on Coinalyze."""
    symbol: str           # e.g. "BTCUSDT_PERP.A"
    exchange: str         # e.g. "A" (Binance)
    base_asset: str       # e.g. "BTC"
    quote_asset: str      # e.g. "USDT"
    is_perpetual: bool
    has_long_short: bool
    has_ohlcv: bool
    has_buy_sell: bool


class CoinalyzeClient:
    """Async wrapper around Coinalyze REST API."""

    def __init__(
        self,
        api_key: str = "",
        exchange: str = "A",
        max_retries: int = 3,
    ):
        self._api_key = api_key or os.getenv("COINALYZE_API_KEY", "")
        self._exchange = exchange
        self._max_retries = max_retries
        self._limiter = RateLimiter(max_rate=_RATE_LIMIT, burst=_BURST)
        self._session: Optional[aiohttp.ClientSession] = None
        self._symbol_map: Optional[Dict[str, str]] = None

    async def __aenter__(self) -> "CoinalyzeClient":
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Issue a GET request with rate limiting and retries."""
        assert self._session is not None, "Client not opened. Use `async with`."

        if not self._api_key:
            logger.warning("Coinalyze API key not configured — skipping request")
            return None

        url = f"{COINALYZE_BASE}/{endpoint}"
        if params is None:
            params = {}
        params["api_key"] = self._api_key

        for attempt in range(1, self._max_retries + 1):
            await self._limiter.acquire()
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 429:
                        retry_after = float(resp.headers.get("Retry-After", 10))
                        retry_after = max(retry_after, 60.0)
                        logger.warning(
                            "Coinalyze rate-limited (429). Waiting %.0fs "
                            "(attempt %d/%d)",
                            retry_after, attempt, self._max_retries,
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status == 401:
                        logger.error("Coinalyze: invalid API key")
                        return None

                    resp.raise_for_status()
                    return await resp.json()

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "Coinalyze request failed (attempt %d/%d): %s -> %s",
                    attempt, self._max_retries, endpoint, exc,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
        return None

    async def _request_with_symbol_cost(
        self,
        endpoint: str,
        symbols: List[str],
        params: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Make a request where each symbol in the batch costs 1 API call.

        Pre-acquires rate limit tokens for len(symbols) - 1 additional calls
        (the _request method acquires 1 token itself).
        """
        for _ in range(len(symbols) - 1):
            await self._limiter.acquire()
        if params is None:
            params = {}
        params["symbols"] = ",".join(symbols)
        return await self._request(endpoint, params)

    # ------------------------------------------------------------------
    # Symbol mapping
    # ------------------------------------------------------------------
    async def get_supported_markets(self) -> List[CoinalyzeMarket]:
        """Fetch all supported futures markets."""
        data = await self._request("future-markets")
        if not data or not isinstance(data, list):
            return []

        markets = []
        for item in data:
            markets.append(CoinalyzeMarket(
                symbol=item.get("symbol", ""),
                exchange=item.get("exchange", ""),
                base_asset=item.get("base_asset", ""),
                quote_asset=item.get("quote_asset", ""),
                is_perpetual=item.get("is_perpetual", False),
                has_long_short=item.get("has_long_short_ratio_data", False),
                has_ohlcv=item.get("has_ohlcv_data", False),
                has_buy_sell=item.get("has_buy_sell_data", False),
            ))
        return markets

    async def build_symbol_map(
        self, our_symbols: List[str]
    ) -> Dict[str, str]:
        """
        Build a mapping from our symbol names (e.g. "BTCUSDT") to
        Coinalyze symbols (e.g. "BTCUSDT_PERP.A").

        Only includes perpetual contracts on the configured exchange.
        Caches the result for subsequent calls.
        """
        if self._symbol_map is not None:
            return self._symbol_map

        markets = await self.get_supported_markets()
        if not markets:
            self._symbol_map = {}
            return self._symbol_map

        ca_by_base_quote: Dict[str, str] = {}
        for m in markets:
            if m.exchange == self._exchange and m.is_perpetual:
                key = f"{m.base_asset}{m.quote_asset}"
                ca_by_base_quote[key] = m.symbol

        self._symbol_map = {}
        for sym in our_symbols:
            normalized = sym.upper().replace("-", "")
            if normalized in ca_by_base_quote:
                self._symbol_map[sym] = ca_by_base_quote[normalized]

        logger.info(
            "Coinalyze symbol mapping: %d of %d symbols matched on exchange %s",
            len(self._symbol_map), len(our_symbols), self._exchange,
        )
        return self._symbol_map

    # ------------------------------------------------------------------
    # History endpoints
    # ------------------------------------------------------------------
    async def get_open_interest_history(
        self,
        symbols: List[str],
        interval: str = "15min",
        from_ts: int = 0,
        to_ts: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        GET /open-interest-history

        Returns OI OHLC data. Each entry has:
        {"symbol": str, "history": [{"t": unix_s, "o": float, "h": float, "l": float, "c": float}]}
        """
        params: Dict[str, str] = {"interval": interval}
        if from_ts:
            params["from"] = str(from_ts)
        if to_ts:
            params["to"] = str(to_ts)

        data = await self._request_with_symbol_cost(
            "open-interest-history", symbols, params
        )
        if data and isinstance(data, list):
            return data
        return []

    async def get_liquidation_history(
        self,
        symbols: List[str],
        interval: str = "15min",
        from_ts: int = 0,
        to_ts: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        GET /liquidation-history

        Returns liquidation data. Each entry has:
        {"symbol": str, "history": [{"t": unix_s, "l": long_vol, "s": short_vol}]}
        """
        params: Dict[str, str] = {"interval": interval}
        if from_ts:
            params["from"] = str(from_ts)
        if to_ts:
            params["to"] = str(to_ts)

        data = await self._request_with_symbol_cost(
            "liquidation-history", symbols, params
        )
        if data and isinstance(data, list):
            return data
        return []

    async def get_long_short_ratio_history(
        self,
        symbols: List[str],
        interval: str = "15min",
        from_ts: int = 0,
        to_ts: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        GET /long-short-ratio-history

        Returns L/S ratio data. Each entry has:
        {"symbol": str, "history": [{"t": unix_s, "r": ratio, "l": long_pct, "s": short_pct}]}
        """
        params: Dict[str, str] = {"interval": interval}
        if from_ts:
            params["from"] = str(from_ts)
        if to_ts:
            params["to"] = str(to_ts)

        data = await self._request_with_symbol_cost(
            "long-short-ratio-history", symbols, params
        )
        if data and isinstance(data, list):
            return data
        return []

    # ------------------------------------------------------------------
    # Batch collection helper
    # ------------------------------------------------------------------
    async def fetch_all_history(
        self,
        symbol_map: Dict[str, str],
        interval: str = "15min",
        from_ts: int = 0,
        to_ts: int = 0,
        batch_size: int = 10,
    ) -> Dict[str, List]:
        """
        Fetch OI, liquidation, and L/S ratio history for all mapped symbols.

        Uses small batches (default 10) with explicit pacing between batches
        to stay well under the 40-calls/min API rate limit.  Each batch
        makes 3 requests (OI, liquidation, L/S), costing batch_size × 3
        API calls.  A 45-second cooldown between batches ensures no minute
        window ever exceeds the limit.
        """
        ca_symbols = list(symbol_map.values())
        batches = [
            ca_symbols[i:i + batch_size]
            for i in range(0, len(ca_symbols), batch_size)
        ]

        total_batches = len(batches)
        est_minutes = (total_batches * 50) / 60
        logger.info(
            "Coinalyze fetch: %d symbols in %d batches of %d "
            "(~%.0f min estimated)",
            len(ca_symbols), total_batches, batch_size, est_minutes,
        )

        all_oi: List[Dict] = []
        all_liq: List[Dict] = []
        all_ls: List[Dict] = []

        for batch_idx, batch in enumerate(batches):
            logger.info(
                "Coinalyze batch %d/%d (%d symbols)",
                batch_idx + 1, total_batches, len(batch),
            )

            oi_data = await self.get_open_interest_history(
                batch, interval, from_ts, to_ts
            )
            all_oi.extend(oi_data)

            liq_data = await self.get_liquidation_history(
                batch, interval, from_ts, to_ts
            )
            all_liq.extend(liq_data)

            ls_data = await self.get_long_short_ratio_history(
                batch, interval, from_ts, to_ts
            )
            all_ls.extend(ls_data)

            if batch_idx < total_batches - 1:
                logger.debug("Cooling down 45s before next batch...")
                await asyncio.sleep(45)

        logger.info(
            "Coinalyze fetch complete: %d OI, %d liquidation, %d L/S entries",
            len(all_oi), len(all_liq), len(all_ls),
        )

        return {
            "oi": all_oi,
            "liquidations": all_liq,
            "long_short": all_ls,
        }
