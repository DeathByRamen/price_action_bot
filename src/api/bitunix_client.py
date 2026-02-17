"""
Async client for BitUnix public market-data APIs.

Covers:
  - Futures tickers   (fapi.bitunix.com)
  - Spot kline history (openapi.bitunix.com)  -- used for OHLCV backfill
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

FUTURES_BASE = "https://fapi.bitunix.com"
SPOT_BASE = "https://openapi.bitunix.com"


@dataclass
class Ticker:
    symbol: str
    last_price: float
    mark_price: float
    open_24h: float
    high_24h: float
    low_24h: float
    base_vol: float
    quote_vol: float


@dataclass
class Candle:
    symbol: str
    ts: str  # ISO-8601 timestamp string from API
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class BitunixClient:
    """Lightweight async wrapper around BitUnix REST endpoints."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        max_retries: int = 3,
        rate_limit: float = 9.0,
    ):
        self._api_key = api_key
        self._api_secret = api_secret
        self._max_retries = max_retries
        self._limiter = RateLimiter(max_rate=rate_limit)
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    async def __aenter__(self) -> "BitunixClient":
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sign(self, body_str: str, timestamp: str, nonce: str) -> str:
        """HMAC-SHA256 signature per BitUnix docs (for future authenticated calls)."""
        digest_input = nonce + timestamp + self._api_key + body_str
        body_hash = hashlib.sha256(digest_input.encode()).hexdigest()
        sign_input = body_hash + self._api_secret
        return hashlib.sha256(sign_input.encode()).hexdigest()

    def _auth_headers(self, body_str: str = "") -> Dict[str, str]:
        ts = str(int(time.time() * 1000))
        nonce = uuid.uuid4().hex
        return {
            "Content-Type": "application/json",
            "api-key": self._api_key,
            "timestamp": ts,
            "nonce": nonce,
            "sign": self._sign(body_str, ts, nonce),
        }

    async def _request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, str]] = None,
        auth: bool = False,
    ) -> Any:
        """Issue an HTTP request with rate limiting and retries."""
        assert self._session is not None, "Client not opened. Use `async with`."

        headers = self._auth_headers() if auth else {"Content-Type": "application/json"}

        for attempt in range(1, self._max_retries + 1):
            await self._limiter.acquire()
            try:
                async with self._session.request(
                    method, url, params=params, headers=headers
                ) as resp:
                    if resp.status == 429:
                        wait = 2 ** attempt
                        logger.warning("Rate-limited (429). Backing off %ss", wait)
                        await asyncio.sleep(wait)
                        continue

                    resp.raise_for_status()
                    data = await resp.json()

                    # BitUnix returns code as string "0" for success
                    code = data.get("code")
                    if str(code) != "0":
                        logger.error(
                            "API error code=%s msg=%s url=%s",
                            code,
                            data.get("msg"),
                            url,
                        )
                        if attempt < self._max_retries:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return data

                    return data

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "Request failed (attempt %d/%d): %s %s -> %s",
                    attempt,
                    self._max_retries,
                    method,
                    url,
                    exc,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

    # ------------------------------------------------------------------
    # Futures endpoints (fapi.bitunix.com)
    # ------------------------------------------------------------------
    async def get_all_tickers(
        self, symbols: Optional[List[str]] = None
    ) -> List[Ticker]:
        """
        GET /api/v1/futures/market/tickers

        Returns 24-hour rolling ticker data for all (or specified) futures pairs.
        No authentication required.
        """
        url = f"{FUTURES_BASE}/api/v1/futures/market/tickers"
        params: Dict[str, str] = {}
        if symbols:
            params["symbols"] = ",".join(symbols)

        data = await self._request("GET", url, params=params or None)
        tickers: List[Ticker] = []

        for item in data.get("data", []):
            try:
                tickers.append(
                    Ticker(
                        symbol=item["symbol"],
                        last_price=float(item.get("lastPrice", 0)),
                        mark_price=float(item.get("markPrice", 0)),
                        open_24h=float(item.get("open", 0)),
                        high_24h=float(item.get("high", 0)),
                        low_24h=float(item.get("low", 0)),
                        base_vol=float(item.get("baseVol", 0)),
                        quote_vol=float(item.get("quoteVol", 0)),
                    )
                )
            except (KeyError, ValueError) as exc:
                logger.debug("Skipping malformed ticker: %s -> %s", item, exc)
        return tickers

    # ------------------------------------------------------------------
    # Spot endpoints (openapi.bitunix.com) -- used for kline backfill
    # ------------------------------------------------------------------
    async def get_kline_history(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 500,
        end_time: Optional[int] = None,
    ) -> List[Candle]:
        """
        GET /api/spot/v1/market/kline/history

        Fetch up to 500 historical candles for *symbol* at the given interval.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. "BTCUSDT"
        interval : str
            Candle interval.  "60" = 1 hour.
            Supported: 1,3,5,15,30,60,120,240,360,720,D,M,W
        limit : int
            Number of candles (1-500). Default 500.
        end_time : int | None
            End timestamp in **seconds** (exclusive upper bound).
            If None, the API returns candles up to the current time.
        """
        url = f"{SPOT_BASE}/api/spot/v1/market/kline/history"
        params: Dict[str, str] = {
            "symbol": symbol.lower(),
            "interval": interval,
            "limit": str(min(limit, 500)),
        }
        if end_time is not None:
            params["endTime"] = str(end_time)

        data = await self._request("GET", url, params=params)
        candles: List[Candle] = []

        for item in data.get("data", []) or []:
            try:
                candles.append(
                    Candle(
                        symbol=symbol,
                        ts=item["ts"],
                        open=float(item["open"]),
                        high=float(item["high"]),
                        low=float(item["low"]),
                        close=float(item["close"]),
                        volume=float(item.get("volume", 0)),
                    )
                )
            except (KeyError, ValueError) as exc:
                logger.debug("Skipping malformed candle: %s -> %s", item, exc)
        return candles

    async def get_latest_kline(
        self, symbol: str, interval: str = "60"
    ) -> Optional[Candle]:
        """
        GET /api/spot/v1/market/kline

        Returns the single most-recent candle for the given pair / interval.
        """
        url = f"{SPOT_BASE}/api/spot/v1/market/kline"
        params = {"symbol": symbol.lower(), "interval": interval}
        data = await self._request("GET", url, params=params)
        item = data.get("data")
        if not item:
            return None
        try:
            return Candle(
                symbol=item.get("symbol", symbol),
                ts=item["ts"],
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=float(item["close"]),
                volume=float(item.get("volume", 0)),
            )
        except (KeyError, ValueError) as exc:
            logger.debug("Malformed latest kline: %s -> %s", item, exc)
            return None

    async def get_coin_pairs(self) -> List[Dict[str, Any]]:
        """
        GET /api/spot/v1/common/coin_pair/list

        Returns metadata for all spot trading pairs.
        """
        url = f"{SPOT_BASE}/api/spot/v1/common/coin_pair/list"
        data = await self._request("GET", url)
        return data.get("data", []) or []
