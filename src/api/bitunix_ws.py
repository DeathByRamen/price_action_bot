"""
BitUnix WebSocket client for real-time order book streaming.

Connects to wss://fapi.bitunix.com/public/ and subscribes to
depth_book15 channels for all futures symbols. Computes order book
metrics (imbalance, spread, depth) and writes snapshots to storage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import websockets
    HAS_WS = True
except ImportError:
    HAS_WS = False
    logger.info("websockets not installed — WebSocket streaming unavailable")

WS_URL = "wss://fapi.bitunix.com/public/"
RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 60.0
PING_INTERVAL = 20


class BitunixWebSocket:
    """
    Async WebSocket client for continuous order book streaming.

    Subscribes to depth_book15 for all provided symbols.
    Calls on_snapshot callback with computed metrics for each update.
    Auto-reconnects with exponential backoff on disconnection.
    """

    def __init__(
        self,
        symbols: List[str],
        on_snapshot: Optional[Callable] = None,
        batch_interval: float = 60.0,
    ):
        self.symbols = symbols
        self.on_snapshot = on_snapshot
        self.batch_interval = batch_interval
        self._running = False
        self._ws = None
        self._snapshot_buffer: Dict[str, dict] = {}
        self._last_flush = time.time()

    async def run(self) -> None:
        """Main loop: connect, subscribe, process messages, auto-reconnect."""
        if not HAS_WS:
            logger.error("websockets library not installed — cannot start WS client")
            return

        self._running = True
        delay = RECONNECT_BASE_DELAY

        while self._running:
            try:
                async with websockets.connect(
                    WS_URL,
                    ping_interval=PING_INTERVAL,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    logger.info("WebSocket connected to %s", WS_URL)
                    delay = RECONNECT_BASE_DELAY

                    await self._subscribe(ws)
                    await self._message_loop(ws)

            except Exception as exc:
                logger.warning("WebSocket disconnected: %s — reconnecting in %.0fs", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX_DELAY)

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()

    async def _subscribe(self, ws) -> None:
        for symbol in self.symbols:
            sub_msg = json.dumps({
                "op": "subscribe",
                "args": [{"channel": "depth_book15", "instId": symbol}],
            })
            await ws.send(sub_msg)
        logger.info("Subscribed to depth_book15 for %d symbols", len(self.symbols))

    async def _message_loop(self, ws) -> None:
        async for raw_msg in ws:
            try:
                data = json.loads(raw_msg)
                if "data" not in data:
                    continue

                symbol = data.get("arg", {}).get("instId", "")
                if not symbol:
                    continue

                snapshot = self._parse_depth(symbol, data["data"])
                if snapshot:
                    self._snapshot_buffer[symbol] = snapshot

                if time.time() - self._last_flush >= self.batch_interval:
                    await self._flush_buffer()

            except Exception as exc:
                logger.debug("WS message parse error: %s", exc)

    def _parse_depth(self, symbol: str, depth_data: dict) -> Optional[dict]:
        """Parse order book depth into metrics."""
        try:
            bids = depth_data.get("bids", [])
            asks = depth_data.get("asks", [])
            if not bids or not asks:
                return None

            bid_prices = [float(b[0]) for b in bids]
            bid_vols = [float(b[1]) for b in bids]
            ask_prices = [float(a[0]) for a in asks]
            ask_vols = [float(a[1]) for a in asks]

            best_bid = bid_prices[0]
            best_ask = ask_prices[0]
            mid_price = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid

            total_bid_vol = sum(bid_vols)
            total_ask_vol = sum(ask_vols)
            total = total_bid_vol + total_ask_vol
            imbalance = total_bid_vol / total if total > 0 else 0.5

            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

            return {
                "symbol": symbol,
                "ts": ts,
                "bid_prices": json.dumps(bid_prices),
                "bid_vols": json.dumps(bid_vols),
                "ask_prices": json.dumps(ask_prices),
                "ask_vols": json.dumps(ask_vols),
                "spread": spread,
                "mid_price": mid_price,
                "imbalance": imbalance,
            }
        except Exception:
            return None

    async def _flush_buffer(self) -> None:
        """Flush buffered snapshots via callback."""
        if not self._snapshot_buffer:
            return

        snapshots = list(self._snapshot_buffer.values())
        self._snapshot_buffer.clear()
        self._last_flush = time.time()

        if self.on_snapshot:
            try:
                await self.on_snapshot(snapshots)
            except Exception as exc:
                logger.error("Snapshot callback failed: %s", exc)

        logger.debug("Flushed %d order book snapshots", len(snapshots))
