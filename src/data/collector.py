"""
Data collector: orchestrates fetching candles from BitUnix and persisting to storage.

Used both for incremental hourly fetches and one-time backfills.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from src.api.bitunix_client import BitunixClient, Candle, Ticker
from src.data.quality import validate_candles
from src.data.storage import Storage

logger = logging.getLogger(__name__)


class DataCollector:
    """Coordinates data retrieval from BitUnix and storage."""

    def __init__(self, client: BitunixClient, storage: Storage):
        self.client = client
        self.storage = storage

    # ------------------------------------------------------------------
    # Symbol discovery
    # ------------------------------------------------------------------
    async def discover_futures_symbols(self) -> List[str]:
        """Fetch all available futures symbols from the tickers endpoint."""
        tickers = await self.client.get_all_tickers()
        symbols = [t.symbol for t in tickers]
        logger.info("Discovered %d futures symbols", len(symbols))
        return sorted(symbols)

    async def discover_tradeable_symbols(self) -> List[str]:
        """
        Return futures symbols that also have a spot pair
        (needed because kline history comes from the spot API).
        """
        futures = await self.discover_futures_symbols()
        spot_pairs = await self.client.get_coin_pairs()
        spot_set = {
            p.get("symbol", "").lower()
            for p in spot_pairs
            if p.get("isOpen") == 1
        }
        valid = [s for s in futures if s.lower() in spot_set]
        skipped = len(futures) - len(valid)
        if skipped:
            logger.info(
                "%d futures symbols have no spot pair (skipped for kline data)",
                skipped,
            )
        logger.info("Using %d symbols with both futures + spot data", len(valid))
        return valid

    # ------------------------------------------------------------------
    # Incremental fetch (hourly run)
    # ------------------------------------------------------------------
    async def fetch_latest_candles(
        self,
        symbols: List[str],
        interval: str = "60",
        lookback: int = 5,
        concurrency: int = 8,
    ) -> int:
        """
        For each symbol, fetch the last *lookback* candles to fill gaps
        since the previous run.  Deduplication happens at the storage layer.

        Uses bounded concurrency via semaphore for faster throughput
        (rate limiting is still enforced at the HTTP client level).

        Returns total number of new candles inserted.
        """
        semaphore = asyncio.Semaphore(concurrency)
        total_inserted = 0

        async def _fetch_one(symbol: str) -> int:
            async with semaphore:
                try:
                    candles = await self.client.get_kline_history(
                        symbol=symbol, interval=interval, limit=lookback
                    )
                    if candles:
                        candles = validate_candles(candles)
                    if candles:
                        rows = self._candles_to_rows(candles)
                        inserted = await self.storage.insert_candles(rows, interval=interval)
                        if inserted:
                            logger.debug("%s: +%d candles", symbol, inserted)
                        return inserted
                except Exception as exc:
                    logger.warning("Failed to fetch candles for %s: %s", symbol, exc)
                return 0

        results = await asyncio.gather(
            *[_fetch_one(s) for s in symbols], return_exceptions=True
        )

        for sym, res in zip(symbols, results):
            if isinstance(res, Exception):
                logger.error("Fetch failed for %s: %s", sym, res)
            else:
                total_inserted += res

        logger.info(
            "Incremental fetch complete: %d new candles across %d symbols",
            total_inserted,
            len(symbols),
        )
        return total_inserted

    # ------------------------------------------------------------------
    # Backfill (one-time historical download)
    # ------------------------------------------------------------------
    async def backfill_symbol(
        self,
        symbol: str,
        interval: str = "60",
        total_candles: int = 2000,
        batch_size: int = 500,
    ) -> int:
        """
        Download up to *total_candles* historical candles for a single symbol,
        paging backwards from the most recent data.

        The Spot kline/history endpoint returns candles in chronological order
        ending just before *endTime* (exclusive upper bound, in seconds).
        """
        inserted_total = 0
        end_time: Optional[int] = None
        remaining = total_candles
        max_iterations = (total_candles // max(batch_size, 1)) + 10
        iteration = 0
        prev_end_time: Optional[int] = None

        while remaining > 0:
            iteration += 1
            if iteration > max_iterations:
                logger.warning(
                    "%s: backfill exceeded %d iterations — stopping to prevent infinite loop",
                    symbol, max_iterations,
                )
                break

            fetch_count = min(remaining, batch_size)
            try:
                candles = await self.client.get_kline_history(
                    symbol=symbol,
                    interval=interval,
                    limit=fetch_count,
                    end_time=end_time,
                )
            except Exception as exc:
                logger.error("Backfill request failed for %s: %s", symbol, exc)
                break

            if not candles:
                logger.debug("%s: no more candles returned", symbol)
                break

            candles = validate_candles(candles)
            if not candles:
                logger.debug("%s: all candles rejected by validation", symbol)
                break

            rows = self._candles_to_rows(candles)
            inserted = await self.storage.insert_candles(rows, interval=interval)
            inserted_total += inserted

            earliest_ts = candles[0].ts
            try:
                from datetime import datetime, timezone

                dt = datetime.fromisoformat(earliest_ts.replace("Z", "+00:00"))
                end_time = int(dt.timestamp())
            except Exception:
                logger.warning(
                    "%s: could not parse earliest ts '%s', stopping backfill",
                    symbol,
                    earliest_ts,
                )
                break

            if end_time == prev_end_time:
                logger.warning(
                    "%s: backfill stuck at same timestamp — breaking", symbol
                )
                break
            prev_end_time = end_time

            remaining -= len(candles)
            if len(candles) < fetch_count:
                break

        logger.info("%s: backfilled %d candles", symbol, inserted_total)
        return inserted_total

    async def backfill_all(
        self,
        symbols: List[str],
        interval: str = "60",
        total_candles: int = 2000,
        concurrency: int = 5,
    ) -> int:
        """Backfill multiple symbols with bounded concurrency."""
        semaphore = asyncio.Semaphore(concurrency)
        total = 0

        async def _backfill_one(sym: str) -> int:
            async with semaphore:
                return await self.backfill_symbol(
                    sym, interval=interval, total_candles=total_candles
                )

        results = await asyncio.gather(
            *[_backfill_one(s) for s in symbols], return_exceptions=True
        )
        for sym, res in zip(symbols, results):
            if isinstance(res, Exception):
                logger.error("Backfill failed for %s: %s", sym, res)
            else:
                total += res

        logger.info(
            "Backfill complete: %d total candles for %d symbols", total, len(symbols)
        )
        return total

    # ------------------------------------------------------------------
    # Order book snapshots
    # ------------------------------------------------------------------
    async def fetch_order_book_snapshots(
        self,
        symbols: List[str],
        concurrency: int = 8,
        max_levels: int = 15,
    ) -> List[tuple]:
        """
        Fetch order book depth for all symbols and return rows ready for
        ``storage.insert_order_book_snapshots()``.

        Each row: (symbol, ts, bid_prices_json, bid_vols_json,
                   ask_prices_json, ask_vols_json, spread, mid_price, imbalance)
        """
        import json
        from datetime import datetime, timezone

        semaphore = asyncio.Semaphore(concurrency)
        ts_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        rows: List[tuple] = []
        errors = 0

        async def _fetch_one(symbol: str) -> Optional[tuple]:
            async with semaphore:
                try:
                    depth = await self.client.get_market_depth(symbol)
                    if not depth:
                        return None

                    raw_bids = depth.get("bids") or depth.get("b") or []
                    raw_asks = depth.get("asks") or depth.get("a") or []

                    bids = raw_bids[:max_levels]
                    asks = raw_asks[:max_levels]

                    bid_prices = []
                    bid_vols = []
                    for entry in bids:
                        if isinstance(entry, dict):
                            bid_prices.append(float(entry.get("price", 0)))
                            bid_vols.append(float(entry.get("volume", 0)))
                        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                            bid_prices.append(float(entry[0]))
                            bid_vols.append(float(entry[1]))

                    ask_prices = []
                    ask_vols = []
                    for entry in asks:
                        if isinstance(entry, dict):
                            ask_prices.append(float(entry.get("price", 0)))
                            ask_vols.append(float(entry.get("volume", 0)))
                        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                            ask_prices.append(float(entry[0]))
                            ask_vols.append(float(entry[1]))

                    if not bid_prices or not ask_prices:
                        return None

                    best_bid = bid_prices[0]
                    best_ask = ask_prices[0]
                    spread = best_ask - best_bid
                    mid_price = (best_ask + best_bid) / 2.0

                    total_bid_vol = sum(bid_vols)
                    total_ask_vol = sum(ask_vols)
                    total_vol = total_bid_vol + total_ask_vol
                    imbalance = total_bid_vol / total_vol if total_vol > 0 else 0.5

                    snapshot_ts = depth.get("ts", ts_now)
                    if isinstance(snapshot_ts, (int, float)):
                        snapshot_ts = datetime.fromtimestamp(
                            snapshot_ts / 1000, tz=timezone.utc
                        ).strftime("%Y-%m-%dT%H:%M:%S")

                    return (
                        symbol,
                        snapshot_ts,
                        json.dumps(bid_prices),
                        json.dumps(bid_vols),
                        json.dumps(ask_prices),
                        json.dumps(ask_vols),
                        spread,
                        mid_price,
                        imbalance,
                    )
                except Exception as exc:
                    logger.debug("Order book fetch failed for %s: %s", symbol, exc)
                    return None

        results = await asyncio.gather(
            *[_fetch_one(s) for s in symbols], return_exceptions=True
        )

        for sym, res in zip(symbols, results):
            if isinstance(res, Exception):
                errors += 1
                logger.debug("Order book error for %s: %s", sym, res)
            elif res is not None:
                rows.append(res)

        logger.info(
            "Order book snapshots: %d collected, %d failed out of %d symbols",
            len(rows), errors, len(symbols),
        )
        return rows

    # ------------------------------------------------------------------
    # Funding rate snapshots
    # ------------------------------------------------------------------
    async def fetch_funding_rate_snapshots(
        self,
        symbols: List[str],
        concurrency: int = 8,
    ) -> List[tuple]:
        """
        Fetch current funding rate for all futures symbols and return rows
        ready for ``storage.insert_funding_rate_snapshots()``.

        Each row: (symbol, ts, funding_rate, mark_price, last_price,
                   next_funding_ts, funding_interval_hours)
        """
        from datetime import datetime, timezone

        semaphore = asyncio.Semaphore(concurrency)
        ts_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        rows: List[tuple] = []
        errors = 0

        async def _fetch_one(symbol: str) -> Optional[tuple]:
            async with semaphore:
                try:
                    data = await self.client.get_funding_rate(symbol)
                    if not data:
                        return None

                    funding_rate = float(data.get("fundingRate", 0))
                    mark_price = float(data.get("markPrice", 0))
                    last_price = float(data.get("lastPrice", 0))
                    next_funding_time = data.get("nextFundingTime")
                    funding_interval = int(data.get("fundingInterval", 8))

                    next_ts = None
                    if next_funding_time:
                        try:
                            nft = int(next_funding_time)
                            next_ts = datetime.fromtimestamp(
                                nft / 1000, tz=timezone.utc
                            ).strftime("%Y-%m-%dT%H:%M:%S")
                        except (ValueError, TypeError):
                            next_ts = str(next_funding_time)

                    return (
                        symbol,
                        ts_now,
                        funding_rate,
                        mark_price,
                        last_price,
                        next_ts,
                        funding_interval,
                    )
                except Exception as exc:
                    logger.debug("Funding rate fetch failed for %s: %s", symbol, exc)
                    return None

        results = await asyncio.gather(
            *[_fetch_one(s) for s in symbols], return_exceptions=True
        )

        for sym, res in zip(symbols, results):
            if isinstance(res, Exception):
                errors += 1
                logger.debug("Funding rate error for %s: %s", sym, res)
            elif res is not None:
                rows.append(res)

        logger.info(
            "Funding rate snapshots: %d collected, %d failed out of %d symbols",
            len(rows), errors, len(symbols),
        )
        return rows

    # ------------------------------------------------------------------
    # Coinalyze data parsing
    # ------------------------------------------------------------------
    @staticmethod
    def parse_coinalyze_results(
        raw: dict,
        reverse_map: dict[str, str],
    ) -> tuple[list[tuple], list[tuple], list[tuple]]:
        """
        Parse raw Coinalyze API results into storage-ready rows.

        Parameters
        ----------
        raw : dict with keys "oi", "liquidations", "long_short"
        reverse_map : mapping from Coinalyze symbol -> our symbol
                      (e.g. "BTCUSDT_PERP.A" -> "BTCUSDT")

        Returns
        -------
        (oi_rows, liq_rows, ls_rows) ready for bulk insert.
        """
        from datetime import datetime, timezone

        def _ts_to_iso(unix_sec: int) -> str:
            return datetime.fromtimestamp(
                unix_sec, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S")

        oi_rows: list[tuple] = []
        for entry in raw.get("oi", []):
            ca_sym = entry.get("symbol", "")
            our_sym = reverse_map.get(ca_sym)
            if not our_sym:
                continue
            for point in entry.get("history", []):
                oi_rows.append((
                    our_sym,
                    _ts_to_iso(point["t"]),
                    float(point.get("o", 0)),
                    float(point.get("h", 0)),
                    float(point.get("l", 0)),
                    float(point.get("c", 0)),
                ))

        liq_rows: list[tuple] = []
        for entry in raw.get("liquidations", []):
            ca_sym = entry.get("symbol", "")
            our_sym = reverse_map.get(ca_sym)
            if not our_sym:
                continue
            for point in entry.get("history", []):
                liq_rows.append((
                    our_sym,
                    _ts_to_iso(point["t"]),
                    float(point.get("l", 0)),
                    float(point.get("s", 0)),
                ))

        ls_rows: list[tuple] = []
        for entry in raw.get("long_short", []):
            ca_sym = entry.get("symbol", "")
            our_sym = reverse_map.get(ca_sym)
            if not our_sym:
                continue
            for point in entry.get("history", []):
                ls_rows.append((
                    our_sym,
                    _ts_to_iso(point["t"]),
                    float(point.get("r", 0)),
                    float(point.get("l", 0)),
                    float(point.get("s", 0)),
                ))

        logger.info(
            "Coinalyze parsed: %d OI rows, %d liquidation rows, %d L/S rows",
            len(oi_rows), len(liq_rows), len(ls_rows),
        )
        return oi_rows, liq_rows, ls_rows

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _candles_to_rows(candles: List[Candle]) -> List[tuple]:
        return [
            (c.symbol, c.ts, c.open, c.high, c.low, c.close, c.volume)
            for c in candles
        ]
