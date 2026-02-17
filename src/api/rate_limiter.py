"""
Token-bucket rate limiter for async HTTP calls.

BitUnix enforces 10 req/sec/ip on public endpoints.
We default to 9 req/sec to leave headroom.
"""

import asyncio
import time


class RateLimiter:
    """Async token-bucket rate limiter."""

    def __init__(self, max_rate: float = 9.0, burst: int = 10):
        """
        Parameters
        ----------
        max_rate : float
            Sustained requests per second.
        burst : int
            Maximum burst size (bucket capacity).
        """
        self._max_rate = max_rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume one."""
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._max_rate
                await asyncio.sleep(wait)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._max_rate)
        self._last_refill = now
