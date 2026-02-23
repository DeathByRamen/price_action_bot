"""Shared fixtures for PA Bot tests."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Generate 500 rows of realistic synthetic OHLCV data."""
    np.random.seed(42)
    n = 500
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, n)
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.abs(np.random.normal(1_000_000, 300_000, n))

    ts_start = pd.Timestamp("2025-01-01", tz="UTC")
    timestamps = pd.date_range(ts_start, periods=n, freq="1h")

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "ts": timestamps.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "symbol": "TESTUSDT",
    })


@pytest.fixture
def multi_symbol_ohlcv(synthetic_ohlcv: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Generate OHLCV for 3 symbols."""
    np.random.seed(123)
    result = {"BTCUSDT": synthetic_ohlcv.copy()}

    for sym, multiplier in [("ETHUSDT", 0.05), ("SOLUSDT", 0.001)]:
        df = synthetic_ohlcv.copy()
        df["open"] *= multiplier
        df["high"] *= multiplier
        df["low"] *= multiplier
        df["close"] *= multiplier
        noise = np.random.normal(1.0, 0.01, len(df))
        df["close"] *= noise
        df["high"] *= np.abs(noise)
        df["low"] *= np.abs(noise)
        df["open"] *= noise
        df["volume"] *= np.random.uniform(0.5, 2.0, len(df))
        df["symbol"] = sym
        result[sym] = df

    return result


@pytest.fixture
def bad_candle_data() -> list[dict]:
    """Known-bad candle data for validation tests."""
    return [
        {"symbol": "TEST", "ts": "2025-01-01T00:00:00+00:00",
         "open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000},
        {"symbol": "TEST", "ts": "2025-01-01T01:00:00+00:00",
         "open": 102, "high": 100, "low": 103, "close": 101, "volume": 500},
        {"symbol": "TEST", "ts": "2025-01-01T02:00:00+00:00",
         "open": 101, "high": 106, "low": 99, "close": -5, "volume": 800},
        {"symbol": "TEST", "ts": "2025-01-01T03:00:00+00:00",
         "open": 103, "high": 107, "low": 100, "close": 105, "volume": -200},
        {"symbol": "TEST", "ts": "2025-01-01T04:00:00+00:00",
         "open": 105, "high": 110, "low": 102, "close": 108, "volume": 1200},
    ]
