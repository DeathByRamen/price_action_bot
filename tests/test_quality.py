"""Tests for src/data/quality.py — candle validation and gap detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.quality import (
    check_training_data_quality,
    detect_gaps,
    has_timestamp_gap_in_window,
    validate_candles,
)


@dataclass
class FakeCandle:
    symbol: str
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class TestValidateCandles:
    def test_valid_candles_pass_through(self):
        candles = [
            FakeCandle("BTC", "2025-01-01T00:00:00+00:00", 100, 105, 95, 102, 1000),
            FakeCandle("BTC", "2025-01-01T01:00:00+00:00", 102, 108, 100, 106, 1200),
        ]
        result = validate_candles(candles)
        assert len(result) == 2

    def test_rejects_negative_close(self):
        candles = [
            FakeCandle("BTC", "2025-01-01T00:00:00+00:00", 100, 105, 95, -5, 1000),
        ]
        result = validate_candles(candles)
        assert len(result) == 0

    def test_rejects_zero_close(self):
        candles = [
            FakeCandle("BTC", "2025-01-01T00:00:00+00:00", 100, 105, 95, 0, 1000),
        ]
        result = validate_candles(candles)
        assert len(result) == 0

    def test_rejects_high_less_than_low(self):
        candles = [
            FakeCandle("BTC", "2025-01-01T00:00:00+00:00", 100, 90, 95, 92, 1000),
        ]
        result = validate_candles(candles)
        assert len(result) == 0

    def test_rejects_negative_volume(self):
        candles = [
            FakeCandle("BTC", "2025-01-01T00:00:00+00:00", 100, 105, 95, 102, -500),
        ]
        result = validate_candles(candles)
        assert len(result) == 0

    def test_mixed_valid_and_invalid(self, bad_candle_data):
        candles = [
            FakeCandle(d["symbol"], d["ts"], d["open"], d["high"],
                       d["low"], d["close"], d["volume"])
            for d in bad_candle_data
        ]
        result = validate_candles(candles)
        # Row 0: valid, Row 1: high<low, Row 2: close<0, Row 3: vol<0, Row 4: valid
        assert len(result) == 2

    def test_warns_on_large_price_jump(self, caplog):
        candles = [
            FakeCandle("BTC", "2025-01-01T00:00:00+00:00", 100, 105, 95, 100, 1000),
            FakeCandle("BTC", "2025-01-01T01:00:00+00:00", 200, 210, 190, 200, 1000),
        ]
        with caplog.at_level("WARNING"):
            result = validate_candles(candles)
        assert len(result) == 2
        assert "Suspicious" in caplog.text
        assert "100.0%" in caplog.text

    def test_empty_input(self):
        assert validate_candles([]) == []


class TestDetectGaps:
    def test_no_gaps(self):
        ts = pd.date_range("2025-01-01", periods=10, freq="1h", tz="UTC")
        df = pd.DataFrame({"ts": ts.strftime("%Y-%m-%dT%H:%M:%S+00:00")})
        gaps = detect_gaps(df, interval="60")
        assert len(gaps) == 0

    def test_detects_single_gap(self):
        ts = list(pd.date_range("2025-01-01", periods=5, freq="1h", tz="UTC"))
        ts[3] = ts[2] + pd.Timedelta(hours=4)
        ts[4] = ts[3] + pd.Timedelta(hours=1)
        df = pd.DataFrame({"ts": [t.strftime("%Y-%m-%dT%H:%M:%S+00:00") for t in ts]})
        gaps = detect_gaps(df, interval="60")
        assert len(gaps) == 1
        assert gaps[0][2] == 240  # 4 hours in minutes

    def test_15m_interval_gaps(self):
        ts = list(pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC"))
        ts[2] = ts[1] + pd.Timedelta(hours=1)
        ts[3] = ts[2] + pd.Timedelta(minutes=15)
        ts[4] = ts[3] + pd.Timedelta(minutes=15)
        df = pd.DataFrame({"ts": [t.strftime("%Y-%m-%dT%H:%M:%S+00:00") for t in ts]})
        gaps = detect_gaps(df, interval="15")
        assert len(gaps) == 1

    def test_empty_dataframe(self):
        df = pd.DataFrame({"ts": []})
        assert detect_gaps(df) == []


class TestHasTimestampGapInWindow:
    def test_no_gap(self):
        ts = np.arange(0, 600, 60, dtype=float)  # 10 minutes, 1-min intervals
        assert has_timestamp_gap_in_window(ts, interval_minutes=1) is False

    def test_has_gap(self):
        ts = np.array([0, 60, 120, 600, 660], dtype=float)  # gap from 120->600
        assert has_timestamp_gap_in_window(ts, interval_minutes=1) is True

    def test_single_element(self):
        assert has_timestamp_gap_in_window(np.array([100.0]), interval_minutes=60) is False


class TestCheckTrainingDataQuality:
    def test_filters_insufficient_data(self, synthetic_ohlcv):
        short_df = synthetic_ohlcv.iloc[:10].copy()
        data = {"SHORT": short_df}
        cleaned, warnings = check_training_data_quality(data, min_candles=100)
        assert len(cleaned) == 0
        assert any("only 10" in w for w in warnings)

    def test_keeps_good_data(self, synthetic_ohlcv):
        data = {"GOOD": synthetic_ohlcv}
        cleaned, warnings = check_training_data_quality(data, min_candles=100)
        assert "GOOD" in cleaned

    def test_flags_zero_volume(self, synthetic_ohlcv):
        df = synthetic_ohlcv.copy()
        df["volume"] = 0.0
        data = {"ZEROVAL": df}
        cleaned, warnings = check_training_data_quality(
            data, min_candles=100, max_zero_volume_pct=0.05
        )
        assert len(cleaned) == 0
        assert any("zero-volume" in w for w in warnings)
