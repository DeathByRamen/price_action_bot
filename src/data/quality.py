"""
Data quality validation for candle data, gap detection, and training data health.

Used by the collector (before insertion), the training pipeline (before training),
and the daily retrain digest (quality reporting).
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.api.bitunix_client import Candle

logger = logging.getLogger(__name__)

# Interval in minutes, used for gap detection
_INTERVAL_MINUTES = {"1": 1, "5": 5, "15": 15, "30": 30, "60": 60,
                     "120": 120, "240": 240, "360": 360, "720": 720}


def validate_candles(candles: List[Candle]) -> List[Candle]:
    """
    Filter out invalid candles and log warnings.

    Rejects candles where:
    - close <= 0
    - high < low
    - volume < 0

    Warns about suspicious candles (price jump > 50% between consecutive candles).
    """
    if not candles:
        return candles

    valid: List[Candle] = []
    rejected = 0

    for c in candles:
        if c.close <= 0:
            logger.warning(
                "Rejected %s candle at %s: close=%.8f <= 0", c.symbol, c.ts, c.close
            )
            rejected += 1
            continue

        if c.high < c.low:
            logger.warning(
                "Rejected %s candle at %s: high=%.8f < low=%.8f",
                c.symbol, c.ts, c.high, c.low,
            )
            rejected += 1
            continue

        if c.volume < 0:
            logger.warning(
                "Rejected %s candle at %s: volume=%.4f < 0", c.symbol, c.ts, c.volume
            )
            rejected += 1
            continue

        valid.append(c)

    if len(valid) >= 2:
        for i in range(1, len(valid)):
            prev_close = valid[i - 1].close
            curr_close = valid[i].close
            if prev_close > 0:
                pct_change = abs(curr_close - prev_close) / prev_close
                if pct_change > 0.5:
                    logger.warning(
                        "Suspicious %s candle at %s: %.1f%% price jump (%.8f -> %.8f)",
                        valid[i].symbol, valid[i].ts,
                        pct_change * 100, prev_close, curr_close,
                    )

    if rejected:
        logger.info(
            "Candle validation: %d rejected, %d valid out of %d",
            rejected, len(valid), len(candles),
        )

    return valid


def detect_gaps(
    df: pd.DataFrame,
    interval: str = "60",
    max_allowed_gap_multiplier: float = 1.5,
) -> List[Tuple[str, str, int]]:
    """
    Detect timestamp gaps in a candle DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a 'ts' column with ISO-8601 timestamps, sorted ascending.
    interval : str
        Candle interval in minutes (e.g. "60", "15").
    max_allowed_gap_multiplier : float
        Gaps exceeding interval * this multiplier are flagged.

    Returns
    -------
    List of (gap_start_ts, gap_end_ts, gap_minutes) tuples.
    """
    if len(df) < 2 or "ts" not in df.columns:
        return []

    interval_min = _INTERVAL_MINUTES.get(interval, 60)
    max_gap = timedelta(minutes=interval_min * max_allowed_gap_multiplier)

    timestamps = pd.to_datetime(df["ts"], utc=True)
    gaps: List[Tuple[str, str, int]] = []

    for i in range(1, len(timestamps)):
        delta = timestamps.iloc[i] - timestamps.iloc[i - 1]
        if delta > max_gap:
            gap_min = int(delta.total_seconds() / 60)
            gaps.append((
                str(timestamps.iloc[i - 1]),
                str(timestamps.iloc[i]),
                gap_min,
            ))

    return gaps


def check_training_data_quality(
    symbol_data: dict[str, pd.DataFrame],
    interval: str = "60",
    min_candles: int = 100,
    max_zero_volume_pct: float = 0.10,
    max_price_jump_pct: float = 0.50,
) -> Tuple[dict[str, pd.DataFrame], List[str]]:
    """
    Validate training data quality per symbol. Returns cleaned data and warnings.

    Parameters
    ----------
    symbol_data : dict[str, pd.DataFrame]
        Raw symbol DataFrames with OHLCV columns.
    min_candles : int
        Minimum candles required after cleaning.
    max_zero_volume_pct : float
        Exclude symbols with more than this fraction of zero-volume candles.
    max_price_jump_pct : float
        Flag single-candle price jumps exceeding this percentage.

    Returns
    -------
    (cleaned_data, warnings) where cleaned_data has bad symbols removed.
    """
    cleaned: dict[str, pd.DataFrame] = {}
    warnings: List[str] = []
    excluded = 0

    for symbol, df in symbol_data.items():
        if len(df) < min_candles:
            warnings.append(f"{symbol}: only {len(df)} candles (need {min_candles})")
            excluded += 1
            continue

        zero_vol_pct = (df["volume"] == 0).mean() if "volume" in df.columns else 0
        if zero_vol_pct > max_zero_volume_pct:
            warnings.append(
                f"{symbol}: {zero_vol_pct:.0%} zero-volume candles — excluded"
            )
            excluded += 1
            continue

        if "close" in df.columns and len(df) > 1:
            pct_changes = df["close"].pct_change().abs()
            big_jumps = pct_changes[pct_changes > max_price_jump_pct]
            if not big_jumps.empty:
                warnings.append(
                    f"{symbol}: {len(big_jumps)} candles with >{max_price_jump_pct:.0%} "
                    f"price jump (max {big_jumps.max():.1%})"
                )

        gaps = detect_gaps(df, interval=interval) if "ts" in df.columns else []
        if gaps:
            total_gap_min = sum(g[2] for g in gaps)
            warnings.append(
                f"{symbol}: {len(gaps)} timestamp gaps totaling {total_gap_min} min"
            )

        cleaned[symbol] = df

    if excluded:
        logger.info(
            "Training data quality: %d symbols excluded, %d retained",
            excluded, len(cleaned),
        )

    if warnings:
        for w in warnings[:20]:
            logger.warning("Data quality: %s", w)
        if len(warnings) > 20:
            logger.warning("... and %d more warnings", len(warnings) - 20)

    return cleaned, warnings


def has_timestamp_gap_in_window(
    timestamps: np.ndarray,
    interval_minutes: int = 60,
    max_gap_multiplier: float = 1.5,
) -> bool:
    """
    Check if a sequence of timestamps contains a gap.

    Parameters
    ----------
    timestamps : np.ndarray
        Array of datetime64 or float timestamps (Unix seconds).
    interval_minutes : int
        Expected interval between consecutive timestamps.
    max_gap_multiplier : float
        Gaps exceeding interval * this multiplier are flagged.

    Returns True if any gap exceeds the allowed threshold.
    """
    if len(timestamps) < 2:
        return False

    max_gap_sec = interval_minutes * 60 * max_gap_multiplier

    if np.issubdtype(timestamps.dtype, np.datetime64):
        diffs = np.diff(timestamps).astype("timedelta64[s]").astype(float)
    else:
        diffs = np.diff(timestamps)

    return bool(np.any(diffs > max_gap_sec))
