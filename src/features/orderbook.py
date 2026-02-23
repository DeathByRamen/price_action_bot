"""
Order book feature engineering.

Computes structural and liquidity features from stored order book snapshots.
These features are merged with OHLCV data by timestamp.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _parse_json_sum(val) -> float:
    """Sum values from a JSON-encoded list string, or return 0."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    try:
        parsed = json.loads(val)
        if isinstance(parsed, list):
            return sum(float(v) for v in parsed if v is not None)
        return float(parsed)
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def compute_orderbook_features(
    ob_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute order book features and merge with OHLCV data.

    Handles both raw DB format (bid_vols/ask_vols as JSON strings) and
    pre-processed format (bid_volume/ask_volume as floats).

    Parameters
    ----------
    ob_df : pd.DataFrame
        Order book snapshots from storage. Expected columns include ts,
        and either bid_volume/ask_volume (floats) or bid_vols/ask_vols
        (JSON strings), plus spread and mid_price.
    ohlcv_df : pd.DataFrame
        OHLCV data with 'ts' column.

    Returns
    -------
    OHLCV DataFrame augmented with order book features.
    """
    if ob_df.empty:
        return _add_empty_ob_columns(ohlcv_df)

    ob = ob_df.copy()

    if "bid_volume" in ob.columns:
        bid_vol = pd.to_numeric(ob["bid_volume"], errors="coerce").fillna(0)
        ask_vol = pd.to_numeric(ob["ask_volume"], errors="coerce").fillna(0)
    elif "bid_vols" in ob.columns:
        bid_vol = ob["bid_vols"].apply(_parse_json_sum)
        ask_vol = ob["ask_vols"].apply(_parse_json_sum)
    else:
        return _add_empty_ob_columns(ohlcv_df)

    total_vol = bid_vol + ask_vol

    if "imbalance" in ob.columns:
        ob["ob_imbalance"] = pd.to_numeric(ob["imbalance"], errors="coerce")
    else:
        ob["ob_imbalance"] = bid_vol / total_vol.replace(0, np.nan)

    mid = pd.to_numeric(ob.get("mid_price", 0), errors="coerce")
    spread = pd.to_numeric(ob.get("spread", 0), errors="coerce")
    ob["ob_spread_bps"] = (spread / mid.replace(0, np.nan)) * 10_000

    ob["ob_depth_ratio"] = bid_vol / ask_vol.replace(0, np.nan)
    ob["ob_total_depth"] = total_vol

    ob["ob_imbalance_change"] = ob["ob_imbalance"].diff()
    ob["ob_spread_change"] = ob["ob_spread_bps"].diff()

    feature_cols = [
        "ts", "ob_imbalance", "ob_spread_bps", "ob_depth_ratio",
        "ob_total_depth", "ob_imbalance_change", "ob_spread_change",
    ]
    ob_features = ob[feature_cols].copy()

    merged = ohlcv_df.merge(ob_features, on="ts", how="left")

    for col in get_orderbook_feature_columns():
        if col in merged.columns:
            merged[col] = merged[col].ffill()

    return merged


def get_orderbook_feature_columns() -> list[str]:
    """Return ordered list of order book feature column names."""
    return [
        "ob_imbalance",
        "ob_spread_bps",
        "ob_depth_ratio",
        "ob_total_depth",
        "ob_imbalance_change",
        "ob_spread_change",
    ]


def _add_empty_ob_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add empty order book columns when no data is available."""
    for col in get_orderbook_feature_columns():
        df[col] = np.nan
    return df
