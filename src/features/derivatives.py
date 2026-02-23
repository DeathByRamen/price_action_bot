"""
Derivatives feature engineering.

Computes features from Coinalyze data (OI, liquidations, long/short ratio)
and BitUnix funding rate snapshots.  All features are designed to be
merged with OHLCV data by timestamp.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_coinalyze_features(
    oi_df: pd.DataFrame,
    liq_df: pd.DataFrame,
    ls_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    rolling_window: int = 24,
) -> pd.DataFrame:
    """
    Compute Coinalyze-derived features and merge with OHLCV.

    Parameters
    ----------
    oi_df : pd.DataFrame
        Open interest data with columns: ts, symbol, open, high, low, close.
    liq_df : pd.DataFrame
        Liquidation data with columns: ts, symbol, long_liq, short_liq.
    ls_df : pd.DataFrame
        Long/short ratio data with columns: ts, symbol, ratio.
    ohlcv_df : pd.DataFrame
        OHLCV data with 'ts' column.
    rolling_window : int
        Window size for rolling calculations.

    Returns
    -------
    OHLCV DataFrame augmented with Coinalyze features.
    """
    result = ohlcv_df.copy()

    if not oi_df.empty:
        oi = oi_df.copy()
        oi_col = "oi_close" if "oi_close" in oi.columns else "close"
        oi["oi_close"] = pd.to_numeric(oi[oi_col], errors="coerce")
        oi["oi_change_pct"] = oi["oi_close"].pct_change()
        oi_mean = oi["oi_close"].rolling(rolling_window * 7).mean()
        oi_std = oi["oi_close"].rolling(rolling_window * 7).std().replace(0, np.nan)
        oi["oi_zscore"] = (oi["oi_close"] - oi_mean) / oi_std

        oi_features = oi[["ts", "oi_change_pct", "oi_zscore"]].copy()
        result = result.merge(oi_features, on="ts", how="left")

    if not liq_df.empty:
        liq = liq_df.copy()
        long_col = "long_liq" if "long_liq" in liq.columns else "long_vol"
        short_col = "short_liq" if "short_liq" in liq.columns else "short_vol"
        long_liq = pd.to_numeric(liq.get(long_col, 0), errors="coerce").fillna(0)
        short_liq = pd.to_numeric(liq.get(short_col, 0), errors="coerce").fillna(0)
        total_liq = long_liq + short_liq + 1e-10

        liq["liq_imbalance"] = (long_liq - short_liq) / total_liq

        liq_avg = total_liq.rolling(rolling_window).mean().replace(0, np.nan)
        liq["liq_spike"] = total_liq / liq_avg

        liq_features = liq[["ts", "liq_imbalance", "liq_spike"]].copy()
        result = result.merge(liq_features, on="ts", how="left")

    if not ls_df.empty:
        ls = ls_df.copy()
        ls["ls_ratio"] = pd.to_numeric(ls.get("ratio", 1.0), errors="coerce")
        ls_mean = ls["ls_ratio"].rolling(rolling_window).mean()
        ls_std = ls["ls_ratio"].rolling(rolling_window).std().replace(0, np.nan)
        ls["ls_ratio_extreme"] = (ls["ls_ratio"] - ls_mean) / ls_std

        ls_features = ls[["ts", "ls_ratio", "ls_ratio_extreme"]].copy()
        result = result.merge(ls_features, on="ts", how="left")

    for col in get_coinalyze_feature_columns():
        if col in result.columns:
            result[col] = result[col].fillna(0.0)

    return result


def compute_funding_rate_features(
    fr_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    rolling_window: int = 24 * 7,
) -> pd.DataFrame:
    """
    Compute funding rate features and merge with OHLCV.

    Parameters
    ----------
    fr_df : pd.DataFrame
        Funding rate snapshots: ts, symbol, funding_rate, mark_price, last_price.
    ohlcv_df : pd.DataFrame
        OHLCV data with 'ts' column.
    rolling_window : int
        Window for Z-score calculation.

    Returns
    -------
    OHLCV DataFrame augmented with funding rate features.
    """
    result = ohlcv_df.copy()

    if fr_df.empty:
        for col in get_funding_rate_feature_columns():
            result[col] = np.nan
        return result

    fr = fr_df.copy()
    fr["fr_rate"] = pd.to_numeric(fr.get("funding_rate", 0), errors="coerce")

    fr_mean = fr["fr_rate"].rolling(rolling_window).mean()
    fr_std = fr["fr_rate"].rolling(rolling_window).std().replace(0, np.nan)
    fr["fr_rate_zscore"] = (fr["fr_rate"] - fr_mean) / fr_std

    fr["fr_rate_momentum"] = fr["fr_rate"].diff()

    mark = pd.to_numeric(fr.get("mark_price", 0), errors="coerce")
    last = pd.to_numeric(fr.get("last_price", 0), errors="coerce")
    fr["fr_mark_spot_divergence"] = (mark - last) / last.replace(0, np.nan)

    fr_features = fr[["ts", "fr_rate", "fr_rate_zscore",
                       "fr_rate_momentum", "fr_mark_spot_divergence"]].copy()
    result = result.merge(fr_features, on="ts", how="left")

    for col in get_funding_rate_feature_columns():
        if col in result.columns:
            result[col] = result[col].fillna(0.0)

    return result


def compute_cross_asset_features(
    btc_ohlcv: pd.DataFrame,
    symbol_ohlcv: pd.DataFrame,
    all_volumes: Optional[pd.DataFrame] = None,
    rolling_window: int = 24,
) -> pd.DataFrame:
    """
    Compute cross-asset features (BTC correlation, dominance proxy).

    Parameters
    ----------
    btc_ohlcv : pd.DataFrame
        BTC OHLCV data with 'ts' and 'close' columns.
    symbol_ohlcv : pd.DataFrame
        Target symbol's OHLCV data.
    all_volumes : pd.DataFrame | None
        Total market volume DataFrame with 'ts' and 'total_volume'.
    rolling_window : int
        Window for rolling correlation.

    Returns
    -------
    Symbol OHLCV augmented with cross-asset features.
    """
    result = symbol_ohlcv.copy()

    if btc_ohlcv.empty:
        for col in get_cross_asset_feature_columns():
            result[col] = np.nan
        return result

    btc = btc_ohlcv[["ts", "close", "volume"]].copy()
    btc = btc.rename(columns={"close": "btc_close", "volume": "btc_volume"})
    result = result.merge(btc, on="ts", how="left")

    result["btc_return_1h"] = result["btc_close"].pct_change()

    sym_returns = result["close"].pct_change()
    btc_returns = result["btc_return_1h"]
    result["correlation_to_btc"] = sym_returns.rolling(rolling_window).corr(btc_returns)

    if "btc_volume" in result.columns:
        total_vol = result.get("total_market_volume", result["btc_volume"] * 3)
        result["btc_dominance_proxy"] = result["btc_volume"] / total_vol.replace(0, np.nan)

    result = result.drop(columns=["btc_close", "btc_volume"], errors="ignore")

    return result


def get_coinalyze_feature_columns() -> list[str]:
    """Return ordered list of Coinalyze feature column names."""
    return [
        "oi_change_pct",
        "oi_zscore",
        "liq_imbalance",
        "liq_spike",
        "ls_ratio",
        "ls_ratio_extreme",
    ]


def get_funding_rate_feature_columns() -> list[str]:
    """Return ordered list of funding rate feature column names."""
    return [
        "fr_rate",
        "fr_rate_zscore",
        "fr_rate_momentum",
        "fr_mark_spot_divergence",
    ]


def get_cross_asset_feature_columns() -> list[str]:
    """Return ordered list of cross-asset feature column names."""
    return [
        "btc_return_1h",
        "correlation_to_btc",
        "btc_dominance_proxy",
    ]
