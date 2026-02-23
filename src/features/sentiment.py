"""
Sentiment and cross-exchange feature engineering.

Computes features from Fear & Greed Index, news sentiment,
and Binance cross-exchange data (funding rate spread, OI divergence).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_sentiment_features(
    fear_greed_df: pd.DataFrame,
    news_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute sentiment features and merge with OHLCV data.

    Parameters
    ----------
    fear_greed_df : pd.DataFrame
        Fear & Greed Index data with columns: ts, value.
    news_df : pd.DataFrame
        News sentiment data with columns: ts, symbol, positive, negative, neutral, total.
    ohlcv_df : pd.DataFrame
        OHLCV data with 'ts' column.

    Returns
    -------
    OHLCV DataFrame augmented with sentiment features.
    """
    result = ohlcv_df.copy()

    if not fear_greed_df.empty:
        fg = fear_greed_df.copy()
        fg["fear_greed_index"] = pd.to_numeric(fg["value"], errors="coerce") / 100.0
        fg["fear_greed_change"] = fg["fear_greed_index"].diff()
        fg_features = fg[["ts", "fear_greed_index", "fear_greed_change"]].copy()
        result = result.merge(fg_features, on="ts", how="left")

    if not news_df.empty:
        news = news_df.copy()
        pos = pd.to_numeric(news.get("positive", 0), errors="coerce").fillna(0)
        neg = pd.to_numeric(news.get("negative", 0), errors="coerce").fillna(0)
        total = pd.to_numeric(news.get("total", 0), errors="coerce").fillna(0)

        news["news_sentiment_score"] = (pos - neg) / total.replace(0, np.nan)
        news_mean = total.rolling(24 * 7, min_periods=1).mean().replace(0, np.nan)
        news_std = total.rolling(24 * 7, min_periods=1).std().replace(0, np.nan)
        news["news_volume_zscore"] = (total - news_mean) / news_std

        news_features = news[["ts", "news_sentiment_score", "news_volume_zscore"]].copy()
        result = result.merge(news_features, on="ts", how="left")

    for col in get_sentiment_feature_columns():
        if col in result.columns:
            result[col] = result[col].fillna(0.0)

    return result


def compute_cross_exchange_features(
    binance_fr_df: pd.DataFrame,
    binance_oi_df: pd.DataFrame,
    bitunix_fr_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute cross-exchange features from Binance vs BitUnix data.

    Features:
    - funding_rate_spread: Binance funding rate - BitUnix funding rate
    - oi_divergence: Binance OI change - Coinalyze OI change (directional)
    """
    result = ohlcv_df.copy()

    if not binance_fr_df.empty and not bitunix_fr_df.empty:
        bn = binance_fr_df[["ts", "funding_rate"]].copy()
        bn = bn.rename(columns={"funding_rate": "bn_fr"})
        bx = bitunix_fr_df[["ts", "funding_rate"]].copy()
        bx = bx.rename(columns={"funding_rate": "bx_fr"})

        merged_fr = bn.merge(bx, on="ts", how="outer").sort_values("ts")
        merged_fr["bn_fr"] = merged_fr["bn_fr"].ffill()
        merged_fr["bx_fr"] = merged_fr["bx_fr"].ffill()
        merged_fr["funding_rate_spread"] = merged_fr["bn_fr"] - merged_fr["bx_fr"]

        result = result.merge(
            merged_fr[["ts", "funding_rate_spread"]], on="ts", how="left"
        )

    if not binance_oi_df.empty:
        oi = binance_oi_df[["ts", "oi_value"]].copy()
        oi["oi_divergence"] = oi["oi_value"].pct_change()
        result = result.merge(oi[["ts", "oi_divergence"]], on="ts", how="left")

    for col in get_cross_exchange_feature_columns():
        if col in result.columns:
            result[col] = result[col].fillna(0.0)

    return result


def get_sentiment_feature_columns() -> list[str]:
    """Return ordered list of sentiment feature column names."""
    return [
        "fear_greed_index",
        "fear_greed_change",
        "news_sentiment_score",
        "news_volume_zscore",
    ]


def get_cross_exchange_feature_columns() -> list[str]:
    """Return ordered list of cross-exchange feature column names."""
    return [
        "funding_rate_spread",
        "oi_divergence",
    ]
