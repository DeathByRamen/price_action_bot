"""
Technical indicator feature engineering.

Accepts a DataFrame with columns [open, high, low, close, volume]
and returns the same frame augmented with ~30 indicator columns ready
for model consumption.

Uses the `ta` library where convenient and raw pandas/numpy otherwise.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import ta
from ta.momentum import RSIIndicator, StochRSIIndicator, WilliamsRIndicator, ROCIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator

logger = logging.getLogger(__name__)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicator columns to *df* **in place** and return it.

    Expects columns: open, high, low, close, volume.
    The first ~50 rows will have NaN in some indicator columns due to
    lookback requirements; callers should either drop or forward-fill them.
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]  # noqa: E741
    o = df["open"]
    v = df["volume"]

    # ----------------------------------------------------------------
    # Trend indicators
    # ----------------------------------------------------------------
    df["ema_9"] = EMAIndicator(c, window=9).ema_indicator()
    df["ema_21"] = EMAIndicator(c, window=21).ema_indicator()
    df["ema_50"] = EMAIndicator(c, window=50).ema_indicator()

    macd = MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    adx = ADXIndicator(h, l, c, window=14)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # ----------------------------------------------------------------
    # Momentum indicators
    # ----------------------------------------------------------------
    df["rsi_14"] = RSIIndicator(c, window=14).rsi()

    stoch_rsi = StochRSIIndicator(c, window=14, smooth1=3, smooth2=3)
    df["stoch_rsi_k"] = stoch_rsi.stochrsi_k()
    df["stoch_rsi_d"] = stoch_rsi.stochrsi_d()

    df["williams_r"] = WilliamsRIndicator(h, l, c, lbp=14).williams_r()
    df["roc_12"] = ROCIndicator(c, window=12).roc()

    # ----------------------------------------------------------------
    # Volatility indicators
    # ----------------------------------------------------------------
    bb = BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pct"] = bb.bollinger_pband()

    df["atr_14"] = AverageTrueRange(h, l, c, window=14).average_true_range()

    kc = KeltnerChannel(h, l, c, window=20, window_atr=10)
    df["kc_upper"] = kc.keltner_channel_hband()
    df["kc_lower"] = kc.keltner_channel_lband()

    # ----------------------------------------------------------------
    # Volume indicators
    # ----------------------------------------------------------------
    df["obv"] = OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["acc_dist"] = AccDistIndexIndicator(h, l, c, v).acc_dist_index()

    vol_sma_20 = v.rolling(window=20).mean()
    df["vol_sma_ratio"] = v / vol_sma_20.replace(0, np.nan)

    vol_std = v.rolling(window=20).std()
    df["vol_z_score"] = (v - vol_sma_20) / vol_std.replace(0, np.nan)

    # VWAP approximation (cumulative within the available window)
    typical = (h + l + c) / 3.0
    cum_tp_vol = (typical * v).cumsum()
    cum_vol = v.cumsum()
    df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)

    # ----------------------------------------------------------------
    # Custom / derived features
    # ----------------------------------------------------------------
    df["pct_change_1"] = c.pct_change(1)
    df["pct_change_4"] = c.pct_change(4)
    df["pct_change_24"] = c.pct_change(24)

    body = c - o
    candle_range = h - l
    df["candle_body_ratio"] = body / candle_range.replace(0, np.nan)
    df["upper_wick_ratio"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / candle_range.replace(0, np.nan)
    df["lower_wick_ratio"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / candle_range.replace(0, np.nan)

    # EMA crossover signals (binary)
    df["ema_9_21_cross"] = (df["ema_9"] > df["ema_21"]).astype(float)
    df["ema_21_50_cross"] = (df["ema_21"] > df["ema_50"]).astype(float)

    # Price relative to Bollinger Bands
    df["price_vs_bb_upper"] = (c - df["bb_upper"]) / df["atr_14"].replace(0, np.nan)
    df["price_vs_bb_lower"] = (c - df["bb_lower"]) / df["atr_14"].replace(0, np.nan)

    return df


def get_feature_columns() -> list[str]:
    """Return the ordered list of indicator column names used as model features."""
    return [
        # Trend
        "ema_9", "ema_21", "ema_50",
        "macd", "macd_signal", "macd_hist",
        "adx", "adx_pos", "adx_neg",
        # Momentum
        "rsi_14", "stoch_rsi_k", "stoch_rsi_d",
        "williams_r", "roc_12",
        # Volatility
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct",
        "atr_14", "kc_upper", "kc_lower",
        # Volume
        "obv", "acc_dist", "vol_sma_ratio", "vol_z_score", "vwap",
        # Custom
        "pct_change_1", "pct_change_4", "pct_change_24",
        "candle_body_ratio", "upper_wick_ratio", "lower_wick_ratio",
        "ema_9_21_cross", "ema_21_50_cross",
        "price_vs_bb_upper", "price_vs_bb_lower",
    ]


def normalize_features(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Z-score normalize feature columns.
    Returns a copy with only the feature columns, NaN-filled rows dropped.
    """
    cols = feature_cols or get_feature_columns()
    out = df[cols].copy()
    for col in cols:
        mean = out[col].mean()
        std = out[col].std()
        if std > 0:
            out[col] = (out[col] - mean) / std
        else:
            out[col] = 0.0
    return out
