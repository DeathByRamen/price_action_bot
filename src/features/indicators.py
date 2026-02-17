"""
Technical indicator feature engineering (quant-grade).

All features are price-relative or bounded — no raw price/volume values
that would differ by orders of magnitude across symbols.

Accepts a DataFrame with columns [open, high, low, close, volume]
and returns the same frame augmented with indicator columns ready
for model consumption.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochRSIIndicator, WilliamsRIndicator, ROCIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator

logger = logging.getLogger(__name__)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicator columns to *df* **in place** and return it.

    Every feature is either:
      - A bounded oscillator (0-100, 0-1, etc.)
      - A ratio or percentage
      - Normalized by ATR or close price

    This ensures features are comparable across symbols with vastly
    different price levels (e.g. BTC at $60k vs an altcoin at $0.003).
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]  # noqa: E741
    o = df["open"]
    v = df["volume"]

    # Compute ATR first — used to normalize price-distance features
    atr_raw = AverageTrueRange(h, l, c, window=14).average_true_range()
    atr_safe = atr_raw.replace(0, np.nan)
    c_safe = c.replace(0, np.nan)

    # ----------------------------------------------------------------
    # Trend indicators (price-relative distances)
    # ----------------------------------------------------------------
    ema_9 = EMAIndicator(c, window=9).ema_indicator()
    ema_21 = EMAIndicator(c, window=21).ema_indicator()
    ema_50 = EMAIndicator(c, window=50).ema_indicator()

    # Distance from EMA in ATR units (scale-invariant)
    df["ema_9_dist"] = (c - ema_9) / atr_safe
    df["ema_21_dist"] = (c - ema_21) / atr_safe
    df["ema_50_dist"] = (c - ema_50) / atr_safe

    macd_obj = MACD(c, window_slow=26, window_fast=12, window_sign=9)
    # Normalize MACD by close price so it's a percentage
    df["macd_norm"] = macd_obj.macd() / c_safe
    df["macd_signal_norm"] = macd_obj.macd_signal() / c_safe
    df["macd_hist_norm"] = macd_obj.macd_diff() / c_safe

    adx_obj = ADXIndicator(h, l, c, window=14)
    df["adx"] = adx_obj.adx()          # 0–100 bounded
    df["adx_pos"] = adx_obj.adx_pos()  # 0–100 bounded
    df["adx_neg"] = adx_obj.adx_neg()  # 0–100 bounded

    # ----------------------------------------------------------------
    # Momentum indicators (already bounded / relative)
    # ----------------------------------------------------------------
    df["rsi_14"] = RSIIndicator(c, window=14).rsi()  # 0–100

    stoch_rsi = StochRSIIndicator(c, window=14, smooth1=3, smooth2=3)
    df["stoch_rsi_k"] = stoch_rsi.stochrsi_k()  # 0–1
    df["stoch_rsi_d"] = stoch_rsi.stochrsi_d()  # 0–1

    df["williams_r"] = WilliamsRIndicator(h, l, c, lbp=14).williams_r()  # -100–0
    df["roc_12"] = ROCIndicator(c, window=12).roc()  # percentage

    # ----------------------------------------------------------------
    # Volatility indicators (price-relative)
    # ----------------------------------------------------------------
    bb = BollingerBands(c, window=20, window_dev=2)
    df["bb_width"] = bb.bollinger_wband()   # already relative (bandwidth)
    df["bb_pct"] = bb.bollinger_pband()     # already relative (%B)
    df["atr_pct"] = atr_raw / c_safe        # ATR as fraction of price

    kc = KeltnerChannel(h, l, c, window=20, window_atr=10)
    kc_upper = kc.keltner_channel_hband()
    kc_lower = kc.keltner_channel_lband()
    df["kc_width"] = (kc_upper - kc_lower) / c_safe  # channel width as % of price
    df["kc_dist"] = (c - (kc_upper + kc_lower) / 2) / atr_safe  # distance from midline

    # ----------------------------------------------------------------
    # Volume indicators (rate-of-change / relative)
    # ----------------------------------------------------------------
    obv = OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["obv_roc"] = obv.pct_change(5).replace([np.inf, -np.inf], np.nan)

    acc_dist = AccDistIndexIndicator(h, l, c, v).acc_dist_index()
    df["acc_dist_roc"] = acc_dist.pct_change(5).replace([np.inf, -np.inf], np.nan)

    vol_sma_20 = v.rolling(window=20).mean()
    df["vol_sma_ratio"] = v / vol_sma_20.replace(0, np.nan)

    vol_std = v.rolling(window=20).std()
    df["vol_z_score"] = (v - vol_sma_20) / vol_std.replace(0, np.nan)

    # VWAP — rolling 24-hour window (resets daily, not cumulative)
    typical = (h + l + c) / 3.0
    tp_vol = typical * v
    vwap_24 = tp_vol.rolling(24, min_periods=1).sum() / v.rolling(24, min_periods=1).sum().replace(0, np.nan)
    df["vwap_dist"] = (c - vwap_24) / atr_safe  # distance from VWAP in ATR units

    # ----------------------------------------------------------------
    # Custom / derived features (all ratios or percentages)
    # ----------------------------------------------------------------
    df["pct_change_1"] = c.pct_change(1)
    df["pct_change_4"] = c.pct_change(4)
    df["pct_change_24"] = c.pct_change(24)

    # Multi-day momentum — captures extended pumps/dumps
    df["pct_change_72"] = c.pct_change(72)    # 3-day change (or 3d of 1h candles)
    df["pct_change_168"] = c.pct_change(168)  # 7-day change

    # Distance from rolling high/low — how extended is the current price?
    # Near 1.0 = at the high (potential reversal), near 0.0 = at the low
    high_48 = h.rolling(48).max()
    low_48 = l.rolling(48).min()
    range_48 = (high_48 - low_48).replace(0, np.nan)
    df["price_position_48"] = (c - low_48) / range_48  # 0-1 bounded

    high_168 = h.rolling(168).max()
    low_168 = l.rolling(168).min()
    range_168 = (high_168 - low_168).replace(0, np.nan)
    df["price_position_168"] = (c - low_168) / range_168  # 0-1 bounded

    # Pump/dump detector: price acceleration (rate of change of rate of change)
    roc_short = c.pct_change(4)
    roc_long = c.pct_change(24)
    df["momentum_accel"] = roc_short - (roc_long / 6)  # normalized to same scale

    # Volatility expansion: current ATR vs historical ATR (mean reversion signal)
    atr_sma_48 = atr_raw.rolling(48).mean().replace(0, np.nan)
    df["atr_expansion"] = atr_raw / atr_sma_48  # >1 = volatility expanding

    # Volume-price divergence: big volume but small price move = distribution
    abs_pct = c.pct_change(1).abs().replace(0, np.nan)
    df["vol_price_ratio"] = df["vol_sma_ratio"] / (abs_pct * 100 + 0.01)

    body = c - o
    candle_range = h - l
    candle_range_safe = candle_range.replace(0, np.nan)
    df["candle_body_ratio"] = body / candle_range_safe
    df["upper_wick_ratio"] = (
        h - pd.concat([o, c], axis=1).max(axis=1)
    ) / candle_range_safe
    df["lower_wick_ratio"] = (
        pd.concat([o, c], axis=1).min(axis=1) - l
    ) / candle_range_safe

    # EMA crossover signals (binary 0/1)
    df["ema_9_21_cross"] = (ema_9 > ema_21).astype(float)
    df["ema_21_50_cross"] = (ema_21 > ema_50).astype(float)

    # Price relative to Bollinger Bands in ATR units
    df["price_vs_bb_upper"] = (c - bb.bollinger_hband()) / atr_safe
    df["price_vs_bb_lower"] = (c - bb.bollinger_lband()) / atr_safe

    return df


def get_feature_columns() -> list[str]:
    """Return the ordered list of indicator column names used as model features."""
    return [
        # Trend (price-relative)
        "ema_9_dist", "ema_21_dist", "ema_50_dist",
        "macd_norm", "macd_signal_norm", "macd_hist_norm",
        "adx", "adx_pos", "adx_neg",
        # Momentum (bounded)
        "rsi_14", "stoch_rsi_k", "stoch_rsi_d",
        "williams_r", "roc_12",
        # Volatility (price-relative)
        "bb_width", "bb_pct", "atr_pct",
        "kc_width", "kc_dist",
        # Volume (rate-of-change / relative)
        "obv_roc", "acc_dist_roc", "vol_sma_ratio", "vol_z_score", "vwap_dist",
        # Custom (ratios / percentages)
        "pct_change_1", "pct_change_4", "pct_change_24",
        "pct_change_72", "pct_change_168",
        "candle_body_ratio", "upper_wick_ratio", "lower_wick_ratio",
        "ema_9_21_cross", "ema_21_50_cross",
        "price_vs_bb_upper", "price_vs_bb_lower",
        # Anti-pump/dump & mean reversion
        "price_position_48", "price_position_168",
        "momentum_accel", "atr_expansion",
        "vol_price_ratio",
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
