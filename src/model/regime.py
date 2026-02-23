"""
Market regime detection using Hidden Markov Models.

Detects 4 market states: trending up, trending down, ranging, high-volatility.
Uses BTC + ETH as market leaders to determine regime.
Regime is used to:
  - Select different model weights per regime
  - Adjust position sizing (smaller in high-vol regimes)
  - Adjust FLAT threshold per regime
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    logger.info("hmmlearn not installed — regime detection uses fallback")


class MarketRegime(IntEnum):
    TRENDING_UP = 0
    TRENDING_DOWN = 1
    RANGING = 2
    HIGH_VOLATILITY = 3


REGIME_NAMES = {
    MarketRegime.TRENDING_UP: "Trending Up",
    MarketRegime.TRENDING_DOWN: "Trending Down",
    MarketRegime.RANGING: "Ranging",
    MarketRegime.HIGH_VOLATILITY: "High Volatility",
}


@dataclass
class RegimeConfig:
    """Configuration for regime detector."""
    n_states: int = 4
    lookback_hours: int = 168
    retrain_interval_hours: int = 24
    vol_window: int = 24
    trend_window: int = 48
    covariance_type: str = "full"
    n_iter: int = 100
    random_state: int = 42


class RegimeDetector:
    """
    Detects current market regime using an HMM on market features.

    Features used for regime classification:
    - Returns (1h, 4h rolling mean)
    - Volatility (rolling std of returns)
    - Trend strength (absolute rolling mean return / rolling std)
    - Volume change rate
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.model = None
        self._regime_map: Dict[int, MarketRegime] = {}
        self.is_fitted = False

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract regime-relevant features from OHLCV data."""
        close = pd.to_numeric(df["close"], errors="coerce")
        volume = pd.to_numeric(df["volume"], errors="coerce")

        cfg = self.config
        returns_1h = close.pct_change().fillna(0)
        returns_4h = close.pct_change(4).fillna(0)
        rolling_mean = returns_1h.rolling(cfg.trend_window).mean().fillna(0)
        rolling_std = returns_1h.rolling(cfg.vol_window).std().fillna(1e-10)
        trend_strength = (rolling_mean.abs() / rolling_std.replace(0, 1e-10)).fillna(0)
        vol_change = volume.pct_change().fillna(0).clip(-5, 5)

        features = np.column_stack([
            returns_1h.values,
            returns_4h.values,
            rolling_std.values,
            trend_strength.values,
            vol_change.values,
        ])

        features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
        return features

    def fit(self, btc_df: pd.DataFrame, eth_df: Optional[pd.DataFrame] = None) -> None:
        """
        Train the regime model on BTC (and optionally ETH) data.

        Parameters
        ----------
        btc_df : pd.DataFrame
            BTC OHLCV data.
        eth_df : pd.DataFrame | None
            ETH OHLCV data for additional signal.
        """
        features = self._extract_features(btc_df)

        if eth_df is not None and not eth_df.empty:
            eth_features = self._extract_features(eth_df)
            min_len = min(len(features), len(eth_features))
            features = np.concatenate([
                features[-min_len:], eth_features[-min_len:]
            ], axis=1)

        valid_mask = np.all(np.isfinite(features), axis=1)
        features = features[valid_mask]

        if len(features) < 100:
            logger.warning("Not enough data to fit regime model (%d rows)", len(features))
            return

        if HAS_HMM:
            self.model = GaussianHMM(
                n_components=self.config.n_states,
                covariance_type=self.config.covariance_type,
                n_iter=self.config.n_iter,
                random_state=self.config.random_state,
            )
            self.model.fit(features)
            states = self.model.predict(features)
        else:
            states = self._fallback_classify(features)

        self._map_states_to_regimes(features, states)
        self.is_fitted = True
        logger.info("Regime detector fitted on %d observations", len(features))

    def predict(self, btc_df: pd.DataFrame) -> MarketRegime:
        """Predict current market regime from recent BTC data."""
        if not self.is_fitted:
            return MarketRegime.RANGING

        features = self._extract_features(btc_df)
        if len(features) == 0:
            return MarketRegime.RANGING

        latest = features[-1:].reshape(1, -1)

        if HAS_HMM and self.model is not None:
            state = int(self.model.predict(latest)[0])
        else:
            state = self._fallback_classify_single(latest[0])

        return self._regime_map.get(state, MarketRegime.RANGING)

    def get_regime_adjustments(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Return recommended parameter adjustments for the detected regime.

        Returns dict with keys: size_multiplier, flat_threshold_adj, confidence_boost.
        """
        adjustments = {
            MarketRegime.TRENDING_UP: {
                "size_multiplier": 1.2,
                "flat_threshold_adj": 0.0,
                "confidence_boost": 0.1,
            },
            MarketRegime.TRENDING_DOWN: {
                "size_multiplier": 1.0,
                "flat_threshold_adj": 0.0,
                "confidence_boost": 0.05,
            },
            MarketRegime.RANGING: {
                "size_multiplier": 0.7,
                "flat_threshold_adj": 0.002,
                "confidence_boost": -0.1,
            },
            MarketRegime.HIGH_VOLATILITY: {
                "size_multiplier": 0.5,
                "flat_threshold_adj": 0.003,
                "confidence_boost": -0.15,
            },
        }
        return adjustments.get(regime, adjustments[MarketRegime.RANGING])

    def _map_states_to_regimes(
        self,
        features: np.ndarray,
        states: np.ndarray,
    ) -> None:
        """Map HMM states to meaningful regime labels based on feature statistics."""
        state_stats = {}
        for s in range(self.config.n_states):
            mask = states == s
            if mask.sum() == 0:
                state_stats[s] = {"mean_return": 0, "mean_vol": 0}
                continue
            state_features = features[mask]
            state_stats[s] = {
                "mean_return": float(np.mean(state_features[:, 0])),
                "mean_vol": float(np.mean(state_features[:, 2])),
            }

        sorted_by_vol = sorted(state_stats.items(), key=lambda x: x[1]["mean_vol"])
        sorted_by_return = sorted(state_stats.items(), key=lambda x: x[1]["mean_return"])

        self._regime_map = {}
        self._regime_map[sorted_by_vol[-1][0]] = MarketRegime.HIGH_VOLATILITY
        self._regime_map[sorted_by_return[-1][0]] = MarketRegime.TRENDING_UP
        self._regime_map[sorted_by_return[0][0]] = MarketRegime.TRENDING_DOWN

        for s in range(self.config.n_states):
            if s not in self._regime_map:
                self._regime_map[s] = MarketRegime.RANGING

    def _fallback_classify(self, features: np.ndarray) -> np.ndarray:
        """Simple rule-based classification when HMM is unavailable."""
        states = np.zeros(len(features), dtype=int)
        for i, row in enumerate(features):
            states[i] = self._fallback_classify_single(row)
        return states

    def _fallback_classify_single(self, feature_row: np.ndarray) -> int:
        """Classify a single observation using rules."""
        ret = feature_row[0]
        vol = feature_row[2] if len(feature_row) > 2 else 0

        if vol > 0.03:
            return MarketRegime.HIGH_VOLATILITY
        elif ret > 0.005:
            return MarketRegime.TRENDING_UP
        elif ret < -0.005:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING
