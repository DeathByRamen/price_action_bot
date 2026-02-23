"""
P&L-based optimization for the adaptive feedback loop.

Replaces accuracy-based threshold tuning with Sharpe-based tuning:
  - Optimizes the FLAT threshold to maximize backtest Sharpe over a recent window
  - Weights samples by P&L impact, not just correctness
  - Optimizes signal score formula via grid search / Bayesian optimization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PnLOptConfig:
    """Configuration for P&L-based optimization."""
    threshold_candidates: int = 50
    threshold_min: float = 0.002
    threshold_max: float = 0.015
    lookback_days: int = 14
    min_trades: int = 20
    risk_free_rate: float = 0.0
    ema_alpha: float = 0.3


class PnLOptimizer:
    """
    Optimizes trading parameters to maximize risk-adjusted returns.

    Key difference from accuracy-based tuning: a prediction that is
    "wrong" in direction but results in a small loss is weighted less
    than one that causes a large loss. Similarly, correct predictions
    with larger profits are weighted more.
    """

    def __init__(self, config: Optional[PnLOptConfig] = None):
        self.config = config or PnLOptConfig()

    def optimize_flat_threshold(
        self,
        actual_magnitudes: np.ndarray,
        predicted_directions: np.ndarray,
        predicted_magnitudes: np.ndarray,
        current_threshold: float = 0.005,
    ) -> Tuple[float, float]:
        """
        Find the FLAT threshold that maximizes Sharpe of a simulated strategy.

        Parameters
        ----------
        actual_magnitudes : (N,) actual % price changes
        predicted_directions : (N,) predicted directions (0=UP, 1=FLAT, 2=DOWN)
        predicted_magnitudes : (N,) predicted magnitudes
        current_threshold : current FLAT threshold

        Returns
        -------
        (optimal_threshold, sharpe_at_optimal)
        """
        cfg = self.config
        candidates = np.linspace(cfg.threshold_min, cfg.threshold_max, cfg.threshold_candidates)
        best_threshold = current_threshold
        best_sharpe = -np.inf

        for thresh in candidates:
            reclassified = self._classify_directions(actual_magnitudes, thresh)
            pnls = self._simulate_pnl(reclassified, actual_magnitudes, predicted_magnitudes)
            sharpe = self._compute_sharpe(pnls)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_threshold = float(thresh)

        smoothed = cfg.ema_alpha * best_threshold + (1 - cfg.ema_alpha) * current_threshold
        smoothed = max(cfg.threshold_min, min(cfg.threshold_max, smoothed))

        logger.info(
            "P&L threshold optimization: optimal=%.4f (Sharpe=%.3f), "
            "smoothed=%.4f (was %.4f)",
            best_threshold, best_sharpe, smoothed, current_threshold,
        )
        return smoothed, best_sharpe

    def compute_pnl_weights(
        self,
        symbols: np.ndarray,
        actual_magnitudes: np.ndarray,
        was_correct: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute per-symbol training weights based on P&L impact.

        Symbols where losses are larger get higher weights.
        """
        unique_symbols = np.unique(symbols)
        weights: Dict[str, float] = {}

        for sym in unique_symbols:
            mask = symbols == sym
            mags = actual_magnitudes[mask]
            correct = was_correct[mask]

            if len(mags) < 5:
                weights[sym] = 1.0
                continue

            losses = np.abs(mags[~correct]) if (~correct).any() else np.array([0.0])
            gains = np.abs(mags[correct]) if correct.any() else np.array([0.0])

            avg_loss = float(np.mean(losses))
            avg_gain = float(np.mean(gains))

            if avg_gain > 0:
                loss_gain_ratio = avg_loss / avg_gain
            else:
                loss_gain_ratio = 2.0

            weights[sym] = max(0.5, min(5.0, 1.0 + loss_gain_ratio))

        return weights

    def _classify_directions(
        self, magnitudes: np.ndarray, threshold: float
    ) -> np.ndarray:
        """Classify directions based on threshold."""
        directions = np.ones(len(magnitudes), dtype=int)  # FLAT=1
        directions[magnitudes > threshold] = 0             # UP=0
        directions[magnitudes < -threshold] = 2            # DOWN=2
        return directions

    def _simulate_pnl(
        self,
        directions: np.ndarray,
        actual_mags: np.ndarray,
        pred_mags: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate P&L from predictions.

        For UP predictions: P&L = actual_magnitude
        For DOWN predictions: P&L = -actual_magnitude
        For FLAT: P&L = 0 (no trade)
        """
        pnls = np.zeros(len(directions))

        up_mask = directions == 0
        down_mask = directions == 2

        pnls[up_mask] = actual_mags[up_mask]
        pnls[down_mask] = -actual_mags[down_mask]

        return pnls

    def _compute_sharpe(self, pnls: np.ndarray) -> float:
        """Compute annualized Sharpe ratio from P&L series."""
        if len(pnls) < 2:
            return 0.0

        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls, ddof=1)

        if std_pnl < 1e-10:
            return 0.0

        periods_per_year = 365 * 24
        return float((mean_pnl / std_pnl) * np.sqrt(periods_per_year))
