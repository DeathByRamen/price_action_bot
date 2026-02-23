"""
Prediction drift monitoring.

Tracks prediction distribution shifts and calibration degradation
over time. Triggers alerts when significant changes are detected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftConfig:
    """Configuration for drift monitoring."""
    kl_threshold: float = 0.1
    calibration_window: int = 100
    distribution_window: int = 200
    min_samples: int = 30
    calibration_bins: int = 10


@dataclass
class DriftReport:
    """Report on prediction drift and calibration quality."""
    kl_divergence: float = 0.0
    distribution_shift: bool = False
    calibration_error: float = 0.0
    calibration_degraded: bool = False
    prediction_counts: Dict[str, int] = field(default_factory=dict)
    calibration_per_bin: List[Tuple[float, float, int]] = field(default_factory=list)
    timestamp: str = ""


class DriftMonitor:
    """
    Monitors prediction drift and calibration quality.

    Detects:
    - Distribution shifts in predictions (% UP/DOWN/FLAT changing)
    - Calibration degradation (predicted 70% UP not matching ~70% actual UP)
    - KL divergence between recent and historical prediction distributions
    """

    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        self._prediction_history: List[Dict] = []
        self._baseline_distribution: Optional[np.ndarray] = None

    def record_prediction(
        self,
        direction: str,
        prob_up: float,
        prob_flat: float,
        prob_down: float,
        actual_direction: Optional[str] = None,
    ) -> None:
        """Record a prediction for drift tracking."""
        self._prediction_history.append({
            "direction": direction,
            "prob_up": prob_up,
            "prob_flat": prob_flat,
            "prob_down": prob_down,
            "actual": actual_direction,
        })

    def set_baseline(self, n_recent: Optional[int] = None) -> None:
        """Set current distribution as baseline for future comparison."""
        n = n_recent or len(self._prediction_history)
        recent = self._prediction_history[-n:]
        if len(recent) < self.config.min_samples:
            return

        counts = {"UP": 0, "FLAT": 0, "DOWN": 0}
        for p in recent:
            counts[p["direction"]] = counts.get(p["direction"], 0) + 1
        total = sum(counts.values())
        self._baseline_distribution = np.array([
            counts["UP"] / total,
            counts["FLAT"] / total,
            counts["DOWN"] / total,
        ])

    def compute_drift(self) -> DriftReport:
        """Compute current drift metrics."""
        report = DriftReport()
        cfg = self.config

        recent = self._prediction_history[-cfg.distribution_window:]
        if len(recent) < cfg.min_samples:
            return report

        counts = {"UP": 0, "FLAT": 0, "DOWN": 0}
        for p in recent:
            counts[p["direction"]] = counts.get(p["direction"], 0) + 1
        report.prediction_counts = counts

        total = sum(counts.values())
        current_dist = np.array([
            counts["UP"] / total,
            counts["FLAT"] / total,
            counts["DOWN"] / total,
        ])

        if self._baseline_distribution is not None:
            report.kl_divergence = float(self._kl_divergence(
                current_dist, self._baseline_distribution
            ))
            report.distribution_shift = report.kl_divergence > cfg.kl_threshold

            if report.distribution_shift:
                logger.warning(
                    "Prediction distribution shift detected: KL=%.4f (threshold=%.4f)",
                    report.kl_divergence, cfg.kl_threshold,
                )

        scored = [p for p in recent if p.get("actual") is not None]
        if len(scored) >= cfg.min_samples:
            report.calibration_error, report.calibration_per_bin = (
                self._compute_calibration(scored)
            )
            report.calibration_degraded = report.calibration_error > 0.15

            if report.calibration_degraded:
                logger.warning(
                    "Calibration degraded: ECE=%.4f — consider recalibration",
                    report.calibration_error,
                )

        return report

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL(P || Q) with smoothing to avoid log(0)."""
        eps = 1e-10
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        p = p / p.sum()
        q = q / q.sum()
        return float(np.sum(p * np.log(p / q)))

    def _compute_calibration(
        self,
        scored_predictions: List[Dict],
    ) -> Tuple[float, List[Tuple[float, float, int]]]:
        """
        Compute Expected Calibration Error (ECE).

        For each probability bin, checks if predicted confidence
        matches actual accuracy.
        """
        bins = self.config.calibration_bins
        bin_edges = np.linspace(0, 1, bins + 1)
        calibration_data: List[Tuple[float, float, int]] = []
        total_ece = 0.0
        total_samples = 0

        for i in range(bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            in_bin = []

            for p in scored_predictions:
                max_prob = max(p["prob_up"], p["prob_flat"], p["prob_down"])
                if lo <= max_prob < hi:
                    correct = p["direction"] == p["actual"]
                    in_bin.append((max_prob, correct))

            if not in_bin:
                continue

            avg_confidence = np.mean([x[0] for x in in_bin])
            avg_accuracy = np.mean([x[1] for x in in_bin])
            count = len(in_bin)

            calibration_data.append((float(avg_confidence), float(avg_accuracy), count))
            total_ece += abs(avg_confidence - avg_accuracy) * count
            total_samples += count

        ece = total_ece / max(total_samples, 1)
        return float(ece), calibration_data
