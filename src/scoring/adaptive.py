"""
Adaptive feedback tuner.

Two mechanisms that improve model performance over time:

1. FLAT_THRESHOLD auto-tuning
   - Analyzes recent prediction outcomes to find the optimal boundary
     between "flat" and directional moves.
   - Uses an EMA to smooth the threshold across days.

2. Per-symbol sample weighting
   - Symbols the model gets wrong more often receive higher training weights
     so the model focuses gradient on its weak spots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.storage import Storage

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    """Parameters controlling the adaptive tuner."""
    flat_threshold_min: float = 0.002
    flat_threshold_max: float = 0.015
    threshold_ema_alpha: float = 0.3
    weight_lookback_days: int = 7
    min_preds_for_weight: int = 5
    max_weight: float = 5.0


async def compute_optimal_threshold(
    storage: Storage,
    lookback_days: int = 7,
    current_threshold: float = 0.005,
    config: Optional[AdaptiveConfig] = None,
) -> float:
    """
    Compute the optimal FLAT_THRESHOLD based on recent prediction outcomes.

    Strategy: find the threshold that maximizes overall direction accuracy
    by testing a range of values against the actual magnitudes observed.
    Then EMA-smooth with the current threshold to avoid sudden jumps.
    """
    cfg = config or AdaptiveConfig()
    df = await storage.get_scored_predictions(days=lookback_days)

    if df.empty or len(df) < 20:
        logger.info("Not enough scored data for threshold tuning, keeping %.4f", current_threshold)
        return current_threshold

    actual_mags = pd.to_numeric(df["actual_magnitude"], errors="coerce").dropna().abs()
    if actual_mags.empty:
        return current_threshold

    # Test a range of thresholds and pick the one that maximizes accuracy
    candidates = np.linspace(cfg.flat_threshold_min, cfg.flat_threshold_max, 50)
    best_threshold = current_threshold
    best_accuracy = 0.0

    for thresh in candidates:
        # Re-classify actuals with this threshold
        actual_dirs = df["actual_magnitude"].apply(
            lambda m: "UP" if float(m) > thresh else ("DOWN" if float(m) < -thresh else "FLAT")
        )
        # Re-classify predictions with this threshold
        pred_dirs = df["magnitude"].apply(
            lambda m: "UP" if float(m) > thresh else ("DOWN" if float(m) < -thresh else "FLAT")
        )
        accuracy = (pred_dirs == actual_dirs).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(thresh)

    # EMA smooth: new = alpha * optimal + (1-alpha) * current
    smoothed = cfg.threshold_ema_alpha * best_threshold + (1 - cfg.threshold_ema_alpha) * current_threshold

    # Clamp to bounds
    smoothed = max(cfg.flat_threshold_min, min(cfg.flat_threshold_max, smoothed))

    logger.info(
        "Threshold tuning: optimal=%.4f, smoothed=%.4f (was %.4f, accuracy=%.1f%%)",
        best_threshold,
        smoothed,
        current_threshold,
        best_accuracy * 100,
    )

    return smoothed


async def compute_sample_weights(
    storage: Storage,
    lookback_days: int = 7,
    config: Optional[AdaptiveConfig] = None,
) -> Dict[str, float]:
    """
    Compute per-symbol training weights based on recent prediction errors.

    Symbols the model gets wrong more often get higher weights so the
    training loop focuses more gradient on hard cases.

    Returns {symbol: weight} dict.  Weight=1.0 is the baseline.
    """
    cfg = config or AdaptiveConfig()
    df = await storage.get_scored_predictions(days=lookback_days)

    if df.empty:
        logger.info("No scored data for weight computation")
        return {}

    # Group by symbol
    sym_stats = df.groupby("symbol").agg(
        n_preds=("was_correct", "count"),
        n_correct=("was_correct", "sum"),
    )
    sym_stats["error_rate"] = 1.0 - (sym_stats["n_correct"] / sym_stats["n_preds"])

    # Only compute weights for symbols with enough predictions
    sym_stats = sym_stats[sym_stats["n_preds"] >= cfg.min_preds_for_weight]

    if sym_stats.empty:
        return {}

    # Weight = 1 + normalized_error_rate
    # Symbols with higher error get more weight, capped at max_weight
    mean_error = sym_stats["error_rate"].mean()
    if mean_error > 0:
        sym_stats["weight"] = 1.0 + (sym_stats["error_rate"] / mean_error)
    else:
        sym_stats["weight"] = 1.0

    sym_stats["weight"] = sym_stats["weight"].clip(upper=cfg.max_weight)

    weights = sym_stats["weight"].to_dict()

    # Persist to DB
    weight_rows = [
        (sym, w, float(sym_stats.loc[sym, "error_rate"]), int(sym_stats.loc[sym, "n_preds"]))
        for sym, w in weights.items()
    ]
    await storage.upsert_sample_weights(weight_rows)

    logger.info(
        "Computed sample weights for %d symbols (mean=%.2f, max=%.2f)",
        len(weights),
        np.mean(list(weights.values())),
        max(weights.values()) if weights else 0,
    )

    return weights
