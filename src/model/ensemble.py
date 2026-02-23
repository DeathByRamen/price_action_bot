"""
Multi-timeframe ensemble predictor.

Combines predictions from two timeframes (e.g. 1h directional bias + 15m
entry timing) into a single ranked signal.  When both timeframes agree,
the signal is strongest; when they conflict, it's flagged.

Typical usage:
    combined = combine_timeframes(hourly_preds, fifteen_min_preds, config)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .predictor import Prediction

logger = logging.getLogger(__name__)

# Agreement classification thresholds
_AGREE_THRESHOLD = 0.55  # both must predict same direction with this min prob


@dataclass
class MultiTimeframePrediction:
    """Combined prediction from two timeframes."""

    symbol: str
    primary: Prediction          # e.g. 1h model
    secondary: Prediction        # e.g. 15m model

    combined_direction: str      # "UP", "FLAT", "DOWN"
    combined_prob_up: float
    combined_prob_flat: float
    combined_prob_down: float
    combined_magnitude: float
    combined_conviction: float
    combined_score: float

    agreement: str               # "STRONG", "PARTIAL", "CONFLICT"
    agreement_label: str         # "STRONG UP", "WEAK UP", "CONFLICT", etc.
    current_price: float


def compute_adaptive_weights(
    primary_accuracy: Optional[float] = None,
    secondary_accuracy: Optional[float] = None,
    default_primary: float = 0.6,
    default_secondary: float = 0.4,
) -> tuple[float, float]:
    """
    Compute ensemble weights from historical per-timeframe accuracy.

    If both accuracies are available, weights are proportional to accuracy.
    Otherwise, falls back to default config weights.
    """
    if primary_accuracy is not None and secondary_accuracy is not None:
        total = primary_accuracy + secondary_accuracy
        if total > 0:
            w1 = primary_accuracy / total
            w2 = secondary_accuracy / total
            logger.info(
                "Adaptive ensemble weights: primary=%.3f (acc=%.1f%%), "
                "secondary=%.3f (acc=%.1f%%)",
                w1, primary_accuracy * 100, w2, secondary_accuracy * 100,
            )
            return w1, w2

    logger.info(
        "Using default ensemble weights: primary=%.2f, secondary=%.2f",
        default_primary, default_secondary,
    )
    return default_primary, default_secondary


def _combine_probs_log_odds(
    p1: np.ndarray,
    p2: np.ndarray,
    w1: float,
    w2: float,
) -> np.ndarray:
    """
    Combine two probability vectors using log-odds (statistically sound).

    Converts softmax probabilities to log-odds space, takes weighted sum,
    converts back, and normalizes.

    Parameters
    ----------
    p1, p2 : np.ndarray of shape (3,) — [prob_up, prob_flat, prob_down]
    w1, w2 : float — ensemble weights

    Returns
    -------
    np.ndarray of shape (3,) — combined probabilities summing to 1.
    """
    eps = 1e-8
    p1 = np.clip(p1, eps, 1.0 - eps)
    p2 = np.clip(p2, eps, 1.0 - eps)

    # Convert each probability to log-odds: log(p / (1-p))
    lo1 = np.log(p1) - np.log(1.0 - p1)
    lo2 = np.log(p2) - np.log(1.0 - p2)

    # Weighted combination in log-odds space
    combined_lo = w1 * lo1 + w2 * lo2

    # Convert back to probabilities
    combined_p = 1.0 / (1.0 + np.exp(-combined_lo))

    # Normalize to sum to 1
    total = combined_p.sum()
    if total > 0:
        combined_p /= total

    return combined_p


def combine_timeframes(
    primary_preds: List[Prediction],
    secondary_preds: List[Prediction],
    primary_weight: float = 0.6,
    secondary_weight: float = 0.4,
) -> List[MultiTimeframePrediction]:
    """
    Combine predictions from two timeframes into an ensemble.

    Uses log-odds probability combination (statistically sound) instead of
    naive weighted averaging, and magnitude-aware conflict resolution.

    Parameters
    ----------
    primary_preds : list
        Predictions from the longer timeframe (e.g. 1h) — directional bias.
    secondary_preds : list
        Predictions from the shorter timeframe (e.g. 15m) — entry timing.
    primary_weight : float
        Weight for the primary timeframe in the ensemble (default 0.6).
    secondary_weight : float
        Weight for the secondary timeframe (default 0.4).

    Returns
    -------
    List of MultiTimeframePrediction, sorted by combined_score descending.
    """
    sec_by_symbol: Dict[str, Prediction] = {p.symbol: p for p in secondary_preds}

    combined: List[MultiTimeframePrediction] = []

    for pri in primary_preds:
        sec = sec_by_symbol.get(pri.symbol)
        if sec is None:
            continue

        # Log-odds probability combination (statistically sound)
        p1 = np.array([pri.prob_up, pri.prob_flat, pri.prob_down])
        p2 = np.array([sec.prob_up, sec.prob_flat, sec.prob_down])
        combined_p = _combine_probs_log_odds(p1, p2, primary_weight, secondary_weight)

        prob_up = float(combined_p[0])
        prob_flat = float(combined_p[1])
        prob_down = float(combined_p[2])

        # Combined magnitude (weighted average)
        magnitude = primary_weight * pri.magnitude + secondary_weight * sec.magnitude

        # Combined conviction (geometric mean — both must be confident)
        conviction = math.sqrt(max(pri.conviction, 0) * max(sec.conviction, 0))

        # Direction from combined probabilities
        probs = {"UP": prob_up, "FLAT": prob_flat, "DOWN": prob_down}
        direction = max(probs, key=probs.get)  # type: ignore[arg-type]

        # Agreement classification (magnitude-aware)
        agreement, agreement_label = _classify_agreement(pri, sec, direction)

        # Combined signal score
        agreement_multiplier = {"STRONG": 1.5, "PARTIAL": 1.0, "CONFLICT": 0.3}
        directional_prob = max(prob_up, prob_down)

        eps = 1e-10
        entropy = -sum(p * math.log(p + eps) for p in [prob_up, prob_flat, prob_down])
        max_entropy = math.log(3)
        norm_conviction = max(0.0, 1.0 - (entropy / max_entropy))

        score = (
            norm_conviction
            * directional_prob
            * abs(magnitude)
            * agreement_multiplier[agreement]
        )

        combined.append(
            MultiTimeframePrediction(
                symbol=pri.symbol,
                primary=pri,
                secondary=sec,
                combined_direction=direction,
                combined_prob_up=prob_up,
                combined_prob_flat=prob_flat,
                combined_prob_down=prob_down,
                combined_magnitude=magnitude,
                combined_conviction=conviction,
                combined_score=score,
                agreement=agreement,
                agreement_label=agreement_label,
                current_price=pri.current_price,
            )
        )

    combined.sort(key=lambda p: p.combined_score, reverse=True)
    logger.info(
        "Ensemble: %d combined predictions (%d strong, %d partial, %d conflict)",
        len(combined),
        sum(1 for p in combined if p.agreement == "STRONG"),
        sum(1 for p in combined if p.agreement == "PARTIAL"),
        sum(1 for p in combined if p.agreement == "CONFLICT"),
    )
    return combined


# Magnitude below which a secondary disagreement is a pullback, not a conflict
_PULLBACK_MAG_THRESHOLD = 0.01  # 1%


def _classify_agreement(
    pri: Prediction, sec: Prediction, combined_dir: str
) -> tuple[str, str]:
    """
    Classify whether the two timeframes agree on direction.

    Uses magnitude-aware logic: if the primary says UP but the secondary says
    DOWN with a tiny magnitude (<1%), it's a pullback within the trend, not a
    genuine conflict.
    """
    pri_dir = pri.direction
    sec_dir = sec.direction

    if pri_dir == sec_dir and pri_dir != "FLAT":
        return "STRONG", f"STRONG {pri_dir}"
    elif pri_dir == sec_dir and pri_dir == "FLAT":
        return "PARTIAL", "BOTH FLAT"
    elif pri_dir != "FLAT" and sec_dir == "FLAT":
        return "PARTIAL", f"WEAK {pri_dir}"
    elif pri_dir == "FLAT" and sec_dir != "FLAT":
        return "PARTIAL", f"TIMING {sec_dir}"
    else:
        # Both directional but opposite — check magnitudes
        sec_mag_small = abs(sec.magnitude) < _PULLBACK_MAG_THRESHOLD
        pri_mag_small = abs(pri.magnitude) < _PULLBACK_MAG_THRESHOLD

        if sec_mag_small and not pri_mag_small:
            # Secondary disagrees but with tiny magnitude = pullback in trend
            return "PARTIAL", f"PULLBACK {pri_dir}"
        elif pri_mag_small and not sec_mag_small:
            # Primary is weak, secondary has conviction
            return "PARTIAL", f"TIMING {sec_dir}"
        else:
            # Both have significant magnitude in opposite directions
            return "CONFLICT", "CONFLICT"


def format_multi_timeframe_message(
    predictions: List[MultiTimeframePrediction],
    top_n: int = 10,
    primary_label: str = "1h",
    secondary_label: str = "15m",
) -> str:
    """Format multi-timeframe predictions as a notification message."""
    from datetime import datetime, timezone

    preds = predictions[:top_n]
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Count totals across ALL predictions (not just top N)
    total_strong = sum(1 for p in predictions if p.agreement == "STRONG")
    total_partial = sum(1 for p in predictions if p.agreement == "PARTIAL")
    total_conflict = sum(1 for p in predictions if p.agreement == "CONFLICT")
    total_up = sum(1 for p in predictions if p.combined_direction == "UP")
    total_down = sum(1 for p in predictions if p.combined_direction == "DOWN")

    header = f"PA Bot — Ensemble ({primary_label} + {secondary_label})  |  {now_utc}"

    lines = [
        header,
        "=" * len(header),
        f"Analyzed {len(predictions)} symbols  |  "
        f"Market: {total_up} UP / {total_down} DOWN  |  "
        f"{total_strong} strong / {total_partial} partial / {total_conflict} conflict",
        "",
        f"Top {len(preds)} signals by conviction:",
        "",
        f"{'#':>3}  {'Symbol':<14} {primary_label:>4} {secondary_label:>4}  {'Signal':<13} "
        f"{'Prob':>6}  {'Mag%':>7}  {'Price':>12}  {'Score':>7}",
        f"{'─'*3}  {'─'*14} {'─'*4} {'─'*4}  {'─'*13} "
        f"{'─'*6}  {'─'*7}  {'─'*12}  {'─'*7}",
    ]

    for i, p in enumerate(preds, 1):
        pri_icon = "▲" if p.primary.direction == "UP" else "▼" if p.primary.direction == "DOWN" else "─"
        sec_icon = "▲" if p.secondary.direction == "UP" else "▼" if p.secondary.direction == "DOWN" else "─"
        lines.append(
            f"{i:>3}  {p.symbol:<14} "
            f" {pri_icon}{p.primary.direction[0]:>1}   {sec_icon}{p.secondary.direction[0]:>1}  "
            f"{p.agreement_label:<13} "
            f"{max(p.combined_prob_up, p.combined_prob_down):>5.1%}  "
            f"{p.combined_magnitude:>+6.2%}  "
            f"{p.current_price:>12.4f}  "
            f"{p.combined_score:>7.4f}"
        )

    lines.append("")
    lines.append(f"Legend: {primary_label} = directional bias | {secondary_label} = entry timing")
    lines.append("        Signal = agreement level | Prob = combined confidence")
    lines.append("        Mag% = predicted move | Score = signal strength")
    lines.append("")
    lines.append("STRONG = both agree  |  PARTIAL = one flat/pullback  |  CONFLICT = oppose")

    return "\n".join(lines)
