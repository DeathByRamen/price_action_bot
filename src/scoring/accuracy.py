"""
Accuracy scorer: joins predictions to actual OHLCV outcomes and computes
performance metrics.

For each un-scored prediction, looks up the candle at `prediction_ts + horizon`
to determine what actually happened, then computes:
  - Direction accuracy (overall + per-class precision/recall)
  - Magnitude MAE (predicted vs actual % change)
  - Per-symbol hit rates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.data.storage import Storage

logger = logging.getLogger(__name__)

DIRECTION_MAP = {"UP": 0, "FLAT": 1, "DOWN": 2}


@dataclass
class AccuracyReport:
    """Container for a scoring run's aggregate metrics."""
    run_date: str
    total_scored: int = 0
    direction_accuracy: float = 0.0
    magnitude_mae: float = 0.0
    up_precision: float = 0.0
    up_recall: float = 0.0
    down_precision: float = 0.0
    down_recall: float = 0.0
    flat_precision: float = 0.0
    flat_recall: float = 0.0
    flat_threshold_used: float = 0.005
    per_symbol_accuracy: Dict[str, float] = field(default_factory=dict)
    top_symbols: List[Tuple[str, float]] = field(default_factory=list)
    worst_symbols: List[Tuple[str, float]] = field(default_factory=list)

    def as_db_row(self) -> tuple:
        return (
            self.run_date,
            self.total_scored,
            self.direction_accuracy,
            self.magnitude_mae,
            self.up_precision,
            self.up_recall,
            self.down_precision,
            self.down_recall,
            self.flat_precision,
            self.flat_recall,
            self.flat_threshold_used,
        )


def classify_direction(pct_change: float, threshold: float = 0.005) -> str:
    if pct_change > threshold:
        return "UP"
    elif pct_change < -threshold:
        return "DOWN"
    return "FLAT"


async def score_predictions(
    storage: Storage,
    flat_threshold: float = 0.005,
) -> int:
    """
    Score all un-scored predictions by looking up the next candle's close price.

    Returns the number of predictions scored.
    """
    unscored = await storage.get_unscored_predictions()
    if unscored.empty:
        logger.info("No un-scored predictions to process")
        return 0

    scored_count = 0
    now_str = datetime.now(timezone.utc).isoformat()

    for _, row in unscored.iterrows():
        symbol = row["symbol"]
        pred_ts = row["ts"]
        pred_direction = row["direction"]
        pred_id = int(row["id"])
        interval = row.get("interval", "60")

        # Get the close price at prediction time (same interval)
        pred_close = await storage.get_close_at_ts(symbol, pred_ts, interval=interval)
        if pred_close is None or pred_close == 0:
            continue

        # Get the next candle after prediction time (same interval)
        next_candle = await storage.get_next_candle_close(symbol, pred_ts, interval=interval)
        if next_candle is None:
            continue

        _, actual_close = next_candle
        actual_magnitude = (actual_close - pred_close) / pred_close
        actual_direction = classify_direction(actual_magnitude, flat_threshold)
        was_correct = pred_direction == actual_direction

        await storage.update_prediction_outcome(
            pred_id=pred_id,
            actual_direction=actual_direction,
            actual_magnitude=actual_magnitude,
            was_correct=was_correct,
            scored_at=now_str,
        )
        scored_count += 1

    await storage.commit()
    logger.info("Scored %d predictions", scored_count)
    return scored_count


async def compute_accuracy_report(
    storage: Storage,
    days: int = 1,
    flat_threshold: float = 0.005,
) -> Optional[AccuracyReport]:
    """
    Compute aggregate accuracy metrics from scored predictions.

    Returns an AccuracyReport, or None if no scored data is available.
    """
    df = await storage.get_scored_predictions(days=days)
    if df.empty:
        logger.info("No scored predictions in the last %d days", days)
        return None

    report = AccuracyReport(
        run_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        total_scored=len(df),
        flat_threshold_used=flat_threshold,
    )

    # Magnitude MAE
    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
    df["actual_magnitude"] = pd.to_numeric(df["actual_magnitude"], errors="coerce")
    valid_mag = df.dropna(subset=["magnitude", "actual_magnitude"])
    if not valid_mag.empty:
        report.magnitude_mae = float(
            (valid_mag["magnitude"] - valid_mag["actual_magnitude"]).abs().mean()
        )

    # Re-classify actual_direction using the CURRENT flat_threshold so that
    # metrics are consistent even if the threshold was different at scoring time.
    df["actual_dir_now"] = df["actual_magnitude"].apply(
        lambda m: classify_direction(float(m), flat_threshold)
        if pd.notna(m) else None
    )
    df["correct_now"] = df["direction"] == df["actual_dir_now"]

    valid = df.dropna(subset=["actual_dir_now"])

    # Overall direction accuracy (re-evaluated against current threshold)
    if not valid.empty:
        report.direction_accuracy = float(valid["correct_now"].mean())
    else:
        report.direction_accuracy = 0.0

    # Per-class precision and recall
    for label, attr_prec, attr_rec in [
        ("UP", "up_precision", "up_recall"),
        ("DOWN", "down_precision", "down_recall"),
        ("FLAT", "flat_precision", "flat_recall"),
    ]:
        predicted_as = valid[valid["direction"] == label]
        actually_is = valid[valid["actual_dir_now"] == label]

        if len(predicted_as) > 0:
            precision = float(
                (predicted_as["actual_dir_now"] == label).sum() / len(predicted_as)
            )
        else:
            precision = 0.0

        if len(actually_is) > 0:
            recall = float(
                (actually_is["direction"] == label).sum() / len(actually_is)
            )
        else:
            recall = 0.0

        setattr(report, attr_prec, precision)
        setattr(report, attr_rec, recall)

    # Per-symbol accuracy
    sym_acc = df.groupby("symbol")["was_correct"].agg(["mean", "count"])
    sym_acc = sym_acc[sym_acc["count"] >= 3]  # need at least 3 predictions
    report.per_symbol_accuracy = sym_acc["mean"].to_dict()

    if not sym_acc.empty:
        sorted_syms = sym_acc["mean"].sort_values(ascending=False)
        report.top_symbols = [
            (sym, float(acc)) for sym, acc in sorted_syms.head(5).items()
        ]
        report.worst_symbols = [
            (sym, float(acc)) for sym, acc in sorted_syms.tail(5).items()
        ]

    logger.info(
        "Accuracy report: %d preds, %.1f%% direction accuracy, %.4f magnitude MAE",
        report.total_scored,
        report.direction_accuracy * 100,
        report.magnitude_mae,
    )

    return report
