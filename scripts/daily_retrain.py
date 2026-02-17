#!/usr/bin/env python3
"""
Daily model retraining entrypoint -- designed for cron.

Workflow:
  1. Gap-fill recent candle data (last 48h)
  2. Score all un-scored predictions against actual outcomes
  3. Compute accuracy report
  4. Auto-tune FLAT_THRESHOLD and compute sample weights
  5. Back up existing model checkpoint
  6. Retrain LSTM on rolling window with adaptive weights + class balancing
  7. Calibrate probability temperature on validation set
  8. Run permutation importance for feature auditing
  9. Send accuracy + retrain digest via notifications

Usage:
    python scripts/daily_retrain.py [--config config/settings.yaml]

Cron (daily at 00:05 UTC):
    5 0 * * * cd /path/to/pa_bot && python scripts/daily_retrain.py >> logs/retrain.log 2>&1
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.bitunix_client import BitunixClient
from src.data.collector import DataCollector
from src.data.storage import Storage
from src.features.indicators import get_feature_columns
from src.model.dataset import walk_forward_split, DEFAULT_FLAT_THRESHOLD
from src.model.trainer import Trainer
from src.model.importance import compute_permutation_importance, format_importance_report
from src.pipeline import build_dispatcher
from src.scoring.accuracy import score_predictions, compute_accuracy_report, AccuracyReport
from src.scoring.adaptive import (
    compute_optimal_threshold,
    compute_sample_weights,
    AdaptiveConfig,
)

from train_model import backup_checkpoint, load_training_data


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


async def gap_fill(db_path: str | None) -> int:
    """Fetch the last 48 hours of candles for all symbols to fill any gaps."""
    async with BitunixClient() as client, Storage(db_path) as storage:
        collector = DataCollector(client, storage)
        symbols = await collector.discover_futures_symbols()
        new_candles = await collector.fetch_latest_candles(
            symbols, interval="60", lookback=48, concurrency=8
        )
        logging.info("Gap-fill complete: %d new candles", new_candles)
        return new_candles


async def run_scoring_and_tuning(
    db_path: str | None,
    scoring_cfg: dict,
) -> tuple[AccuracyReport | None, float, dict[str, float]]:
    """
    Score predictions, compute accuracy, tune threshold, compute weights.

    Returns (accuracy_report, new_flat_threshold, symbol_weights).
    """
    current_threshold = scoring_cfg.get("current_flat_threshold", DEFAULT_FLAT_THRESHOLD)
    lookback_days = scoring_cfg.get("weight_lookback_days", 7)

    adaptive_cfg = AdaptiveConfig(
        flat_threshold_min=scoring_cfg.get("flat_threshold_min", 0.002),
        flat_threshold_max=scoring_cfg.get("flat_threshold_max", 0.015),
        threshold_ema_alpha=scoring_cfg.get("threshold_ema_alpha", 0.3),
        weight_lookback_days=lookback_days,
    )

    async with Storage(db_path) as storage:
        n_scored = await score_predictions(storage, flat_threshold=current_threshold)
        logging.info("Scored %d predictions", n_scored)

        report = await compute_accuracy_report(
            storage, days=1, flat_threshold=current_threshold
        )

        if report:
            await storage.insert_accuracy_log(report.as_db_row())

        new_threshold = await compute_optimal_threshold(
            storage,
            lookback_days=lookback_days,
            current_threshold=current_threshold,
            config=adaptive_cfg,
        )

        weights = await compute_sample_weights(
            storage, lookback_days=lookback_days, config=adaptive_cfg
        )

    return report, new_threshold, weights


async def send_accuracy_digest(
    config: dict,
    report: AccuracyReport | None,
    new_threshold: float,
    old_threshold: float,
    importance_report: str | None = None,
) -> None:
    """Send accuracy report via configured notification channels."""
    dispatcher = build_dispatcher(config)
    if not dispatcher._channels:
        return

    message = format_accuracy_digest(report, new_threshold, old_threshold)
    if importance_report:
        message += "\n\n" + importance_report

    for channel in dispatcher._channels:
        try:
            await channel.send(message)
        except Exception as exc:
            logging.error("Failed to send accuracy digest via %s: %s", channel.name, exc)


def format_accuracy_digest(
    report: AccuracyReport | None,
    new_threshold: float,
    old_threshold: float,
) -> str:
    """Format the accuracy report as a notification message."""
    lines = ["**PA Bot Daily Accuracy Report**\n"]

    if report is None or report.total_scored == 0:
        lines.append("No predictions were scored in the last 24 hours.")
        lines.append("(This is normal if the system just started collecting data.)")
        return "\n".join(lines)

    lines.append("```")
    lines.append(f"Predictions scored:   {report.total_scored}")
    lines.append(f"Direction accuracy:   {report.direction_accuracy:.1%}")
    lines.append(f"  UP  prec/recall:    {report.up_precision:.0%} / {report.up_recall:.0%}")
    lines.append(f"  DOWN prec/recall:   {report.down_precision:.0%} / {report.down_recall:.0%}")
    lines.append(f"  FLAT prec/recall:   {report.flat_precision:.0%} / {report.flat_recall:.0%}")
    lines.append(f"Magnitude MAE:        {report.magnitude_mae:.4f}")
    lines.append(
        f"FLAT threshold:       {new_threshold:.4f}"
        + (f" (adjusted from {old_threshold:.4f})" if abs(new_threshold - old_threshold) > 0.0001 else "")
    )
    lines.append("```")

    if report.top_symbols:
        top_str = ", ".join(f"{s} ({a:.0%})" for s, a in report.top_symbols[:3])
        lines.append(f"\nTop performers: {top_str}")

    if report.worst_symbols:
        worst_str = ", ".join(f"{s} ({a:.0%})" for s, a in report.worst_symbols[:3])
        lines.append(f"Worst performers: {worst_str}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily model retrain (cron-friendly)")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml"),
        help="Path to settings.yaml",
    )
    parser.add_argument("--db", type=str, default=None, help="Override SQLite DB path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    start = datetime.now(timezone.utc)
    logging.info("=" * 60)
    logging.info("PA Bot daily retrain starting at %s", start.isoformat())
    logging.info("=" * 60)

    config = load_config(args.config)
    retrain_cfg = config.get("retrain", {})
    model_cfg = config.get("model", {})
    scoring_cfg = config.get("scoring", {})

    rolling_days = retrain_cfg.get("rolling_days", 60)
    epochs = retrain_cfg.get("epochs", 50)
    patience = retrain_cfg.get("patience", 8)
    folds = retrain_cfg.get("folds", 3)
    window_size = model_cfg.get("window_size", 168)
    hidden_dim = model_cfg.get("hidden_dim", 128)
    old_threshold = scoring_cfg.get("current_flat_threshold", DEFAULT_FLAT_THRESHOLD)

    # Step 1: Gap-fill recent candles
    logging.info("Step 1/8: Gap-filling recent candle data...")
    asyncio.run(gap_fill(args.db))

    # Step 2: Score predictions and tune
    logging.info("Step 2/8: Scoring predictions and computing accuracy...")
    report, new_threshold, symbol_weights = asyncio.run(
        run_scoring_and_tuning(args.db, scoring_cfg)
    )

    if report:
        logging.info(
            "Accuracy: %.1f%% direction, %.4f MAE, %d scored",
            report.direction_accuracy * 100,
            report.magnitude_mae,
            report.total_scored,
        )

    logging.info("Step 3/8: Threshold %.4f -> %.4f", old_threshold, new_threshold)

    # Step 4: Back up existing checkpoint
    logging.info("Step 4/8: Backing up existing model checkpoint...")
    backup_checkpoint()

    # Step 5: Load data and train
    logging.info(
        "Step 5/8: Retraining on %d-day rolling window (epochs=%d, patience=%d, threshold=%.4f)...",
        rolling_days, epochs, patience, new_threshold,
    )

    symbol_data = asyncio.run(
        load_training_data(args.db, rolling_days=rolling_days)
    )

    feature_cols = get_feature_columns()

    # Walk-forward splits with adaptive threshold and weights
    use_weights = bool(symbol_weights)
    splits = walk_forward_split(
        symbol_data,
        n_splits=folds,
        window_size=window_size,
        horizon=1,
        feature_cols=feature_cols,
        flat_threshold=new_threshold,
        symbol_weights=symbol_weights if use_weights else None,
    )

    if not splits:
        logging.error("No valid train/val splits. Not enough data for retrain.")
        sys.exit(1)

    train_ds, val_ds = splits[-1]
    logging.info(
        "Training on final fold: %d train / %d val samples",
        len(train_ds), len(val_ds),
    )

    # Compute class weights for balanced training
    label_counts = train_ds.get_label_counts()
    logging.info(
        "Label distribution: UP=%d, FLAT=%d, DOWN=%d",
        label_counts[0], label_counts[1], label_counts[2],
    )
    class_weights = None
    if label_counts.min() > 0:
        inv_freq = 1.0 / label_counts.astype(np.float64)
        normed = (inv_freq / inv_freq.sum()) * 3
        class_weights = torch.tensor(normed, dtype=torch.float32)

    trainer = Trainer(
        num_features=len(feature_cols),
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.3,
        lr=1e-3,
        batch_size=64,
        max_epochs=epochs,
        patience=patience,
        class_weights=class_weights,
    )

    history = trainer.fit(
        train_ds, val_ds, tag="retrain", use_sample_weights=use_weights
    )

    # Step 6: Calibrate temperature
    logging.info("Step 6/8: Calibrating probability temperature...")
    trainer.calibrate_temperature(val_ds)

    final_path = trainer.save_final()

    # Step 7: Permutation importance
    logging.info("Step 7/8: Computing permutation importance...")
    importance_text = None
    try:
        importances = compute_permutation_importance(
            model=trainer.model,
            dataset=val_ds,
            feature_names=feature_cols,
            device=trainer.device,
            n_repeats=3,
        )
        importance_text = format_importance_report(importances)
        logging.info("\n%s", importance_text)
    except Exception as exc:
        logging.warning("Permutation importance failed: %s", exc)

    # Step 8: Send digest
    logging.info("Step 8/8: Sending accuracy digest...")
    asyncio.run(
        send_accuracy_digest(config, report, new_threshold, old_threshold, importance_text)
    )

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logging.info("=" * 60)
    logging.info("Retrain complete in %.1f seconds", elapsed)
    logging.info("Model saved to: %s", final_path)
    logging.info(
        "Final metrics: val_loss=%.4f  val_acc=%.2f%%  val_mae=%.4f",
        history["val_loss"][-1],
        history["val_cls_acc"][-1] * 100,
        history["val_reg_mae"][-1],
    )
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
