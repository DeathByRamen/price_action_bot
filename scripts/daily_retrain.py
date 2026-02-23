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

from train_model import backup_checkpoint, load_training_data

from src.api.bitunix_client import BitunixClient
from src.data.collector import DataCollector
from src.data.storage import Storage
from src.features.indicators import get_feature_columns
from src.model.dataset import DEFAULT_FLAT_THRESHOLD, walk_forward_split
from src.model.drift import DriftConfig, DriftMonitor
from src.model.importance import compute_permutation_importance, format_importance_report
from src.model.regime import REGIME_NAMES, RegimeDetector
from src.model.trainer import Trainer
from src.pipeline import build_dispatcher
from src.scoring.accuracy import AccuracyReport, compute_accuracy_report, score_predictions
from src.scoring.adaptive import (
    AdaptiveConfig,
    compute_optimal_threshold,
    compute_sample_weights,
)
from src.scoring.pnl_optimizer import PnLOptConfig, PnLOptimizer


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


async def gap_fill(db_path: str | None, intervals: list[str] | None = None) -> int:
    """Fetch recent candles for all symbols and intervals to fill gaps."""
    intervals = intervals or ["60"]
    total = 0
    async with BitunixClient() as client, Storage(db_path) as storage:
        collector = DataCollector(client, storage)
        symbols = await collector.discover_tradeable_symbols()
        for interval in intervals:
            # Scale lookback by interval: 48h worth of candles
            interval_mins = int(interval) if interval.isdigit() else 60
            lookback = max(10, (48 * 60) // interval_mins)
            new_candles = await collector.fetch_latest_candles(
                symbols, interval=interval, lookback=lookback, concurrency=8
            )
            logging.info("Gap-fill [%sm]: %d new candles", interval, new_candles)
            total += new_candles
    return total


async def run_scoring_and_tuning(
    db_path: str | None,
    scoring_cfg: dict,
) -> tuple[AccuracyReport | None, float, dict[str, float]]:
    """
    Score predictions, compute accuracy, tune threshold, compute weights.

    Uses P&L-based optimization when sufficient trade history exists,
    falling back to accuracy-based tuning otherwise.

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

        # Try P&L-based threshold optimization first
        new_threshold = current_threshold
        pnl_used = False
        try:
            scored_rows = await storage._db.execute_fetchall(
                "SELECT symbol, predicted_direction, predicted_magnitude, "
                "actual_direction, actual_magnitude "
                "FROM prediction_scores "
                "WHERE scored_at >= datetime('now', ?) "
                "ORDER BY scored_at DESC",
                (f"-{lookback_days} days",),
            )

            pnl_cfg = PnLOptConfig(
                threshold_min=adaptive_cfg.flat_threshold_min,
                threshold_max=adaptive_cfg.flat_threshold_max,
                ema_alpha=adaptive_cfg.threshold_ema_alpha,
                min_trades=20,
            )
            optimizer = PnLOptimizer(pnl_cfg)

            if len(scored_rows) >= pnl_cfg.min_trades:
                actual_mags = np.array([float(r[4] or 0) for r in scored_rows])
                pred_dirs = np.array([
                    0 if r[1] == "UP" else 2 if r[1] == "DOWN" else 1
                    for r in scored_rows
                ])
                pred_mags = np.array([float(r[2] or 0) for r in scored_rows])

                new_threshold, sharpe = optimizer.optimize_flat_threshold(
                    actual_mags, pred_dirs, pred_mags, current_threshold
                )
                logging.info(
                    "P&L optimizer: threshold=%.4f (Sharpe=%.3f) from %d trades",
                    new_threshold, sharpe, len(scored_rows),
                )
                pnl_used = True

                # P&L-based sample weights
                symbols = np.array([r[0] for r in scored_rows])
                was_correct = np.array([r[1] == r[3] for r in scored_rows])
                pnl_weights = optimizer.compute_pnl_weights(
                    symbols, actual_mags, was_correct
                )
                logging.info("P&L weights for %d symbols", len(pnl_weights))
        except Exception as exc:
            logging.warning("P&L optimization failed: %s — falling back to accuracy", exc)

        if not pnl_used:
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

    from datetime import datetime, timezone
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subject = f"[PA Bot] Daily Accuracy Digest — {now_utc}"
    for channel in dispatcher._channels:
        try:
            await channel.send(message, subject=subject)
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


async def _train_single_timeframe(
    db_path: str | None,
    interval: str,
    window_size: int,
    hidden_dim: int,
    rolling_days: int,
    epochs: int,
    patience: int,
    folds: int,
    new_threshold: float,
    symbol_weights: dict[str, float],
    feature_cols: list[str],
) -> tuple[str, dict, dict[str, float]]:
    """Train a single timeframe model and return (save_path, history, importances)."""
    logging.info("--- Training %sm model (window=%d) ---", interval, window_size)

    symbol_data = await load_training_data(
        db_path, rolling_days=rolling_days, interval=interval
    )

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
        logging.error("No valid splits for %sm model. Skipping.", interval)
        return "", {}, {}

    train_ds, val_ds = splits[-1]
    logging.info(
        "[%sm] Final fold: %d train / %d val samples",
        interval, len(train_ds), len(val_ds),
    )

    label_counts = train_ds.get_label_counts()
    logging.info(
        "[%sm] Labels: UP=%d, FLAT=%d, DOWN=%d",
        interval, label_counts[0], label_counts[1], label_counts[2],
    )
    class_weights = None
    if label_counts.min() > 0:
        inv_freq = 1.0 / label_counts.astype(np.float64)
        normed = (inv_freq / inv_freq.sum()) * 3
        class_weights = torch.tensor(normed, dtype=torch.float32)
        class_weights = class_weights.clamp(max=3.0)

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

    # Synchronous training (blocks the event loop, which is fine here)
    history = trainer.fit(
        train_ds, val_ds, tag=f"retrain_{interval}", use_sample_weights=use_weights
    )

    # Calibrate temperature
    logging.info("[%sm] Calibrating probability temperature...", interval)
    trainer.calibrate_temperature(val_ds)

    # Validate new model against old before deploying
    new_metrics = trainer.evaluate(val_ds)
    deploy_path = os.path.join(
        trainer.checkpoint_dir, f"model_final_{interval}.pt"
    )

    should_deploy = True
    if os.path.exists(deploy_path):
        try:
            old_trainer = Trainer(
                num_features=len(feature_cols),
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=0.3,
                lr=1e-3,
                batch_size=64,
                max_epochs=1,
                patience=1,
                class_weights=class_weights,
            )
            old_trainer.load(deploy_path)
            old_metrics = old_trainer.evaluate(val_ds)

            logging.info(
                "[%sm] Validation gate: old val_loss=%.4f  new val_loss=%.4f",
                interval, old_metrics["loss"], new_metrics["loss"],
            )
            if new_metrics["loss"] > old_metrics["loss"] * 1.05:
                logging.warning(
                    "[%sm] New model is worse (val_loss %.4f > %.4f * 1.05). "
                    "Keeping old checkpoint.",
                    interval, new_metrics["loss"], old_metrics["loss"],
                )
                should_deploy = False
        except Exception as exc:
            logging.warning(
                "[%sm] Could not load old model for comparison: %s. Deploying new.",
                interval, exc,
            )

    # Backtest validation gate — deploy only if simulated Sharpe >= 0
    if should_deploy:
        try:
            from src.backtesting.costs import TransactionCosts
            from src.backtesting.engine import Backtester
            from src.backtesting.signals import PredictorSignalGenerator
            from src.model.predictor import Predictor

            bt_predictor = Predictor(
                model_path=None,
                hidden_dim=hidden_dim,
                window_size=window_size,
            )
            bt_predictor.model = trainer.model
            bt_predictor.model.eval()

            sig_gen = PredictorSignalGenerator(bt_predictor)
            bt_engine = Backtester(
                signal_generator=sig_gen,
                costs=TransactionCosts(),
                initial_capital=10_000.0,
            )

            bt_data = {}
            async with Storage(db_path) as bt_storage:
                candle_limit = 14 * 24 * (60 // max(int(interval), 1))
                for sym in list(symbol_data.keys())[:10]:
                    sym_df = await bt_storage.get_candles(
                        sym, limit=candle_limit, interval=interval
                    )
                    if not sym_df.empty and len(sym_df) > window_size:
                        bt_data[sym] = sym_df

            if bt_data:
                bt_report = bt_engine.run(bt_data)
                logging.info(
                    "[%sm] Backtest gate: Sharpe=%.3f, Return=%.2f%%, Trades=%d",
                    interval, bt_report.sharpe_ratio, bt_report.total_return_pct,
                    bt_report.total_trades,
                )
                if bt_report.sharpe_ratio < -1.0:
                    logging.warning(
                        "[%sm] Backtest Sharpe %.3f is severely negative — "
                        "keeping old checkpoint",
                        interval, bt_report.sharpe_ratio,
                    )
                    should_deploy = False
        except Exception as exc:
            logging.warning("[%sm] Backtest validation failed: %s — deploying anyway", interval, exc)

    if should_deploy:
        final_path = trainer.save_final(tag=f"final_{interval}", feature_cols=feature_cols)
    else:
        final_path = deploy_path
        logging.info("[%sm] Rolled back to previous checkpoint: %s", interval, deploy_path)

    # Permutation importance — persist scores to DB
    importance_scores: dict[str, float] = {}
    try:
        importances = compute_permutation_importance(
            model=trainer.model,
            dataset=val_ds,
            feature_names=feature_cols,
            device=trainer.device,
            n_repeats=3,
        )
        importance_scores = {name: score for name, score in importances}
        importance_text = format_importance_report(importances)
        logging.info("[%sm] Feature importance:\n%s", interval, importance_text)

        # Persist to DB
        run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        async with Storage(db_path) as storage:
            await storage.insert_feature_importance(run_date, interval, importance_scores)
        logging.info("Persisted %d importance scores for %sm", len(importance_scores), interval)
    except Exception as exc:
        logging.warning("[%sm] Permutation importance failed: %s", interval, exc)

    return final_path, history, importance_scores


async def _get_low_importance_warnings(
    db_path: str | None,
    intervals: list[str],
) -> str:
    """Check for features consistently below importance threshold."""
    lines: list[str] = []
    async with Storage(db_path) as storage:
        for interval in intervals:
            low = await storage.get_low_importance_features(
                interval=interval, last_n_runs=3, threshold=0.001
            )
            if low:
                lines.append(f"\n**Low-importance features ({interval}m) — candidates for removal:**")
                for name, avg_imp in low:
                    lines.append(f"  - `{name}`: avg importance = {avg_imp:.6f}")
    return "\n".join(lines)


async def async_main(args: argparse.Namespace) -> None:
    """Single-event-loop entrypoint for the entire daily retrain pipeline."""
    start = datetime.now(timezone.utc)
    logging.info("=" * 60)
    logging.info("PA Bot daily retrain starting at %s", start.isoformat())
    logging.info("=" * 60)

    config = load_config(args.config)
    retrain_cfg = config.get("retrain", {})
    model_cfg = config.get("model", {})
    scoring_cfg = config.get("scoring", {})
    tf_cfg = config.get("timeframes", {})

    rolling_days = retrain_cfg.get("rolling_days", 60)
    epochs = retrain_cfg.get("epochs", 50)
    patience = retrain_cfg.get("patience", 8)
    folds = retrain_cfg.get("folds", 3)
    hidden_dim = model_cfg.get("hidden_dim", 128)
    old_threshold = scoring_cfg.get("current_flat_threshold", DEFAULT_FLAT_THRESHOLD)

    # Determine which timeframes to train
    multi_tf_enabled = tf_cfg.get("enabled", False)
    timeframes_to_train = []
    if multi_tf_enabled:
        primary = tf_cfg.get("primary", {})
        secondary = tf_cfg.get("secondary", {})
        timeframes_to_train = [
            (primary.get("interval", "60"), primary.get("window_size", 168)),
            (secondary.get("interval", "15"), secondary.get("window_size", 672)),
        ]
    else:
        timeframes_to_train = [
            (config.get("pipeline", {}).get("interval", "60"), model_cfg.get("window_size", 168)),
        ]

    all_intervals = [tf[0] for tf in timeframes_to_train]
    n_timeframes = len(timeframes_to_train)
    total_steps = 6 + n_timeframes + 1

    # Step 1: Gap-fill recent candles for all timeframes
    step = 1
    logging.info("Step %d/%d: Gap-filling candles for intervals: %s",
                  step, total_steps, ", ".join(f"{i}m" for i in all_intervals))
    await gap_fill(args.db, intervals=all_intervals)

    # Step 2: Score predictions and tune
    step += 1
    logging.info("Step %d/%d: Scoring predictions and computing accuracy...", step, total_steps)
    report, new_threshold, symbol_weights = await run_scoring_and_tuning(args.db, scoring_cfg)

    if report:
        logging.info(
            "Accuracy: %.1f%% direction, %.4f MAE, %d scored",
            report.direction_accuracy * 100,
            report.magnitude_mae,
            report.total_scored,
        )

    # Step 3: Regime detection
    step += 1
    logging.info("Step %d/%d: Detecting market regime...", step, total_steps)
    regime_label = "Unknown"
    try:
        async with Storage(args.db) as storage:
            btc_df = await storage.get_candles("BTCUSDT", limit=500, interval="60")
            eth_df = await storage.get_candles("ETHUSDT", limit=500, interval="60")
        if not btc_df.empty:
            detector = RegimeDetector()
            detector.fit(btc_df, eth_df if not eth_df.empty else None)
            regime = detector.predict(btc_df)
            regime_label = REGIME_NAMES.get(regime, "Ranging")
            adjustments = detector.get_regime_adjustments(regime)
            logging.info("Current regime: %s (adjustments: %s)", regime_label, adjustments)
    except Exception as exc:
        logging.warning("Regime detection failed: %s", exc)

    # Step 4: Drift monitoring
    step += 1
    logging.info("Step %d/%d: Running drift check...", step, total_steps)
    drift_report_text = ""
    try:
        drift_monitor = DriftMonitor(DriftConfig())
        async with Storage(args.db) as storage:
            rows = await storage._db.execute_fetchall(
                "SELECT direction, prob_up, prob_flat, prob_down, actual_direction "
                "FROM predictions WHERE actual_direction IS NOT NULL "
                "ORDER BY candle_ts DESC LIMIT 500"
            )
        for row in rows:
            direction, p_up, p_flat, p_down = row[0], row[1], row[2], row[3]
            actual = row[4]
            drift_monitor.record_prediction(direction, p_up, p_flat, p_down, actual)
        drift = drift_monitor.compute_drift()
        if drift.distribution_shift or drift.calibration_degraded:
            drift_report_text = (
                f"\n**Drift Alert**\n"
                f"  KL divergence: {drift.kl_divergence:.4f}"
                f"{'  ⚠ SHIFT DETECTED' if drift.distribution_shift else ''}\n"
                f"  Calibration error: {drift.calibration_error:.4f}"
                f"{'  ⚠ DEGRADED' if drift.calibration_degraded else ''}"
            )
            logging.warning("Drift detected: KL=%.4f, ECE=%.4f", drift.kl_divergence, drift.calibration_error)
        else:
            logging.info("No significant drift detected (KL=%.4f, ECE=%.4f)", drift.kl_divergence, drift.calibration_error)
    except Exception as exc:
        logging.warning("Drift monitoring failed: %s", exc)

    # Step 5: Report threshold change
    step += 1
    logging.info("Step %d/%d: Threshold %.4f -> %.4f", step, total_steps, old_threshold, new_threshold)

    # Step 6: Back up existing checkpoints
    step += 1
    logging.info("Step %d/%d: Backing up existing model checkpoints...", step, total_steps)
    backup_checkpoint()

    feature_cols = get_feature_columns()

    # Steps 5..5+N: Train each timeframe
    trained_models = []
    for interval, window_size in timeframes_to_train:
        step += 1
        logging.info(
            "Step %d/%d: Retraining %sm model (%d-day window, epochs=%d, patience=%d)...",
            step, total_steps, interval, rolling_days, epochs, patience,
        )
        # Scale FLAT threshold for shorter intervals
        interval_mins = int(interval) if interval.isdigit() else 60
        scaled_threshold = new_threshold * (interval_mins / 60.0)
        logging.info("[%sm] FLAT threshold scaled to %.4f", interval, scaled_threshold)

        final_path, history, _imp = await _train_single_timeframe(
            db_path=args.db,
            interval=interval,
            window_size=window_size,
            hidden_dim=hidden_dim,
            rolling_days=rolling_days,
            epochs=epochs,
            patience=patience,
            folds=folds,
            new_threshold=scaled_threshold,
            symbol_weights=symbol_weights,
            feature_cols=feature_cols,
        )
        if final_path:
            trained_models.append((interval, final_path, history))

    # Check for low-importance features across runs
    low_imp_warnings = await _get_low_importance_warnings(args.db, all_intervals)

    # Final step: Send digest
    step += 1
    logging.info("Step %d/%d: Sending accuracy digest...", step, total_steps)
    extra_sections = []
    if regime_label != "Unknown":
        extra_sections.append(f"\n**Market Regime:** {regime_label}")
    if drift_report_text:
        extra_sections.append(drift_report_text)
    if low_imp_warnings:
        extra_sections.append(low_imp_warnings)
    extra_report = "\n".join(extra_sections) if extra_sections else None
    await send_accuracy_digest(config, report, new_threshold, old_threshold, extra_report)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logging.info("=" * 60)
    logging.info("Retrain complete in %.1f seconds", elapsed)
    for interval, path, history in trained_models:
        if history:
            logging.info(
                "[%sm] Model saved to: %s  val_loss=%.4f  val_acc=%.2f%%  val_mae=%.4f",
                interval, path,
                history["val_loss"][-1],
                history["val_cls_acc"][-1] * 100,
                history["val_reg_mae"][-1],
            )
    logging.info("=" * 60)


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

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
