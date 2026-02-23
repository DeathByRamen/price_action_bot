#!/usr/bin/env python3
"""
Train the LSTM predictor on historical OHLCV data stored in SQLite.

Usage:
    python scripts/train_model.py [--db data/ohlcv.db] [--epochs 100] [--window 168]
    python scripts/train_model.py --rolling-days 60   # retrain on last 60 days only
"""

import argparse
import asyncio
import logging
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.storage import Storage
from src.features.derivatives import (
    compute_coinalyze_features,
    compute_cross_asset_features,
    compute_funding_rate_features,
)
from src.features.indicators import compute_indicators, get_feature_columns
from src.features.orderbook import compute_orderbook_features
from src.model.dataset import walk_forward_split
from src.model.trainer import DEFAULT_CHECKPOINT_DIR, Trainer


async def load_training_data(
    db_path: str | None,
    min_candles: int = 200,
    rolling_days: int | None = None,
    interval: str = "60",
) -> Dict[str, pd.DataFrame]:
    """
    Load per-symbol DataFrames with computed indicators.

    Returns a dict mapping symbol name to its indicator DataFrame.
    Each symbol's data is self-contained — no cross-symbol concatenation
    that could cause window boundary or normalization issues.

    Parameters
    ----------
    db_path : str | None
        Path to the SQLite database.
    min_candles : int
        Minimum candles required per symbol after indicator computation.
    rolling_days : int | None
        If set, only use the most recent N days of data per symbol.
    interval : str
        Candle interval to load (e.g. "60", "15").
    """
    # For sub-hourly intervals, scale max candles proportionally
    interval_mins = int(interval) if interval.isdigit() else 60
    candles_per_day = (24 * 60) // interval_mins
    max_candles_per_sym = 10_000
    if rolling_days is not None:
        max_candles_per_sym = rolling_days * candles_per_day + 100

    symbol_data: Dict[str, pd.DataFrame] = {}

    async with Storage(db_path) as storage:
        symbols = await storage.get_all_symbols()
        logging.info("Found %d symbols in database", len(symbols))

        btc_df = await storage.get_candles("BTCUSDT", limit=max_candles_per_sym, interval=interval)

        for sym in symbols:
            df = await storage.get_candles(sym, limit=max_candles_per_sym, interval=interval)
            if len(df) < min_candles:
                logging.debug("Skipping %s: only %d candles (%sm)", sym, len(df), interval)
                continue

            if rolling_days is not None and "ts" in df.columns:
                try:
                    cutoff = datetime.now(timezone.utc) - timedelta(days=rolling_days)
                    cutoff_str = cutoff.isoformat()
                    df = df[df["ts"] >= cutoff_str].reset_index(drop=True)
                except Exception:
                    pass

            df = compute_indicators(df)

            start_ts = str(df["ts"].iloc[0]) if "ts" in df.columns and len(df) > 0 else None

            try:
                ob_df = await storage.get_order_book_snapshots(sym, start_ts=start_ts, limit=max_candles_per_sym)
                if not ob_df.empty:
                    df = compute_orderbook_features(ob_df, df)
            except Exception:
                pass

            try:
                oi_df = await storage.get_coinalyze_oi(sym, start_ts=start_ts, limit=max_candles_per_sym)
                liq_df = await storage.get_coinalyze_liquidations(sym, start_ts=start_ts, limit=max_candles_per_sym)
                ls_df = await storage.get_coinalyze_long_short(sym, start_ts=start_ts, limit=max_candles_per_sym)
                if not oi_df.empty or not liq_df.empty or not ls_df.empty:
                    df = compute_coinalyze_features(oi_df, liq_df, ls_df, df)
            except Exception:
                pass

            try:
                fr_df = await storage.get_funding_rate_snapshots(sym, start_ts=start_ts, limit=max_candles_per_sym)
                if not fr_df.empty:
                    df = compute_funding_rate_features(fr_df, df)
            except Exception:
                pass

            try:
                if not btc_df.empty and sym != "BTCUSDT":
                    df = compute_cross_asset_features(btc_df, df)
            except Exception:
                pass

            for col in get_feature_columns():
                if col not in df.columns:
                    df[col] = 0.0
            df = df.fillna(0.0)

            df = df.dropna(subset=["close"]).reset_index(drop=True)

            if len(df) >= min_candles:
                symbol_data[sym] = df
                logging.info("%s: %d rows after indicators (%sm)", sym, len(df), interval)

    if not symbol_data:
        raise RuntimeError(
            f"No symbols have enough {interval}m data for training"
        )

    total_rows = sum(len(df) for df in symbol_data.values())
    logging.info(
        "Loaded training data: %d total rows from %d symbols (%sm interval)",
        total_rows, len(symbol_data), interval,
    )
    return symbol_data


def backup_checkpoint(checkpoint_dir: str | None = None) -> str | None:
    """
    Create a timestamped backup of model_final.pt before overwriting.
    Returns the backup path, or None if no existing checkpoint was found.
    """
    ckpt_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    final_path = os.path.join(ckpt_dir, "model_final.pt")
    if not os.path.exists(final_path):
        return None

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(ckpt_dir, f"model_final_{ts}.pt")
    shutil.copy2(final_path, backup_path)
    logging.info("Backed up previous checkpoint to %s", backup_path)
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the crypto predictor model")
    parser.add_argument("--db", type=str, default=None, help="Path to SQLite database")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--window", type=int, default=168, help="Input window size (hours)")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon (hours)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=128, help="LSTM hidden dim")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--folds", type=int, default=3, help="Walk-forward CV folds")
    parser.add_argument(
        "--rolling-days",
        type=int,
        default=None,
        help="Only use the most recent N days of data per symbol (e.g. 60 for ~2 months). "
             "Omit to use all available data.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=False,
        help="Back up the existing model_final.pt before overwriting",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="60",
        help="Candle interval to train on: 15, 30, 60, etc. (default: 60). "
             "Checkpoint is saved as model_final_{interval}.pt",
    )
    parser.add_argument(
        "--flat-threshold",
        type=float,
        default=None,
        help="FLAT threshold for labeling (default: 0.005 for 60m, auto-scaled for shorter intervals). "
             "Moves smaller than this are labeled FLAT.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    interval = args.interval
    logging.info("Training for %sm interval", interval)

    # Auto-scale FLAT threshold for shorter intervals if not explicitly set
    if args.flat_threshold is not None:
        flat_threshold = args.flat_threshold
    else:
        interval_mins = int(interval) if interval.isdigit() else 60
        flat_threshold = 0.005 * (interval_mins / 60.0)
    logging.info("FLAT threshold: %.4f (%.2f%%)", flat_threshold, flat_threshold * 100)

    if args.rolling_days:
        logging.info("Rolling window: using last %d days of data per symbol", args.rolling_days)

    if args.backup:
        backup_checkpoint()

    # Load per-symbol data (no global concatenation)
    symbol_data = asyncio.run(
        load_training_data(args.db, rolling_days=args.rolling_days, interval=interval)
    )

    feature_cols = get_feature_columns()

    # Walk-forward splits (temporal, per-symbol)
    splits = walk_forward_split(
        symbol_data,
        n_splits=args.folds,
        window_size=args.window,
        horizon=args.horizon,
        feature_cols=feature_cols,
        flat_threshold=flat_threshold,
    )

    if not splits:
        logging.error("No valid train/val splits. Need more data.")
        sys.exit(1)

    # Train on last fold (most data)
    train_ds, val_ds = splits[-1]
    logging.info(
        "Training on final fold: %d train samples, %d val samples",
        len(train_ds), len(val_ds),
    )

    # Compute inverse-frequency class weights for balanced training
    label_counts = train_ds.get_label_counts()
    logging.info(
        "Label distribution: UP=%d, FLAT=%d, DOWN=%d",
        label_counts[0], label_counts[1], label_counts[2],
    )
    class_weights = None
    if label_counts.min() > 0:
        inv_freq = 1.0 / label_counts.astype(np.float64)
        normed = (inv_freq / inv_freq.sum()) * 3  # scale to num_classes
        class_weights = torch.tensor(normed, dtype=torch.float32).clamp(max=3.0)
        logging.info("Class weights: UP=%.3f, FLAT=%.3f, DOWN=%.3f",
                      class_weights[0], class_weights[1], class_weights[2])

    trainer = Trainer(
        num_features=len(feature_cols),
        hidden_dim=args.hidden,
        num_layers=2,
        dropout=0.3,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        class_weights=class_weights,
    )

    history = trainer.fit(train_ds, val_ds, tag="final")

    # Calibrate temperature on validation data
    logging.info("Calibrating probability temperature on validation set...")
    trainer.calibrate_temperature(val_ds)

    final_path = trainer.save_final(tag=f"final_{interval}", feature_cols=feature_cols)

    logging.info("Training complete. Best model saved to: %s", final_path)
    logging.info(
        "Final metrics: val_loss=%.4f  val_acc=%.2f%%  val_mae=%.4f",
        history["val_loss"][-1],
        history["val_cls_acc"][-1] * 100,
        history["val_reg_mae"][-1],
    )


if __name__ == "__main__":
    main()
