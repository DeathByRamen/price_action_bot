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

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.storage import Storage
from src.features.indicators import compute_indicators, get_feature_columns, normalize_features
from src.model.dataset import walk_forward_split
from src.model.trainer import Trainer, DEFAULT_CHECKPOINT_DIR


async def load_training_data(
    db_path: str | None,
    min_candles: int = 200,
    rolling_days: int | None = None,
) -> pd.DataFrame:
    """
    Load and concatenate candles from all symbols with enough data.

    Parameters
    ----------
    db_path : str | None
        Path to the SQLite database.
    min_candles : int
        Minimum candles required per symbol after indicator computation.
    rolling_days : int | None
        If set, only use the most recent N days of data per symbol.
        This keeps the model trained on current market regimes.
    """
    max_candles_per_sym = 10_000
    if rolling_days is not None:
        max_candles_per_sym = rolling_days * 24 + 100  # extra for indicator warm-up

    async with Storage(db_path) as storage:
        symbols = await storage.get_all_symbols()
        logging.info("Found %d symbols in database", len(symbols))

        frames = []
        for sym in symbols:
            df = await storage.get_candles(sym, limit=max_candles_per_sym)
            if len(df) < min_candles:
                logging.debug("Skipping %s: only %d candles", sym, len(df))
                continue

            # If rolling window, filter by timestamp
            if rolling_days is not None and "ts" in df.columns:
                try:
                    cutoff = datetime.now(timezone.utc) - timedelta(days=rolling_days)
                    cutoff_str = cutoff.isoformat()
                    df = df[df["ts"] >= cutoff_str].reset_index(drop=True)
                except Exception:
                    pass  # if timestamp parsing fails, use all data

            df = compute_indicators(df)
            df = df.dropna().reset_index(drop=True)
            if len(df) >= min_candles:
                frames.append(df)
                logging.info("%s: %d rows after indicators", sym, len(df))

    if not frames:
        raise RuntimeError("No symbols have enough data for training")

    combined = pd.concat(frames, ignore_index=True)
    logging.info("Combined training data: %d rows from %d symbols", len(combined), len(frames))
    return combined


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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.rolling_days:
        logging.info("Rolling window: using last %d days of data per symbol", args.rolling_days)

    # Back up existing checkpoint if requested
    if args.backup:
        backup_checkpoint()

    # Load data
    combined = asyncio.run(
        load_training_data(args.db, rolling_days=args.rolling_days)
    )

    feature_cols = get_feature_columns()

    # Normalize features in-place for training
    for col in feature_cols:
        mean = combined[col].mean()
        std = combined[col].std()
        if std > 0:
            combined[col] = (combined[col] - mean) / std
        else:
            combined[col] = 0.0

    # Walk-forward splits
    splits = walk_forward_split(
        combined,
        n_splits=args.folds,
        window_size=args.window,
        horizon=args.horizon,
        feature_cols=feature_cols,
    )

    if not splits:
        logging.error("No valid train/val splits. Need more data.")
        sys.exit(1)

    # Train on last fold (most data) -- use earlier folds for hyperparameter insight
    train_ds, val_ds = splits[-1]
    logging.info(
        "Training on final fold: %d train samples, %d val samples",
        len(train_ds),
        len(val_ds),
    )

    trainer = Trainer(
        num_features=len(feature_cols),
        hidden_dim=args.hidden,
        num_layers=2,
        dropout=0.3,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
    )

    history = trainer.fit(train_ds, val_ds, tag="final")
    final_path = trainer.save_final()

    logging.info("Training complete. Best model saved to: %s", final_path)
    logging.info(
        "Final metrics: val_loss=%.4f  val_acc=%.2f%%  val_mae=%.4f",
        history["val_loss"][-1],
        history["val_cls_acc"][-1] * 100,
        history["val_reg_mae"][-1],
    )


if __name__ == "__main__":
    main()
