#!/usr/bin/env python3
"""
Weekly hyperparameter optimization using Optuna.

Runs 50 trials per model family (LSTM, TFT, GBM) and saves
the best hyperparameters to data/hpo_results.json.
daily_retrain.py reads from this file instead of hardcoded params.

Cron (weekly, Sunday 02:00 UTC):
    0 2 * * 0 cd /path/to/pa_bot && python scripts/weekly_hpo.py
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_model import load_training_data

from src.features.indicators import get_feature_columns
from src.model.dataset import walk_forward_split
from src.model.hpo import (
    optimize_gbm_hyperparams,
    optimize_lstm_hyperparams,
    optimize_tft_hyperparams,
)
from src.model.trainer import Trainer

HPO_RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "hpo_results.json"
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def make_lstm_train_fn(symbol_data, feature_cols, flat_threshold):
    """Create a training function for LSTM HPO."""
    def train_fn(params):
        import torch
        splits = walk_forward_split(
            symbol_data,
            n_splits=2,
            window_size=params.get("window_size", 168),
            horizon=1,
            feature_cols=feature_cols,
            flat_threshold=flat_threshold,
        )
        if not splits:
            return -1.0

        train_ds, val_ds = splits[-1]
        if len(train_ds) < 100 or len(val_ds) < 50:
            return -1.0

        label_counts = train_ds.get_label_counts()
        class_weights = None
        if label_counts.min() > 0:
            inv_freq = 1.0 / label_counts.astype(np.float64)
            normed = (inv_freq / inv_freq.sum()) * 3
            class_weights = torch.tensor(normed, dtype=torch.float32).clamp(max=3.0)

        trainer = Trainer(
            num_features=len(feature_cols),
            hidden_dim=params.get("hidden_dim", 128),
            num_layers=params.get("num_layers", 2),
            dropout=params.get("dropout", 0.3),
            lr=params.get("learning_rate", 1e-3),
            batch_size=params.get("batch_size", 64),
            max_epochs=20,
            patience=5,
            class_weights=class_weights,
        )
        trainer.fit(train_ds, val_ds, tag="hpo")
        metrics = trainer.evaluate(val_ds)
        return metrics.get("sharpe", -metrics.get("loss", 1.0))

    return train_fn


def make_tft_train_fn(symbol_data, feature_cols, flat_threshold):
    """Create a training function for TFT HPO."""
    def train_fn(params):
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        from src.model.tft import TemporalFusionTransformer

        splits = walk_forward_split(
            symbol_data, n_splits=2, window_size=168, horizon=1,
            feature_cols=feature_cols, flat_threshold=flat_threshold,
        )
        if not splits:
            return -1.0

        train_ds, val_ds = splits[-1]
        if len(train_ds) < 100:
            return -1.0

        label_counts = train_ds.get_label_counts()
        class_weights = None
        if label_counts.min() > 0:
            inv_freq = 1.0 / label_counts.astype(np.float64)
            normed = (inv_freq / inv_freq.sum()) * 3
            class_weights = torch.tensor(normed, dtype=torch.float32).clamp(max=3.0)

        trainer = Trainer(
            num_features=len(feature_cols), hidden_dim=params.get("d_model", 64),
            num_layers=1, dropout=params.get("dropout", 0.1),
            lr=params.get("learning_rate", 1e-3),
            batch_size=params.get("batch_size", 64),
            max_epochs=15, patience=5, class_weights=class_weights,
        )
        tft = TemporalFusionTransformer(
            num_features=len(feature_cols),
            d_model=params.get("d_model", 64),
            num_heads=params.get("num_heads", 4),
            num_lstm_layers=params.get("num_lstm_layers", 1),
            dropout=params.get("dropout", 0.1),
        ).to(trainer.device)
        trainer.model = tft
        trainer.optimizer = AdamW(tft.parameters(), lr=params.get("learning_rate", 1e-3))
        trainer.scheduler = ReduceLROnPlateau(trainer.optimizer, mode="min", factor=0.5, patience=3)
        trainer.fit(train_ds, val_ds, tag="hpo_tft")
        metrics = trainer.evaluate(val_ds)
        return metrics.get("sharpe", -metrics.get("loss", 1.0))

    return train_fn


def make_gbm_train_fn(symbol_data, feature_cols, flat_threshold):
    """Create a training function for GBM HPO."""
    def train_fn(params):
        from src.model.gbm import GBMConfig, GBMPredictor

        splits = walk_forward_split(
            symbol_data, n_splits=2, window_size=168, horizon=1,
            feature_cols=feature_cols, flat_threshold=flat_threshold,
        )
        if not splits:
            return -1.0

        train_ds, val_ds = splits[-1]
        cfg = GBMConfig(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
        )
        gbm = GBMPredictor(cfg)
        gbm.fit(train_ds.features, train_ds.labels, train_ds.magnitudes,
                val_ds.features, val_ds.labels, val_ds.magnitudes)
        probs, preds, mags = gbm.predict(val_ds.features)
        correct = (preds == val_ds.labels).mean()
        return float(correct)

    return train_fn


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    start = datetime.now(timezone.utc)
    logging.info("Weekly HPO starting at %s", start.isoformat())

    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
    config = load_config(config_path)
    scoring_cfg = config.get("scoring", {})
    flat_threshold = scoring_cfg.get("current_flat_threshold", 0.005)
    rolling_days = config.get("retrain", {}).get("rolling_days", 60)

    feature_cols = get_feature_columns()
    symbol_data = await load_training_data(None, rolling_days=rolling_days, interval="60")

    results = {}

    logging.info("Optimizing LSTM hyperparameters...")
    lstm_fn = make_lstm_train_fn(symbol_data, feature_cols, flat_threshold)
    results["lstm"] = optimize_lstm_hyperparams(lstm_fn, n_trials=50)

    logging.info("Optimizing TFT hyperparameters...")
    tft_fn = make_tft_train_fn(symbol_data, feature_cols, flat_threshold)
    results["tft"] = optimize_tft_hyperparams(tft_fn, n_trials=50)

    logging.info("Optimizing GBM hyperparameters...")
    gbm_fn = make_gbm_train_fn(symbol_data, feature_cols, flat_threshold)
    results["gbm"] = optimize_gbm_hyperparams(gbm_fn, n_trials=50)

    results["_meta"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rolling_days": rolling_days,
        "n_symbols": len(symbol_data),
    }

    os.makedirs(os.path.dirname(HPO_RESULTS_PATH), exist_ok=True)
    with open(HPO_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logging.info("HPO complete in %.0f seconds. Results saved to %s", elapsed, HPO_RESULTS_PATH)
    for model_name, params in results.items():
        if model_name != "_meta":
            logging.info("  %s: %s", model_name, params)


if __name__ == "__main__":
    asyncio.run(main())
