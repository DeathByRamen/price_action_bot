"""
Time-series dataset for the crypto predictor.

Handles:
  - Sliding-window extraction from indicator DataFrames
  - Label generation (direction class + magnitude)
  - Train/validation splitting with walk-forward logic
  - Optional per-sample weighting for adaptive training
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.features.indicators import get_feature_columns

logger = logging.getLogger(__name__)

# Label encoding
LABEL_UP = 0
LABEL_FLAT = 1
LABEL_DOWN = 2

# Default threshold for classifying "flat" moves (absolute % change)
DEFAULT_FLAT_THRESHOLD = 0.005  # 0.5%


class CryptoTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset that yields (feature_window, direction_label, magnitude).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain feature columns (from indicators.py) plus 'close' for label
        generation.  Rows should be sorted chronologically.
    window_size : int
        Number of timesteps in each input window (default 168 = 7 days @ 1h).
    horizon : int
        How many hours ahead to predict the price change (default 1).
    feature_cols : list[str] | None
        Which columns to include as features; defaults to get_feature_columns().
    flat_threshold : float
        Absolute % change below which a move is classified as FLAT.
    symbol_weights : dict[str, float] | None
        Per-symbol weights for adaptive training.  If provided, the DataFrame
        must have a 'symbol' column.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 168,
        horizon: int = 1,
        feature_cols: list[str] | None = None,
        flat_threshold: float = DEFAULT_FLAT_THRESHOLD,
        symbol_weights: Optional[Dict[str, float]] = None,
    ):
        self.window_size = window_size
        self.horizon = horizon
        self.feature_cols = feature_cols or get_feature_columns()

        # Pre-compute labels: % change at +horizon
        close = df["close"].values.astype(np.float64)
        future = np.roll(close, -horizon)
        pct_change = (future - close) / np.where(close != 0, close, 1.0)

        # Classify direction
        direction = np.full(len(close), LABEL_FLAT, dtype=np.int64)
        direction[pct_change > flat_threshold] = LABEL_UP
        direction[pct_change < -flat_threshold] = LABEL_DOWN

        # Extract feature matrix
        features = df[self.feature_cols].values.astype(np.float32)

        # Determine valid indices (need window_size before, horizon after, no NaN)
        valid_mask = np.ones(len(df), dtype=bool)
        valid_mask[:window_size] = False
        valid_mask[-horizon:] = False

        # Drop rows with any NaN in features
        nan_rows = np.any(np.isnan(features), axis=1)
        valid_mask[nan_rows] = False

        self._indices = np.where(valid_mask)[0]
        self._features = features
        self._direction = direction
        self._magnitude = pct_change.astype(np.float32)

        # Per-sample weights (for WeightedRandomSampler)
        self._sample_weights = np.ones(len(self._indices), dtype=np.float64)
        if symbol_weights and "symbol" in df.columns:
            symbols = df["symbol"].values
            for i, idx in enumerate(self._indices):
                sym = symbols[idx]
                self._sample_weights[i] = symbol_weights.get(sym, 1.0)

        logger.info(
            "Dataset: %d valid samples (window=%d, horizon=%d, features=%d, flat_thresh=%.4f)",
            len(self._indices),
            window_size,
            horizon,
            len(self.feature_cols),
            flat_threshold,
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        end = self._indices[idx]
        start = end - self.window_size

        x = torch.from_numpy(self._features[start:end].copy())
        y_dir = torch.tensor(self._direction[end], dtype=torch.long)
        y_mag = torch.tensor(self._magnitude[end], dtype=torch.float32)

        return x, y_dir, y_mag

    def get_sampler(self) -> WeightedRandomSampler:
        """Return a WeightedRandomSampler using per-sample weights."""
        return WeightedRandomSampler(
            weights=self._sample_weights.tolist(),
            num_samples=len(self),
            replacement=True,
        )


def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_ratio: float = 0.7,
    window_size: int = 168,
    horizon: int = 1,
    feature_cols: list[str] | None = None,
    flat_threshold: float = DEFAULT_FLAT_THRESHOLD,
    symbol_weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[CryptoTimeSeriesDataset, CryptoTimeSeriesDataset]]:
    """
    Create walk-forward train/validation splits to avoid lookahead bias.

    Each fold uses a growing training window and a fixed-size validation
    window that immediately follows it chronologically.
    """
    total = len(df)
    min_train = int(total * 0.3)
    fold_size = (total - min_train) // n_splits

    splits = []
    for i in range(n_splits):
        train_end = min_train + fold_size * i
        val_end = min(train_end + fold_size, total)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[:val_end].copy()

        train_ds = CryptoTimeSeriesDataset(
            train_df,
            window_size=window_size,
            horizon=horizon,
            feature_cols=feature_cols,
            flat_threshold=flat_threshold,
            symbol_weights=symbol_weights,
        )
        val_ds = CryptoTimeSeriesDataset(
            val_df,
            window_size=window_size,
            horizon=horizon,
            feature_cols=feature_cols,
            flat_threshold=flat_threshold,
        )
        # For the validation dataset, only use indices past the training boundary
        val_indices = val_ds._indices[val_ds._indices >= train_end]
        val_ds._indices = val_indices

        if len(train_ds) > 0 and len(val_ds) > 0:
            splits.append((train_ds, val_ds))

    logger.info("Walk-forward: %d valid splits from %d total rows", len(splits), total)
    return splits
