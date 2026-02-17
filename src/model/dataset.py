"""
Time-series dataset for the crypto predictor (quant-grade).

Key improvements over naive concatenation:
  - Per-symbol data isolation: windows never cross symbol boundaries
  - Per-window Z-score normalization: matches inference exactly
  - Temporal walk-forward splits: prevents cross-symbol data leakage
  - Per-sample weighting for adaptive training
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

    Accepts per-symbol DataFrames so that sliding windows never cross symbol
    boundaries.  Each window is Z-score normalized independently (matching
    the inference normalization in predictor.py).

    Parameters
    ----------
    symbol_data : dict[str, pd.DataFrame]
        Mapping from symbol name to its OHLCV + indicator DataFrame.
        Each DataFrame must be sorted chronologically and contain the
        columns returned by ``get_feature_columns()`` plus ``close``.
    window_size : int
        Number of timesteps in each input window (default 168 = 7 days @ 1h).
    horizon : int
        How many hours ahead to predict the price change (default 1).
    feature_cols : list[str] | None
        Which columns to include as features.
    flat_threshold : float
        Absolute % change below which a move is classified as FLAT.
    symbol_weights : dict[str, float] | None
        Per-symbol weights for adaptive training.
    min_timestamp : str | None
        If set, only include samples whose timestamp is strictly greater
        than this value.  Used to create validation splits that don't
        overlap with training data.
    """

    def __init__(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        window_size: int = 168,
        horizon: int = 1,
        feature_cols: Optional[List[str]] = None,
        flat_threshold: float = DEFAULT_FLAT_THRESHOLD,
        symbol_weights: Optional[Dict[str, float]] = None,
        min_timestamp: Optional[str] = None,
    ):
        self.window_size = window_size
        self.horizon = horizon
        self.feature_cols = feature_cols or get_feature_columns()

        # Per-symbol arrays (indexed by integer for DataLoader compatibility)
        self._sym_features: List[np.ndarray] = []
        self._sym_directions: List[np.ndarray] = []
        self._sym_magnitudes: List[np.ndarray] = []
        self._sym_names: List[str] = []

        # Flat index of valid (sym_idx, end_row) pairs
        self._entries: List[Tuple[int, int]] = []
        sample_weights: List[float] = []

        for sym_idx, symbol in enumerate(sorted(symbol_data.keys())):
            df = symbol_data[symbol]
            close = df["close"].values.astype(np.float64)

            # Labels: % change at +horizon
            future = np.roll(close, -horizon)
            pct = (future - close) / np.where(close != 0, close, 1.0)

            direction = np.full(len(close), LABEL_FLAT, dtype=np.int64)
            direction[pct > flat_threshold] = LABEL_UP
            direction[pct < -flat_threshold] = LABEL_DOWN

            features = df[self.feature_cols].values.astype(np.float32)

            self._sym_features.append(features)
            self._sym_directions.append(direction)
            self._sym_magnitudes.append(pct.astype(np.float32))
            self._sym_names.append(symbol)

            w = symbol_weights.get(symbol, 1.0) if symbol_weights else 1.0
            has_ts = "ts" in df.columns

            for end_idx in range(window_size, len(df) - horizon):
                start = end_idx - window_size

                # Skip windows containing NaN
                if np.any(np.isnan(features[start:end_idx])):
                    continue

                # Temporal filter for validation splits
                if min_timestamp and has_ts:
                    if str(df["ts"].iloc[end_idx]) <= min_timestamp:
                        continue

                self._entries.append((sym_idx, end_idx))
                sample_weights.append(w)

        self._sample_weights = np.array(sample_weights, dtype=np.float64)

        logger.info(
            "Dataset: %d samples from %d symbols (window=%d, horizon=%d, "
            "features=%d, flat_thresh=%.4f)",
            len(self._entries),
            len(self._sym_names),
            window_size,
            horizon,
            len(self.feature_cols),
            flat_threshold,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sym_idx, end_idx = self._entries[idx]
        start = end_idx - self.window_size

        window = self._sym_features[sym_idx][start:end_idx].copy()

        # Per-window Z-score normalization (matches inference in predictor.py)
        means = np.nanmean(window, axis=0, keepdims=True)
        stds = np.nanstd(window, axis=0, keepdims=True)
        stds[stds == 0] = 1.0
        window = (window - means) / stds
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

        x = torch.from_numpy(window)
        y_dir = torch.tensor(
            int(self._sym_directions[sym_idx][end_idx]), dtype=torch.long
        )
        y_mag = torch.tensor(
            float(self._sym_magnitudes[sym_idx][end_idx]), dtype=torch.float32
        )
        return x, y_dir, y_mag

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_label_counts(self) -> np.ndarray:
        """Return counts of [UP, FLAT, DOWN] labels for class balancing."""
        counts = np.zeros(3, dtype=np.int64)
        for sym_idx, end_idx in self._entries:
            label = int(self._sym_directions[sym_idx][end_idx])
            counts[label] += 1
        return counts

    def get_sampler(self) -> WeightedRandomSampler:
        """Return a WeightedRandomSampler using per-sample weights."""
        return WeightedRandomSampler(
            weights=self._sample_weights.tolist(),
            num_samples=len(self),
            replacement=True,
        )

    def get_timestamps(self, symbol_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Return the timestamp for each entry (for diagnostics)."""
        sorted_keys = sorted(symbol_data.keys())
        timestamps = []
        for sym_idx, end_idx in self._entries:
            sym = sorted_keys[sym_idx]
            df = symbol_data[sym]
            if "ts" in df.columns:
                timestamps.append(str(df["ts"].iloc[end_idx]))
            else:
                timestamps.append("")
        return timestamps


def walk_forward_split(
    symbol_data: Dict[str, pd.DataFrame],
    n_splits: int = 5,
    window_size: int = 168,
    horizon: int = 1,
    feature_cols: Optional[List[str]] = None,
    flat_threshold: float = DEFAULT_FLAT_THRESHOLD,
    symbol_weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[CryptoTimeSeriesDataset, CryptoTimeSeriesDataset]]:
    """
    Create temporal walk-forward train/validation splits.

    Splits are based on *timestamps*, not row indices, so data from all
    symbols is correctly partitioned — no future data from any symbol
    can leak into the training set.

    Each fold uses a growing training window and a fixed-size validation
    window that immediately follows it chronologically.
    """
    # Collect all unique timestamps across all symbols
    all_ts: List[str] = []
    for df in symbol_data.values():
        if "ts" in df.columns:
            all_ts.extend(df["ts"].astype(str).tolist())

    if not all_ts:
        logger.warning("No 'ts' column found — falling back to row-count splitting")
        return _fallback_row_split(
            symbol_data, n_splits, window_size, horizon,
            feature_cols, flat_threshold, symbol_weights,
        )

    all_ts = sorted(set(all_ts))
    total = len(all_ts)

    min_train_idx = int(total * 0.3)
    fold_size = max(1, (total - min_train_idx) // n_splits)

    splits: List[Tuple[CryptoTimeSeriesDataset, CryptoTimeSeriesDataset]] = []

    for i in range(n_splits):
        train_cutoff_idx = min(min_train_idx + fold_size * i, total - 1)
        val_cutoff_idx = min(train_cutoff_idx + fold_size, total - 1)

        train_cutoff_ts = all_ts[train_cutoff_idx]
        val_cutoff_ts = all_ts[val_cutoff_idx]

        # Filter symbol data up to cutoffs
        train_data: Dict[str, pd.DataFrame] = {}
        val_data: Dict[str, pd.DataFrame] = {}

        for symbol, df in symbol_data.items():
            if "ts" not in df.columns:
                continue
            ts_col = df["ts"].astype(str)

            train_df = df[ts_col <= train_cutoff_ts].copy()
            val_df = df[ts_col <= val_cutoff_ts].copy()

            min_rows = window_size + horizon + 10
            if len(train_df) >= min_rows:
                train_data[symbol] = train_df
            if len(val_df) >= min_rows:
                val_data[symbol] = val_df

        if not train_data or not val_data:
            continue

        train_ds = CryptoTimeSeriesDataset(
            train_data,
            window_size=window_size,
            horizon=horizon,
            feature_cols=feature_cols,
            flat_threshold=flat_threshold,
            symbol_weights=symbol_weights,
        )
        val_ds = CryptoTimeSeriesDataset(
            val_data,
            window_size=window_size,
            horizon=horizon,
            feature_cols=feature_cols,
            flat_threshold=flat_threshold,
            min_timestamp=train_cutoff_ts,  # only val samples *after* training data
        )

        if len(train_ds) > 0 and len(val_ds) > 0:
            splits.append((train_ds, val_ds))

    logger.info("Walk-forward: %d valid temporal splits", len(splits))
    return splits


def _fallback_row_split(
    symbol_data: Dict[str, pd.DataFrame],
    n_splits: int,
    window_size: int,
    horizon: int,
    feature_cols: Optional[List[str]],
    flat_threshold: float,
    symbol_weights: Optional[Dict[str, float]],
) -> List[Tuple[CryptoTimeSeriesDataset, CryptoTimeSeriesDataset]]:
    """Row-count fallback when timestamps are unavailable."""
    # Use a 70/30 split per symbol
    train_data: Dict[str, pd.DataFrame] = {}
    val_data: Dict[str, pd.DataFrame] = {}

    for symbol, df in symbol_data.items():
        split_idx = int(len(df) * 0.7)
        train_data[symbol] = df.iloc[:split_idx].copy()
        val_data[symbol] = df.iloc[:].copy()  # val sees all, filtered by min_timestamp=None

    train_ds = CryptoTimeSeriesDataset(
        train_data, window_size=window_size, horizon=horizon,
        feature_cols=feature_cols, flat_threshold=flat_threshold,
        symbol_weights=symbol_weights,
    )

    # For val, only use second half — approximate by row position
    val_all = CryptoTimeSeriesDataset(
        val_data, window_size=window_size, horizon=horizon,
        feature_cols=feature_cols, flat_threshold=flat_threshold,
    )
    # Filter entries to only the second half
    filtered_entries = []
    filtered_weights = []
    for j, (sym_idx, end_idx) in enumerate(val_all._entries):
        sym_name = val_all._sym_names[sym_idx]
        sym_len = len(symbol_data[sym_name])
        if end_idx >= int(sym_len * 0.7):
            filtered_entries.append((sym_idx, end_idx))
            filtered_weights.append(val_all._sample_weights[j])
    val_all._entries = filtered_entries
    val_all._sample_weights = np.array(filtered_weights, dtype=np.float64)

    if len(train_ds) > 0 and len(val_all) > 0:
        return [(train_ds, val_all)]
    return []
