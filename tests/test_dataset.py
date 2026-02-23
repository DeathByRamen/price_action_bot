"""Tests for src/model/dataset.py — window isolation, labels, normalization."""

from __future__ import annotations

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.features.indicators import compute_indicators, get_feature_columns
from src.model.dataset import (
    LABEL_DOWN,
    LABEL_FLAT,
    LABEL_UP,
    CryptoTimeSeriesDataset,
)


def _prepare_symbol_data(df: pd.DataFrame) -> pd.DataFrame:
    return compute_indicators(df.copy()).dropna().reset_index(drop=True)


class TestCryptoTimeSeriesDataset:
    def test_creates_nonzero_samples(self, synthetic_ohlcv):
        prepared = _prepare_symbol_data(synthetic_ohlcv)
        ds = CryptoTimeSeriesDataset(
            {"TEST": prepared}, window_size=48, horizon=1
        )
        assert len(ds) > 0

    def test_window_shape(self, synthetic_ohlcv):
        prepared = _prepare_symbol_data(synthetic_ohlcv)
        ds = CryptoTimeSeriesDataset(
            {"TEST": prepared}, window_size=48, horizon=1
        )
        x, y_dir, y_mag = ds[0]
        assert x.shape == (48, len(get_feature_columns()))
        assert y_dir.dim() == 0  # scalar
        assert y_mag.dim() == 0  # scalar

    def test_labels_are_valid(self, synthetic_ohlcv):
        prepared = _prepare_symbol_data(synthetic_ohlcv)
        ds = CryptoTimeSeriesDataset(
            {"TEST": prepared}, window_size=48, horizon=1
        )
        for i in range(min(20, len(ds))):
            _, y_dir, _ = ds[i]
            assert y_dir.item() in (LABEL_UP, LABEL_FLAT, LABEL_DOWN)

    def test_normalization_mean_near_zero(self, synthetic_ohlcv):
        prepared = _prepare_symbol_data(synthetic_ohlcv)
        ds = CryptoTimeSeriesDataset(
            {"TEST": prepared}, window_size=48, horizon=1
        )
        x, _, _ = ds[0]
        assert abs(x.mean().item()) < 1.0, "Z-score mean should be near 0"

    def test_no_cross_symbol_leakage(self, multi_symbol_ohlcv):
        symbol_data = {}
        for sym, df in multi_symbol_ohlcv.items():
            prepared = _prepare_symbol_data(df)
            if len(prepared) >= 60:
                symbol_data[sym] = prepared

        ds = CryptoTimeSeriesDataset(symbol_data, window_size=48, horizon=1)
        assert len(ds) > 0

        sym_names = ds._sym_names
        for sym_idx, end_idx in ds._entries:
            sym_len = len(ds._sym_features[sym_idx])
            start = end_idx - ds.window_size
            assert start >= 0, f"Window starts before data for {sym_names[sym_idx]}"
            assert end_idx < sym_len, f"Window exceeds data for {sym_names[sym_idx]}"

    def test_label_counts(self, synthetic_ohlcv):
        prepared = _prepare_symbol_data(synthetic_ohlcv)
        ds = CryptoTimeSeriesDataset(
            {"TEST": prepared}, window_size=48, horizon=1
        )
        counts = ds.get_label_counts()
        assert len(counts) == 3
        assert counts.sum() == len(ds)

    def test_sampler_returns_correct_length(self, synthetic_ohlcv):
        prepared = _prepare_symbol_data(synthetic_ohlcv)
        ds = CryptoTimeSeriesDataset(
            {"TEST": prepared}, window_size=48, horizon=1
        )
        sampler = ds.get_sampler()
        assert len(list(sampler)) == len(ds)

    def test_flat_threshold_affects_labels(self, synthetic_ohlcv):
        prepared = _prepare_symbol_data(synthetic_ohlcv)
        ds_narrow = CryptoTimeSeriesDataset(
            {"TEST": prepared}, window_size=48, flat_threshold=0.001
        )
        ds_wide = CryptoTimeSeriesDataset(
            {"TEST": prepared}, window_size=48, flat_threshold=0.05
        )
        counts_narrow = ds_narrow.get_label_counts()
        counts_wide = ds_wide.get_label_counts()
        assert counts_wide[LABEL_FLAT] >= counts_narrow[LABEL_FLAT]
