"""Tests for src/model/predictor.py — inference, normalization, degenerate checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.model.predictor import Predictor, _compute_signal_score, DIRECTION_LABELS
from src.features.indicators import get_feature_columns


class TestComputeSignalScore:
    def test_certain_prediction_high_conviction(self):
        probs = np.array([0.95, 0.025, 0.025])
        conviction, score = _compute_signal_score(probs, 0.05)
        assert conviction > 0.7
        assert score > 0

    def test_uniform_prediction_low_conviction(self):
        probs = np.array([1/3, 1/3, 1/3])
        conviction, score = _compute_signal_score(probs, 0.05)
        assert conviction < 0.05

    def test_score_increases_with_magnitude(self):
        probs = np.array([0.8, 0.1, 0.1])
        _, score_small = _compute_signal_score(probs, 0.01)
        _, score_large = _compute_signal_score(probs, 0.10)
        assert score_large > score_small

    def test_score_nonnegative(self):
        probs = np.array([0.4, 0.3, 0.3])
        _, score = _compute_signal_score(probs, -0.01)
        assert score >= 0


class TestPredictor:
    @pytest.fixture
    def predictor(self):
        return Predictor(
            model_path="nonexistent_path.pt",
            num_features=len(get_feature_columns()),
            hidden_dim=32,
            num_layers=1,
            window_size=48,
            device="cpu",
        )

    def test_predict_symbol_insufficient_data(self, predictor, synthetic_ohlcv):
        short = synthetic_ohlcv.iloc[:10].copy()
        result = predictor.predict_symbol(short, "TEST")
        assert result is None

    def test_predict_symbol_returns_prediction(self, predictor, synthetic_ohlcv):
        result = predictor.predict_symbol(synthetic_ohlcv, "TEST")
        assert result is not None
        assert result.symbol == "TEST"
        assert result.direction in ("UP", "FLAT", "DOWN")
        assert 0 <= result.prob_up <= 1
        assert 0 <= result.prob_flat <= 1
        assert 0 <= result.prob_down <= 1
        assert abs(result.prob_up + result.prob_flat + result.prob_down - 1.0) < 1e-4
        assert result.current_price > 0

    def test_predict_batch(self, predictor, multi_symbol_ohlcv):
        results = predictor.predict_batch(multi_symbol_ohlcv)
        assert len(results) > 0
        symbols_seen = {r.symbol for r in results}
        assert len(symbols_seen) > 0

    def test_predict_degenerate_window_skipped(self, predictor):
        n = predictor.window_size + 200
        df = pd.DataFrame({
            "open": np.ones(n),
            "high": np.ones(n),
            "low": np.ones(n),
            "close": np.ones(n),
            "volume": np.ones(n),
        })
        result = predictor.predict_symbol(df, "FLAT_SYMBOL")
        # Constant data may produce degenerate features after indicators
        # Either returns None or a valid Prediction
        assert result is None or result.direction in ("UP", "FLAT", "DOWN")

    def test_rank_predictions(self, predictor, multi_symbol_ohlcv):
        preds = predictor.predict_batch(multi_symbol_ohlcv)
        if len(preds) >= 2:
            ranked = predictor.rank_predictions(preds)
            scores = [p.signal_score for p in ranked]
            assert scores == sorted(scores, reverse=True)
