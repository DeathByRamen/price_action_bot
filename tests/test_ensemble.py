"""Tests for src/model/ensemble.py — multi-timeframe prediction combination."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.model.ensemble import (
    _combine_probs_log_odds,
    combine_timeframes,
    format_multi_timeframe_message,
)
from src.model.predictor import Prediction


def _make_prediction(
    symbol: str = "BTCUSDT",
    direction: str = "UP",
    prob_up: float = 0.6,
    prob_flat: float = 0.2,
    prob_down: float = 0.2,
    magnitude: float = 0.02,
    conviction: float = 0.5,
    signal_score: float = 0.01,
    current_price: float = 50000.0,
) -> Prediction:
    return Prediction(
        symbol=symbol,
        direction=direction,
        prob_up=prob_up,
        prob_flat=prob_flat,
        prob_down=prob_down,
        magnitude=magnitude,
        signal_score=signal_score,
        conviction=conviction,
        current_price=current_price,
    )


class TestLogOddsCombination:
    def test_probabilities_sum_to_one(self):
        probs_a = [0.6, 0.2, 0.2]
        probs_b = [0.5, 0.3, 0.2]
        combined = _combine_probs_log_odds(probs_a, probs_b, w_a=0.6, w_b=0.4)
        assert abs(sum(combined) - 1.0) < 1e-6

    def test_agreeing_signals_boost(self):
        probs_a = [0.7, 0.1, 0.2]
        probs_b = [0.8, 0.1, 0.1]
        combined = _combine_probs_log_odds(probs_a, probs_b)
        assert combined[0] > 0.7  # combined UP should be higher than either alone

    def test_conflicting_signals(self):
        probs_a = [0.7, 0.1, 0.2]  # UP
        probs_b = [0.1, 0.1, 0.8]  # DOWN
        combined = _combine_probs_log_odds(probs_a, probs_b, w_a=0.5, w_b=0.5)
        assert combined[0] < 0.7
        assert combined[2] < 0.8

    def test_equal_weights(self):
        probs = [0.5, 0.3, 0.2]
        combined = _combine_probs_log_odds(probs, probs, w_a=0.5, w_b=0.5)
        for i in range(3):
            assert abs(combined[i] - probs[i]) < 0.05


class TestCombineTimeframes:
    def test_combines_matching_symbols(self):
        primary = [_make_prediction("BTCUSDT", "UP", 0.7, 0.1, 0.2)]
        secondary = [_make_prediction("BTCUSDT", "UP", 0.6, 0.2, 0.2)]
        result = combine_timeframes(primary, secondary)
        assert len(result) == 1
        assert result[0].symbol == "BTCUSDT"
        assert result[0].agreement in ("STRONG", "PARTIAL")

    def test_skips_unmatched_symbols(self):
        primary = [_make_prediction("BTCUSDT")]
        secondary = [_make_prediction("ETHUSDT")]
        result = combine_timeframes(primary, secondary)
        assert len(result) == 0

    def test_conflict_classification(self):
        primary = [_make_prediction("BTCUSDT", "UP", 0.7, 0.1, 0.2, magnitude=0.03)]
        secondary = [_make_prediction("BTCUSDT", "DOWN", 0.1, 0.1, 0.8, magnitude=-0.03)]
        result = combine_timeframes(primary, secondary)
        assert len(result) == 1
        assert result[0].agreement == "CONFLICT"


class TestFormatMessage:
    def test_produces_nonempty_string(self):
        primary = [_make_prediction("BTCUSDT", "UP", 0.7, 0.1, 0.2)]
        secondary = [_make_prediction("BTCUSDT", "UP", 0.6, 0.2, 0.2)]
        combined = combine_timeframes(primary, secondary)
        msg = format_multi_timeframe_message(combined)
        assert len(msg) > 50
        assert "BTCUSDT" in msg
