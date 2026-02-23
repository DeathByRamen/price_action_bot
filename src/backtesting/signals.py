"""
Signal generation interface for backtesting.

Translates model predictions into actionable trade signals (LONG/SHORT/FLAT).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """A trading signal for a single symbol at a point in time."""
    symbol: str
    action: str          # "LONG", "SHORT", "CLOSE", "HOLD"
    strength: float      # 0 to 1, how strongly we believe in this signal
    magnitude: float     # predicted % change
    timestamp: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SignalGenerator(ABC):
    """Abstract interface for generating trade signals from market data."""

    @abstractmethod
    def generate_signals(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        timestamp: str,
    ) -> List[TradeSignal]:
        """Generate signals for all symbols at a given timestamp."""
        ...


class PredictorSignalGenerator(SignalGenerator):
    """
    Wraps the existing PA Bot Predictor to generate trade signals.

    Parameters
    ----------
    predictor : Predictor
        Trained model predictor instance.
    min_conviction : float
        Minimum conviction to generate a directional signal.
    min_prob : float
        Minimum directional probability threshold.
    min_magnitude : float
        Minimum predicted magnitude to trade.
    max_hold_candles : int
        Close position after this many candles if not stopped out.
    """

    def __init__(
        self,
        predictor,
        min_conviction: float = 0.3,
        min_prob: float = 0.45,
        min_magnitude: float = 0.002,
        max_hold_candles: int = 24,
    ):
        self.predictor = predictor
        self.min_conviction = min_conviction
        self.min_prob = min_prob
        self.min_magnitude = min_magnitude
        self.max_hold_candles = max_hold_candles

    def generate_signals(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        timestamp: str,
    ) -> List[TradeSignal]:
        predictions = self.predictor.predict_batch(symbol_data)
        signals: List[TradeSignal] = []

        for pred in predictions:
            if pred.conviction < self.min_conviction:
                continue
            if abs(pred.magnitude) < self.min_magnitude:
                continue

            if pred.direction == "UP" and pred.prob_up >= self.min_prob:
                signals.append(TradeSignal(
                    symbol=pred.symbol,
                    action="LONG",
                    strength=pred.conviction,
                    magnitude=pred.magnitude,
                    timestamp=timestamp,
                    metadata={
                        "prob_up": pred.prob_up,
                        "signal_score": pred.signal_score,
                    },
                ))
            elif pred.direction == "DOWN" and pred.prob_down >= self.min_prob:
                signals.append(TradeSignal(
                    symbol=pred.symbol,
                    action="SHORT",
                    strength=pred.conviction,
                    magnitude=pred.magnitude,
                    timestamp=timestamp,
                    metadata={
                        "prob_down": pred.prob_down,
                        "signal_score": pred.signal_score,
                    },
                ))

        signals.sort(key=lambda s: s.strength, reverse=True)
        return signals


class SimpleThresholdSignalGenerator(SignalGenerator):
    """
    A simple signal generator that uses pre-computed predictions stored
    as columns in the DataFrame. Useful for backtesting without loading
    the full model.

    Expects columns: 'pred_direction', 'pred_prob_up', 'pred_prob_down',
                     'pred_magnitude', 'pred_conviction'
    """

    def __init__(
        self,
        min_conviction: float = 0.3,
        min_prob: float = 0.45,
        min_magnitude: float = 0.002,
    ):
        self.min_conviction = min_conviction
        self.min_prob = min_prob
        self.min_magnitude = min_magnitude

    def generate_signals(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        timestamp: str,
    ) -> List[TradeSignal]:
        signals: List[TradeSignal] = []

        for symbol, df in symbol_data.items():
            row = df[df["ts"] == timestamp]
            if row.empty:
                continue
            row = row.iloc[-1]

            direction = row.get("pred_direction", "FLAT")
            conviction = float(row.get("pred_conviction", 0))
            magnitude = float(row.get("pred_magnitude", 0))
            prob_up = float(row.get("pred_prob_up", 0.33))
            prob_down = float(row.get("pred_prob_down", 0.33))

            if conviction < self.min_conviction:
                continue
            if abs(magnitude) < self.min_magnitude:
                continue

            if direction == "UP" and prob_up >= self.min_prob:
                signals.append(TradeSignal(
                    symbol=symbol, action="LONG", strength=conviction,
                    magnitude=magnitude, timestamp=timestamp,
                ))
            elif direction == "DOWN" and prob_down >= self.min_prob:
                signals.append(TradeSignal(
                    symbol=symbol, action="SHORT", strength=conviction,
                    magnitude=magnitude, timestamp=timestamp,
                ))

        signals.sort(key=lambda s: s.strength, reverse=True)
        return signals
