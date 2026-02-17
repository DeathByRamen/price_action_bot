"""
Inference engine: loads a trained model checkpoint and generates predictions
for all symbols from their latest feature windows.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from src.features.indicators import compute_indicators, get_feature_columns
from .architecture import CryptoPredictorLSTM

logger = logging.getLogger(__name__)

DIRECTION_LABELS = {0: "UP", 1: "FLAT", 2: "DOWN"}

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "models", "model_final.pt"
)


@dataclass
class Prediction:
    symbol: str
    direction: str  # "UP", "FLAT", "DOWN"
    prob_up: float
    prob_flat: float
    prob_down: float
    magnitude: float  # predicted % change
    signal_score: float  # |max_directional_prob - 0.33| * |magnitude|
    current_price: float


class Predictor:
    """Load a trained model and run inference on prepared feature DataFrames."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_features: Optional[int] = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        window_size: int = 168,
        device: Optional[str] = None,
    ):
        self.window_size = window_size
        self.feature_cols = get_feature_columns()
        num_features = num_features or len(self.feature_cols)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CryptoPredictorLSTM(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.0,  # no dropout at inference
        ).to(self.device)

        path = model_path or DEFAULT_MODEL_PATH
        if os.path.exists(path):
            self.model.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
            logger.info("Loaded model from %s", path)
        else:
            logger.warning("No model checkpoint at %s -- predictions will be random!", path)

        self.model.eval()

    def predict_symbol(self, df: pd.DataFrame, symbol: str) -> Optional[Prediction]:
        """
        Generate a prediction for a single symbol given its OHLCV DataFrame.

        The DataFrame must have at least `window_size + 50` rows (50 for indicator
        warm-up) and columns: open, high, low, close, volume.
        """
        if len(df) < self.window_size + 50:
            logger.debug(
                "%s: insufficient data (%d rows, need %d)",
                symbol,
                len(df),
                self.window_size + 50,
            )
            return None

        # Compute indicators
        df = compute_indicators(df.copy())
        df = df.dropna().reset_index(drop=True)

        if len(df) < self.window_size:
            logger.debug("%s: insufficient data after indicator NaN drop", symbol)
            return None

        # Z-score normalize the feature window
        feature_data = df[self.feature_cols].values.astype(np.float32)
        window = feature_data[-self.window_size:]

        # Per-window normalization (same approach used during training)
        means = np.nanmean(window, axis=0, keepdims=True)
        stds = np.nanstd(window, axis=0, keepdims=True)
        stds[stds == 0] = 1.0
        window = (window - means) / stds

        # Replace any remaining NaN/inf
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

        x = torch.from_numpy(window).unsqueeze(0).to(self.device)

        with torch.no_grad():
            cls_logits, mag_pred = self.model(x)
            probs = torch.softmax(cls_logits, dim=1).squeeze(0).cpu().numpy()
            magnitude = mag_pred.squeeze().cpu().item()

        direction_idx = int(np.argmax(probs))
        direction = DIRECTION_LABELS[direction_idx]

        max_prob = float(probs[direction_idx])
        signal_score = abs(max_prob - 1.0 / 3.0) * abs(magnitude)

        current_price = float(df["close"].iloc[-1])

        return Prediction(
            symbol=symbol,
            direction=direction,
            prob_up=float(probs[0]),
            prob_flat=float(probs[1]),
            prob_down=float(probs[2]),
            magnitude=magnitude,
            signal_score=signal_score,
            current_price=current_price,
        )

    def rank_predictions(self, predictions: List[Prediction]) -> List[Prediction]:
        """Sort predictions by signal_score descending (strongest conviction first)."""
        return sorted(predictions, key=lambda p: p.signal_score, reverse=True)
