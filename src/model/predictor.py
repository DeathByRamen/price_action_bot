"""
Inference engine: loads a trained model checkpoint and generates predictions
for all symbols from their latest feature windows.

Uses calibrated temperature for well-calibrated probabilities and
entropy-based conviction scoring for signal ranking.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from src.features.indicators import compute_indicators, get_feature_columns, MAX_WARMUP_PERIODS
from .architecture import CryptoPredictorLSTM

logger = logging.getLogger(__name__)

DIRECTION_LABELS = {0: "UP", 1: "FLAT", 2: "DOWN"}

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "models", "model_final.pt"
)


@dataclass
class Prediction:
    symbol: str
    direction: str            # "UP", "FLAT", "DOWN"
    prob_up: float
    prob_flat: float
    prob_down: float
    magnitude: float          # predicted % change
    signal_score: float       # entropy-weighted conviction * directional prob * magnitude
    conviction: float         # 1 - normalized_entropy (0 = random, 1 = certain)
    current_price: float
    feature_attention: Optional[Dict[str, float]] = field(default=None, repr=False)
    temporal_attention: Optional[np.ndarray] = field(default=None, repr=False)


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
            data = torch.load(path, map_location=self.device, weights_only=False)
            if isinstance(data, dict) and "model_state_dict" in data:
                ckpt_feats = data.get("num_features")
                ckpt_hidden = data.get("hidden_dim")
                if ckpt_feats and ckpt_feats != num_features:
                    raise RuntimeError(
                        f"Checkpoint expects {ckpt_feats} features but model "
                        f"has {num_features}. Retrain required."
                    )
                if ckpt_hidden and ckpt_hidden != hidden_dim:
                    raise RuntimeError(
                        f"Checkpoint expects hidden_dim={ckpt_hidden} but model "
                        f"has {hidden_dim}. Config mismatch."
                    )
                self.model.load_state_dict(data["model_state_dict"])
                logger.info(
                    "Loaded model from %s (temperature=%.4f, created=%s)",
                    path, self.model.temperature.item(),
                    data.get("created_at", "unknown"),
                )
            else:
                # Legacy checkpoint: plain state_dict
                self.model.load_state_dict(data)
                logger.info("Loaded model from %s (legacy, temperature=%.4f)",
                            path, self.model.temperature.item())
        else:
            logger.warning("No model checkpoint at %s -- predictions will be random!", path)

        self.model.eval()

    def predict_symbol(
        self,
        df: pd.DataFrame,
        symbol: str,
        return_attention: bool = False,
    ) -> Optional[Prediction]:
        """
        Generate a prediction for a single symbol given its OHLCV DataFrame.

        The DataFrame must have at least ``window_size + 50`` rows (50 for
        indicator warm-up) and columns: open, high, low, close, volume.
        """
        min_rows = self.window_size + MAX_WARMUP_PERIODS
        if len(df) < min_rows:
            logger.debug(
                "%s: insufficient data (%d rows, need %d)",
                symbol, len(df), min_rows,
            )
            return None

        # Compute indicators
        df = compute_indicators(df.copy())
        df = df.dropna().reset_index(drop=True)

        if len(df) < self.window_size:
            logger.debug("%s: insufficient data after indicator NaN drop", symbol)
            return None

        # Extract and normalize the feature window
        feature_data = df[self.feature_cols].values.astype(np.float32)
        window = feature_data[-self.window_size:]

        # Per-window Z-score normalization (matches training exactly)
        means = np.nanmean(window, axis=0, keepdims=True)
        stds = np.nanstd(window, axis=0, keepdims=True)
        stds[stds == 0] = 1.0
        window = (window - means) / stds
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

        if np.all(window == 0) or np.std(window) < 1e-8:
            logger.warning(
                "%s: degenerate feature window (all zeros or no variance) — skipping",
                symbol,
            )
            return None

        x = torch.from_numpy(window).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if return_attention:
                cls_logits, mag_pred, feat_w, temp_w = self.model(
                    x, return_attention=True
                )
            else:
                cls_logits, mag_pred = self.model(x)
                feat_w = temp_w = None

            # Apply calibrated temperature for well-calibrated probabilities
            temperature = self.model.temperature.clamp(min=0.01)
            scaled_logits = cls_logits / temperature
            probs = torch.softmax(scaled_logits, dim=1).squeeze(0).cpu().numpy()
            magnitude = mag_pred.squeeze().cpu().item()

        direction_idx = int(np.argmax(probs))
        direction = DIRECTION_LABELS[direction_idx]

        # Entropy-based conviction scoring
        conviction, signal_score = _compute_signal_score(probs, magnitude)

        current_price = float(df["close"].iloc[-1])

        # Attention weights (optional, for interpretability)
        feat_attention = None
        temp_attention = None
        if feat_w is not None:
            avg_feat = feat_w.squeeze(0).mean(dim=0).cpu().numpy()
            feat_attention = dict(zip(self.feature_cols, avg_feat.tolist()))
        if temp_w is not None:
            temp_attention = temp_w.squeeze(0).cpu().numpy()

        return Prediction(
            symbol=symbol,
            direction=direction,
            prob_up=float(probs[0]),
            prob_flat=float(probs[1]),
            prob_down=float(probs[2]),
            magnitude=magnitude,
            signal_score=signal_score,
            conviction=conviction,
            current_price=current_price,
            feature_attention=feat_attention,
            temporal_attention=temp_attention,
        )

    def predict_batch(
        self,
        symbol_dfs: Dict[str, pd.DataFrame],
        return_attention: bool = False,
    ) -> List[Prediction]:
        """
        Generate predictions for many symbols in a single batched forward pass.

        Parameters
        ----------
        symbol_dfs : dict[str, pd.DataFrame]
            Mapping from symbol name to its OHLCV DataFrame (raw, before indicators).
        return_attention : bool
            Whether to include attention weights in each Prediction.

        Returns
        -------
        List of Prediction objects (unranked).
        """
        min_rows = self.window_size + MAX_WARMUP_PERIODS

        # Step 1: Prepare all valid windows
        valid_symbols: List[str] = []
        windows: List[np.ndarray] = []
        prices: List[float] = []

        for symbol, df in symbol_dfs.items():
            if len(df) < min_rows:
                continue
            df_ind = compute_indicators(df.copy())
            df_ind = df_ind.dropna().reset_index(drop=True)
            if len(df_ind) < self.window_size:
                continue

            feature_data = df_ind[self.feature_cols].values.astype(np.float32)
            window = feature_data[-self.window_size:]

            # Per-window Z-score normalization
            means = np.nanmean(window, axis=0, keepdims=True)
            stds = np.nanstd(window, axis=0, keepdims=True)
            stds[stds == 0] = 1.0
            window = (window - means) / stds
            window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

            if np.all(window == 0) or np.std(window) < 1e-8:
                logger.warning(
                    "%s: degenerate feature window (batch) — skipping", symbol
                )
                continue

            valid_symbols.append(symbol)
            windows.append(window)
            prices.append(float(df_ind["close"].iloc[-1]))

        if not windows:
            return []

        # Step 2: Single batched forward pass
        batch = torch.from_numpy(np.stack(windows, axis=0)).to(self.device)

        with torch.no_grad():
            if return_attention:
                cls_logits, mag_pred, feat_w, temp_w = self.model(
                    batch, return_attention=True
                )
            else:
                cls_logits, mag_pred = self.model(batch)
                feat_w = temp_w = None

            temperature = self.model.temperature.clamp(min=0.01)
            scaled_logits = cls_logits / temperature
            all_probs = torch.softmax(scaled_logits, dim=1).cpu().numpy()
            all_mags = mag_pred.squeeze(-1).cpu().numpy()

        # Step 3: Build Prediction objects
        predictions: List[Prediction] = []
        for i, symbol in enumerate(valid_symbols):
            probs = all_probs[i]
            magnitude = float(all_mags[i])
            direction_idx = int(np.argmax(probs))
            direction = DIRECTION_LABELS[direction_idx]
            conviction, signal_score = _compute_signal_score(probs, magnitude)

            feat_attention = None
            temp_attention = None
            if feat_w is not None:
                avg_feat = feat_w[i].mean(dim=0).cpu().numpy()
                feat_attention = dict(zip(self.feature_cols, avg_feat.tolist()))
            if temp_w is not None:
                temp_attention = temp_w[i].cpu().numpy()

            predictions.append(Prediction(
                symbol=symbol,
                direction=direction,
                prob_up=float(probs[0]),
                prob_flat=float(probs[1]),
                prob_down=float(probs[2]),
                magnitude=magnitude,
                signal_score=signal_score,
                conviction=conviction,
                current_price=prices[i],
                feature_attention=feat_attention,
                temporal_attention=temp_attention,
            ))

        return predictions

    def rank_predictions(self, predictions: List[Prediction]) -> List[Prediction]:
        """Sort predictions by signal_score descending (strongest conviction first)."""
        return sorted(predictions, key=lambda p: p.signal_score, reverse=True)


def _compute_signal_score(
    probs: np.ndarray,
    magnitude: float,
) -> tuple[float, float]:
    """
    Compute conviction and signal score from probability distribution.

    Conviction is based on Shannon entropy:
      conviction = 1 - H(p) / H_max
    where H_max = log(3) for 3 classes.

    Signal score combines conviction, directional probability, and magnitude:
      score = conviction * max(prob_up, prob_down) * |magnitude|

    This replaces the old ad-hoc ``|max_prob - 0.33| * |magnitude|`` formula
    which gave near-zero scores for marginal directional leans.
    """
    eps = 1e-10
    entropy = -float(np.sum(probs * np.log(probs + eps)))
    max_entropy = math.log(3)

    conviction = max(0.0, 1.0 - (entropy / max_entropy))

    directional_prob = max(float(probs[0]), float(probs[2]))  # max of UP, DOWN
    signal_score = conviction * directional_prob * abs(magnitude)

    return conviction, signal_score
