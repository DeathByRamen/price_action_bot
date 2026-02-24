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

from src.features.indicators import MAX_WARMUP_PERIODS, compute_indicators, get_feature_columns

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
    uncertainty: float = 0.0  # MC Dropout uncertainty (0 = certain, higher = less certain)
    regime: str = ""          # market regime label (set by pipeline)
    feature_attention: Optional[Dict[str, float]] = field(default=None, repr=False)
    temporal_attention: Optional[np.ndarray] = field(default=None, repr=False)


class Predictor:
    """Load trained models and run inference on prepared feature DataFrames.

    Supports multi-model ensemble: loads LSTM (primary), TFT, and GBM
    when their checkpoints exist. Combines via MultiModelEnsemble with
    Sharpe-weighted averaging.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_features: Optional[int] = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        window_size: int = 168,
        device: Optional[str] = None,
        model_sharpes: Optional[Dict[str, float]] = None,
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
            dropout=0.0,
        ).to(self.device)

        path = model_path or DEFAULT_MODEL_PATH
        self._load_checkpoint(self.model, path)
        self.model.eval()

        self.tft_model = None
        self.gbm_model = None
        self.fold_models = []
        self.ensemble = None

        self._try_load_tft(path, num_features)
        self._try_load_gbm(path)
        self._try_load_fold_models(path, num_features, hidden_dim, num_layers)
        self._setup_ensemble(model_sharpes)

    def _load_checkpoint(self, model, path: str) -> bool:
        if not os.path.exists(path):
            logger.warning("No model checkpoint at %s -- predictions will be random!", path)
            return False
        data = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(data, dict) and "model_state_dict" in data:
            model.load_state_dict(data["model_state_dict"])
            logger.info("Loaded model from %s (temperature=%.4f)", path, model.temperature.item())
        else:
            model.load_state_dict(data)
            logger.info("Loaded model from %s (legacy)", path)
        return True

    def _try_load_tft(self, lstm_path: str, num_features: int) -> None:
        try:
            from .tft import TemporalFusionTransformer
            tft_path = lstm_path.replace(".pt", "_tft.pt") if lstm_path else ""
            if not tft_path or not os.path.exists(tft_path):
                base = os.path.dirname(lstm_path or DEFAULT_MODEL_PATH)
                interval = os.path.basename(lstm_path or "").replace("model_final_", "").replace(".pt", "")
                tft_path = os.path.join(base, f"model_final_{interval}_tft.pt")
            if os.path.exists(tft_path):
                self.tft_model = TemporalFusionTransformer(
                    num_features=num_features, d_model=64, num_heads=4,
                    num_lstm_layers=1, dropout=0.0,
                ).to(self.device)
                self._load_checkpoint(self.tft_model, tft_path)
                self.tft_model.eval()
                logger.info("TFT model loaded from %s", tft_path)
        except Exception as exc:
            logger.debug("TFT model not available: %s", exc)

    def _try_load_gbm(self, lstm_path: str) -> None:
        try:
            from .gbm import GBMPredictor
            base = os.path.dirname(lstm_path or DEFAULT_MODEL_PATH)
            interval = os.path.basename(lstm_path or "").replace("model_final_", "").replace(".pt", "")
            gbm_path = os.path.join(base, f"model_final_{interval}_gbm.pkl")
            if os.path.exists(gbm_path):
                self.gbm_model = GBMPredictor()
                self.gbm_model.load(gbm_path)
                logger.info("GBM model loaded from %s", gbm_path)
        except Exception as exc:
            logger.debug("GBM model not available: %s", exc)

    def _try_load_fold_models(self, lstm_path: str, num_features: int, hidden_dim: int, num_layers: int) -> None:
        """Load CV fold models for ensemble averaging."""
        self.fold_models = []
        base_dir = os.path.dirname(lstm_path or DEFAULT_MODEL_PATH)
        interval = os.path.basename(lstm_path or "").replace("model_final_", "").replace(".pt", "")
        for fold_idx in range(10):
            fp = os.path.join(base_dir, f"model_final_{interval}_fold{fold_idx}.pt")
            if not os.path.exists(fp):
                break
            try:
                m = CryptoPredictorLSTM(
                    num_features=num_features, hidden_dim=hidden_dim,
                    num_layers=num_layers, dropout=0.0,
                ).to(self.device)
                self._load_checkpoint(m, fp)
                m.eval()
                self.fold_models.append(m)
            except Exception as exc:
                logger.debug("Fold model %d failed to load: %s", fold_idx, exc)
        if self.fold_models:
            logger.info("Loaded %d CV fold models for ensemble averaging", len(self.fold_models))

    def _setup_ensemble(self, model_sharpes: Optional[Dict[str, float]]) -> None:
        from .multi_ensemble import MultiModelEnsemble
        n_models = 1 + (1 if self.tft_model else 0) + (1 if self.gbm_model else 0)
        if n_models > 1:
            self.ensemble = MultiModelEnsemble(weighting="sharpe")
            if model_sharpes:
                self.ensemble.set_model_sharpes(model_sharpes)
            logger.info("Multi-model ensemble active with %d models", n_models)

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

        df = compute_indicators(df.copy())

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        df = df.dropna(subset=["close"]).reset_index(drop=True)

        if len(df) < self.window_size:
            logger.debug("%s: insufficient data after indicator NaN drop", symbol)
            return None

        feature_data = df[self.feature_cols].fillna(0.0).values.astype(np.float32)
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

            temperature = self.model.temperature.clamp(min=0.01)
            scaled_logits = cls_logits / temperature
            probs = torch.softmax(scaled_logits, dim=1).squeeze(0).cpu().numpy()
            magnitude = mag_pred.squeeze().cpu().item()

        # CV fold averaging: average LSTM fold model outputs
        if self.fold_models:
            all_probs = [probs]
            all_mags = [magnitude]
            for fm in self.fold_models:
                fp, fm_mag = self._run_model(fm, x)
                all_probs.append(fp)
                all_mags.append(fm_mag)
            probs = np.mean(all_probs, axis=0)
            probs /= probs.sum() + 1e-10
            magnitude = float(np.mean(all_mags))

        # Ensemble: combine LSTM + TFT + GBM predictions
        if self.ensemble:
            from .multi_ensemble import ModelPrediction
            model_preds = [ModelPrediction("lstm", probs, magnitude, 0.0)]

            if self.tft_model is not None:
                tft_probs, tft_mag = self._run_model(self.tft_model, x)
                model_preds.append(ModelPrediction("tft", tft_probs, tft_mag, 0.0))

            if self.gbm_model is not None:
                gbm_probs, gbm_mag = self._run_gbm(window)
                model_preds.append(ModelPrediction("gbm", gbm_probs, gbm_mag, 0.0))

            ens = self.ensemble.combine(model_preds, symbol=symbol)
            probs = ens.class_probs
            magnitude = ens.magnitude
            # Primary uncertainty = ensemble disagreement (predictive entropy)
            uncertainty = ens.uncertainty
        elif self.fold_models:
            # Use fold disagreement as uncertainty
            all_fold_probs = [probs] + [self._run_model(fm, x)[0] for fm in self.fold_models]
            uncertainty = float(np.std(all_fold_probs, axis=0).mean())
        else:
            # Fallback: MC Dropout
            uncertainty = self._estimate_uncertainty(x)

        direction_idx = int(np.argmax(probs))
        direction = DIRECTION_LABELS[direction_idx]

        conviction, signal_score = _compute_signal_score(probs, magnitude)
        signal_score *= max(0.1, 1.0 - uncertainty)

        current_price = float(df["close"].iloc[-1])

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
            uncertainty=uncertainty,
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

            for col in self.feature_cols:
                if col not in df_ind.columns:
                    df_ind[col] = 0.0

            df_ind = df_ind.dropna(subset=["close"]).reset_index(drop=True)
            if len(df_ind) < self.window_size:
                continue

            feature_data = df_ind[self.feature_cols].fillna(0.0).values.astype(np.float32)
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

        # Step 3: MC Dropout uncertainty for the batch
        uncertainties = self._estimate_uncertainty_batch(batch, len(valid_symbols))

        # Step 4: Build Prediction objects
        predictions: List[Prediction] = []
        for i, symbol in enumerate(valid_symbols):
            probs = all_probs[i]
            magnitude = float(all_mags[i])
            direction_idx = int(np.argmax(probs))
            direction = DIRECTION_LABELS[direction_idx]
            conviction, signal_score = _compute_signal_score(probs, magnitude)
            unc = uncertainties[i]
            signal_score *= max(0.1, 1.0 - unc)

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
                uncertainty=unc,
                feature_attention=feat_attention,
                temporal_attention=temp_attention,
            ))

        return predictions

    def _run_model(self, model, x: torch.Tensor) -> tuple:
        """Run a PyTorch model and return (probs, magnitude)."""
        with torch.no_grad():
            cls_logits, mag_pred = model(x)[:2]
            temp = model.temperature.clamp(min=0.01)
            probs = torch.softmax(cls_logits / temp, dim=-1).squeeze(0).cpu().numpy()
            mag = mag_pred.squeeze().cpu().item()
        return probs, mag

    def _run_gbm(self, window: np.ndarray) -> tuple:
        """Run GBM model and return (probs, magnitude)."""
        x_in = window[np.newaxis, ...]  # (1, T, F)
        class_probs, _, mag = self.gbm_model.predict(x_in)
        return class_probs[0], float(mag[0])

    def _estimate_uncertainty(self, x: torch.Tensor, n_samples: int = 10) -> float:
        """Run MC Dropout and return mean probability std as uncertainty."""
        try:
            self.model.train()
            all_probs = []
            with torch.no_grad():
                for _ in range(n_samples):
                    cls_logits, _ = self.model(x)[:2]
                    temp = self.model.temperature.clamp(min=0.01)
                    probs = torch.softmax(cls_logits / temp, dim=-1)
                    all_probs.append(probs.cpu().numpy())
            self.model.eval()
            stacked = np.array(all_probs).squeeze()
            return float(stacked.std(axis=0).mean())
        except Exception:
            self.model.eval()
            return 0.0

    def _estimate_uncertainty_batch(
        self, batch: torch.Tensor, n_items: int, n_samples: int = 10
    ) -> List[float]:
        """Run MC Dropout for a batch and return per-item uncertainty."""
        try:
            self.model.train()
            all_probs = []
            with torch.no_grad():
                for _ in range(n_samples):
                    cls_logits, _ = self.model(batch)[:2]
                    temp = self.model.temperature.clamp(min=0.01)
                    probs = torch.softmax(cls_logits / temp, dim=-1)
                    all_probs.append(probs.cpu().numpy())
            self.model.eval()
            stacked = np.array(all_probs)  # (n_samples, n_items, 3)
            per_item_std = stacked.std(axis=0).mean(axis=1)  # (n_items,)
            return [float(v) for v in per_item_std]
        except Exception:
            self.model.eval()
            return [0.0] * n_items

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
    signal_score = conviction * directional_prob * abs(magnitude) * 100

    return conviction, signal_score
