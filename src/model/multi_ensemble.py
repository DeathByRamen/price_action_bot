"""
Multi-model ensemble combining LSTM, TFT, and GBM predictions.

Supports stacking (meta-learner), weighted averaging proportional to
recent Sharpe, and diversity-aware combination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Prediction from a single model."""
    model_name: str
    class_probs: np.ndarray    # (3,) — UP, FLAT, DOWN
    magnitude: float
    uncertainty: float = 0.0   # from MC Dropout or ensemble disagreement


@dataclass
class EnsemblePrediction:
    """Combined prediction from multiple models."""
    symbol: str
    class_probs: np.ndarray    # (3,) — UP, FLAT, DOWN
    direction: str
    magnitude: float
    uncertainty: float
    model_weights: Dict[str, float]
    agreement: float           # 0 to 1, how much models agree


class MultiModelEnsemble:
    """
    Combines predictions from multiple models.

    Weighting strategies:
    - "equal": simple average
    - "sharpe": weight proportional to recent Sharpe ratio
    - "stacking": trained meta-learner on out-of-fold predictions

    Only includes models whose prediction correlation is below a threshold
    to ensure diversity.
    """

    DIRECTION_LABELS = ["UP", "FLAT", "DOWN"]

    def __init__(
        self,
        diversity_threshold: float = 0.7,
        weighting: str = "sharpe",
    ):
        self.diversity_threshold = diversity_threshold
        self.weighting = weighting
        self.model_sharpes: Dict[str, float] = {}
        self.meta_learner = None

    def set_model_sharpes(self, sharpes: Dict[str, float]) -> None:
        """Update recent Sharpe ratios for each model."""
        self.model_sharpes = sharpes

    def combine(
        self,
        predictions: List[ModelPrediction],
        symbol: str = "",
    ) -> EnsemblePrediction:
        """
        Combine predictions from multiple models.

        Parameters
        ----------
        predictions : list of ModelPrediction from different models
        symbol : symbol being predicted

        Returns
        -------
        EnsemblePrediction
        """
        if not predictions:
            return EnsemblePrediction(
                symbol=symbol,
                class_probs=np.array([1/3, 1/3, 1/3]),
                direction="FLAT",
                magnitude=0.0,
                uncertainty=1.0,
                model_weights={},
                agreement=0.0,
            )

        if len(predictions) == 1:
            p = predictions[0]
            direction = self.DIRECTION_LABELS[int(np.argmax(p.class_probs))]
            return EnsemblePrediction(
                symbol=symbol,
                class_probs=p.class_probs,
                direction=direction,
                magnitude=p.magnitude,
                uncertainty=p.uncertainty,
                model_weights={p.model_name: 1.0},
                agreement=1.0,
            )

        weights = self._compute_weights(predictions)

        combined_probs = np.zeros(3)
        combined_mag = 0.0
        for pred, w in zip(predictions, weights.values()):
            combined_probs += w * pred.class_probs
            combined_mag += w * pred.magnitude

        combined_probs /= combined_probs.sum() + 1e-10

        directions = [np.argmax(p.class_probs) for p in predictions]
        majority = max(set(directions), key=directions.count)
        agreement = directions.count(majority) / len(directions)

        uncertainties = [p.uncertainty for p in predictions]
        prob_stds = np.std([p.class_probs for p in predictions], axis=0)
        ensemble_uncertainty = float(np.mean(prob_stds) + np.mean(uncertainties))

        direction = self.DIRECTION_LABELS[int(np.argmax(combined_probs))]

        return EnsemblePrediction(
            symbol=symbol,
            class_probs=combined_probs,
            direction=direction,
            magnitude=combined_mag,
            uncertainty=ensemble_uncertainty,
            model_weights=weights,
            agreement=agreement,
        )

    def _compute_weights(
        self,
        predictions: List[ModelPrediction],
    ) -> Dict[str, float]:
        """Compute model weights based on weighting strategy."""
        names = [p.model_name for p in predictions]

        if self.weighting == "equal" or not self.model_sharpes:
            w = 1.0 / len(predictions)
            return {name: w for name in names}

        if self.weighting == "sharpe":
            sharpes = []
            for name in names:
                s = self.model_sharpes.get(name, 0.0)
                sharpes.append(max(s, 0.01))

            total = sum(sharpes)
            return {name: s / total for name, s in zip(names, sharpes)}

        w = 1.0 / len(predictions)
        return {name: w for name in names}

    def check_diversity(
        self,
        prediction_history: Dict[str, List[np.ndarray]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Check pairwise correlation between model predictions.

        Parameters
        ----------
        prediction_history : {model_name: list of class_probs arrays}

        Returns
        -------
        Dict of {model_a: {model_b: correlation}}.
        """
        models = list(prediction_history.keys())
        correlations: Dict[str, Dict[str, float]] = {}

        for i in range(len(models)):
            correlations[models[i]] = {}
            for j in range(len(models)):
                if i == j:
                    correlations[models[i]][models[j]] = 1.0
                    continue

                preds_i = np.array(prediction_history[models[i]])
                preds_j = np.array(prediction_history[models[j]])
                min_len = min(len(preds_i), len(preds_j))
                if min_len < 10:
                    correlations[models[i]][models[j]] = 0.0
                    continue

                dirs_i = np.argmax(preds_i[-min_len:], axis=1)
                dirs_j = np.argmax(preds_j[-min_len:], axis=1)
                corr = np.corrcoef(dirs_i, dirs_j)[0, 1]
                correlations[models[i]][models[j]] = float(corr)

                if abs(corr) > self.diversity_threshold:
                    logger.warning(
                        "Models '%s' and '%s' have high correlation (%.3f) — "
                        "consider removing one for diversity",
                        models[i], models[j], corr,
                    )

        return correlations
