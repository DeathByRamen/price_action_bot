"""
Uncertainty quantification via MC Dropout and deep ensemble disagreement.

Provides confidence estimates for predictions to filter out low-confidence
trades and improve risk-adjusted returns.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MCDropoutEstimator:
    """
    Monte Carlo Dropout uncertainty estimation.

    Runs inference N times with dropout enabled and measures
    the variance in predictions as a proxy for model uncertainty.
    """

    def __init__(self, n_samples: int = 30):
        self.n_samples = n_samples

    def estimate(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Run MC Dropout inference and compute uncertainty.

        Parameters
        ----------
        model : nn.Module
            The prediction model (must have dropout layers).
        x : torch.Tensor
            Input tensor (batch_size=1, seq_len, num_features).

        Returns
        -------
        mean_probs : (3,) mean class probabilities
        mean_magnitude : float
        prob_uncertainty : float — std of predicted probabilities
        magnitude_uncertainty : float — std of predicted magnitude
        """
        model.train()

        all_probs = []
        all_magnitudes = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                cls_logits, mag_pred = model(x)[:2]
                probs = torch.softmax(cls_logits / model.temperature.clamp(min=0.01), dim=-1)
                all_probs.append(probs.cpu().numpy())
                all_magnitudes.append(mag_pred.cpu().numpy())

        model.eval()

        all_probs = np.array(all_probs).squeeze()     # (N, 3)
        all_mags = np.array(all_magnitudes).squeeze()  # (N,) or (N, 1)
        if all_mags.ndim > 1:
            all_mags = all_mags[:, 0]

        mean_probs = all_probs.mean(axis=0)
        mean_mag = float(all_mags.mean())

        prob_uncertainty = float(all_probs.std(axis=0).mean())
        mag_uncertainty = float(all_mags.std())

        return mean_probs, mean_mag, prob_uncertainty, mag_uncertainty


class DeepEnsembleEstimator:
    """
    Uncertainty estimation via ensemble disagreement.

    Uses M independently trained models and measures their
    disagreement as a proxy for epistemic uncertainty.
    """

    def __init__(self, models: List[nn.Module]):
        self.models = models

    def estimate(
        self,
        x: torch.Tensor,
    ) -> tuple[np.ndarray, float, float, float]:
        """
        Run all ensemble models and compute uncertainty from disagreement.

        Returns
        -------
        mean_probs : (3,) mean class probabilities
        mean_magnitude : float
        prob_uncertainty : float
        magnitude_uncertainty : float
        """
        all_probs = []
        all_mags = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                cls_logits, mag_pred = model(x)[:2]
                temp = getattr(model, "temperature", torch.ones(1))
                probs = torch.softmax(cls_logits / temp.clamp(min=0.01), dim=-1)
                all_probs.append(probs.cpu().numpy())
                all_mags.append(mag_pred.cpu().numpy())

        all_probs = np.array(all_probs).squeeze()
        all_mags = np.array(all_mags).squeeze()
        if all_mags.ndim > 1:
            all_mags = all_mags[:, 0]

        mean_probs = all_probs.mean(axis=0)
        mean_mag = float(all_mags.mean())
        prob_uncertainty = float(all_probs.std(axis=0).mean())
        mag_uncertainty = float(all_mags.std())

        return mean_probs, mean_mag, prob_uncertainty, mag_uncertainty

    def predictive_entropy(self, probs_array: np.ndarray) -> float:
        """Compute predictive entropy H[y|x] from ensemble probabilities."""
        mean_probs = probs_array.mean(axis=0)
        mean_probs = np.clip(mean_probs, 1e-10, 1.0)
        entropy = -np.sum(mean_probs * np.log(mean_probs))
        return float(entropy)

    def mutual_information(self, probs_array: np.ndarray) -> float:
        """
        Compute mutual information I[y; theta|x] (epistemic uncertainty).

        MI = H[y|x] - E[H[y|x, theta]]
        """
        pred_entropy = self.predictive_entropy(probs_array)

        expected_entropy = 0.0
        for probs in probs_array:
            probs = np.clip(probs, 1e-10, 1.0)
            expected_entropy -= np.sum(probs * np.log(probs))
        expected_entropy /= len(probs_array)

        return pred_entropy - expected_entropy
