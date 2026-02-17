"""
Permutation importance for feature auditing.

Shuffles each feature independently across the batch dimension and
measures the resulting degradation in model performance.  Features
that cause the most degradation when shuffled are the most important
to the model's predictions.

Usage:
    importance = compute_permutation_importance(
        model, val_ds, feature_names, device
    )
    for name, score in importance.items():
        print(f"{name}: {score:.4f}")
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .architecture import CryptoPredictorLSTM, PredictionLoss
from .dataset import CryptoTimeSeriesDataset

logger = logging.getLogger(__name__)


def compute_permutation_importance(
    model: CryptoPredictorLSTM,
    dataset: CryptoTimeSeriesDataset,
    feature_names: List[str],
    device: torch.device,
    n_repeats: int = 5,
    batch_size: int = 64,
) -> Dict[str, float]:
    """
    Compute permutation importance for each feature.

    For each feature:
      1. Shuffle that feature across samples in the batch
      2. Measure the increase in loss compared to baseline
      3. Repeat ``n_repeats`` times and average

    Parameters
    ----------
    model : trained CryptoPredictorLSTM
    dataset : validation dataset to evaluate on
    feature_names : ordered list of feature column names
    device : torch device
    n_repeats : number of shuffle repetitions per feature
    batch_size : DataLoader batch size

    Returns
    -------
    Dict mapping feature name to importance score (higher = more important).
    Sorted descending by importance.
    """
    model.eval()
    criterion = PredictionLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    baseline_loss = _compute_loss(model, loader, criterion, device)
    logger.info("Baseline validation loss: %.4f", baseline_loss)

    importances: Dict[str, float] = {}

    for feat_idx, feat_name in enumerate(feature_names):
        shuffled_losses = []

        for _ in range(n_repeats):
            loss = _compute_loss_with_shuffled_feature(
                model, loader, criterion, device, feat_idx
            )
            shuffled_losses.append(loss)

        mean_shuffled = float(np.mean(shuffled_losses))
        importance = mean_shuffled - baseline_loss
        importances[feat_name] = importance

    # Sort by importance descending
    importances = dict(
        sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    )

    logger.info(
        "Permutation importance for %d features. Top 5: %s",
        len(importances),
        ", ".join(f"{k}={v:.4f}" for k, v in list(importances.items())[:5]),
    )

    return importances


def format_importance_report(importances: Dict[str, float], top_n: int = 15) -> str:
    """Format permutation importance as a human-readable report."""
    lines = ["**Feature Importance (Permutation)**\n```"]
    items = list(importances.items())

    for i, (name, score) in enumerate(items[:top_n]):
        bar = "#" * max(1, int(score * 200))  # visual bar
        lines.append(f"  {i+1:2d}. {name:<22s}  {score:+.4f}  {bar}")

    if len(items) > top_n:
        lines.append(f"  ... ({len(items) - top_n} more features)")

    # Bottom 3 (least important)
    if len(items) > top_n:
        lines.append("")
        lines.append("  Least important:")
        for name, score in items[-3:]:
            lines.append(f"      {name:<22s}  {score:+.4f}")

    lines.append("```")
    return "\n".join(lines)


@torch.no_grad()
def _compute_loss(
    model: CryptoPredictorLSTM,
    loader: DataLoader,
    criterion: PredictionLoss,
    device: torch.device,
) -> float:
    """Compute total loss on the dataset."""
    total_loss = 0.0
    total_samples = 0

    for x, y_dir, y_mag in loader:
        x = x.to(device)
        y_dir = y_dir.to(device)
        y_mag = y_mag.to(device)

        cls_logits, mag_pred = model(x)
        loss, _, _ = criterion(cls_logits, mag_pred, y_dir, y_mag)

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def _compute_loss_with_shuffled_feature(
    model: CryptoPredictorLSTM,
    loader: DataLoader,
    criterion: PredictionLoss,
    device: torch.device,
    feat_idx: int,
) -> float:
    """Compute loss with one feature shuffled across the batch."""
    total_loss = 0.0
    total_samples = 0

    for x, y_dir, y_mag in loader:
        x = x.clone()

        # Shuffle the target feature across the batch dimension
        perm = torch.randperm(x.size(0))
        x[:, :, feat_idx] = x[perm, :, feat_idx]

        x = x.to(device)
        y_dir = y_dir.to(device)
        y_mag = y_mag.to(device)

        cls_logits, mag_pred = model(x)
        loss, _, _ = criterion(cls_logits, mag_pred, y_dir, y_mag)

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / max(total_samples, 1)
