"""
Training loop for the CryptoPredictorLSTM model.

Supports:
  - Single-fold or walk-forward cross-validation training
  - Automatic class-weight balancing for imbalanced labels
  - Automatic checkpointing of best model
  - Early stopping
  - Learning rate scheduling
  - Post-training temperature calibration for probability calibration
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .architecture import CryptoPredictorLSTM, PredictionLoss
from .dataset import CryptoTimeSeriesDataset

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "models"
)


class Trainer:
    """End-to-end model trainer with validation, checkpointing, and scheduling."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_cls: float = 1.0,
        lambda_reg: float = 1.0,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        checkpoint_dir: Optional[str] = None,
        device: Optional[str] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CryptoPredictorLSTM(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        # Move class weights to device if provided
        if class_weights is not None:
            class_weights = class_weights.to(self.device)

        self.criterion = PredictionLoss(
            lambda_cls=lambda_cls,
            lambda_reg=lambda_reg,
            class_weights=class_weights,
        )
        self.optimizer = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5,
        )

    def fit(
        self,
        train_ds: CryptoTimeSeriesDataset,
        val_ds: CryptoTimeSeriesDataset,
        tag: str = "fold",
        use_sample_weights: bool = False,
        periodic_checkpoint_every: int = 10,
    ) -> Dict[str, list]:
        """
        Train the model on *train_ds* and evaluate on *val_ds*.

        Parameters
        ----------
        periodic_checkpoint_every : int
            Save a periodic checkpoint every N epochs (in addition to best).
            Set to 0 to disable.

        Returns a dict of training history: {train_loss, val_loss, val_acc, ...}
        """
        if use_sample_weights:
            sampler = train_ds.get_sampler()
            train_loader = DataLoader(
                train_ds, batch_size=self.batch_size, sampler=sampler, drop_last=True
            )
            logger.info("[%s] Using weighted sampling for training", tag)
        else:
            train_loader = DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True
            )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False
        )

        history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_cls_acc": [],
            "val_reg_mae": [],
        }

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_path = os.path.join(self.checkpoint_dir, f"best_{tag}.pt")
        periodic_path = os.path.join(self.checkpoint_dir, f"periodic_{tag}.pt")

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_metrics = self._validate_epoch(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_cls_acc"].append(val_metrics["cls_acc"])
            history["val_reg_mae"].append(val_metrics["reg_mae"])

            self.scheduler.step(val_metrics["loss"])

            logger.info(
                "[%s] Epoch %3d | train_loss=%.4f  val_loss=%.4f  "
                "val_acc=%.2f%%  val_mae=%.4f  lr=%.2e",
                tag,
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["cls_acc"] * 100,
                val_metrics["reg_mae"],
                self.optimizer.param_groups[0]["lr"],
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), best_path)
                logger.info("[%s] Saved best model (val_loss=%.4f)", tag, best_val_loss)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info("[%s] Early stopping at epoch %d", tag, epoch)
                    break

            # Periodic checkpoint for crash recovery
            if periodic_checkpoint_every > 0 and epoch % periodic_checkpoint_every == 0:
                torch.save(self.model.state_dict(), periodic_path)
                logger.info("[%s] Periodic checkpoint saved at epoch %d", tag, epoch)

        # Reload best checkpoint
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, weights_only=True))
            logger.info("[%s] Reloaded best checkpoint", tag)

        # Clean up periodic checkpoint
        if os.path.exists(periodic_path):
            os.remove(periodic_path)

        return history

    def calibrate_temperature(self, val_ds: CryptoTimeSeriesDataset) -> float:
        """
        Calibrate the model's temperature parameter on validation data.

        Freezes all model weights except ``temperature`` and optimizes it
        to minimize Negative Log-Likelihood on the validation set.  This
        ensures softmax probabilities are well-calibrated (i.e. "70% UP"
        actually corresponds to ~70% observed frequency).

        Returns the calibrated temperature value.
        """
        self.model.eval()
        loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Collect all logits and labels
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for x, y_dir, _y_mag in loader:
                x = x.to(self.device)
                cls_logits, _ = self.model(x)
                all_logits.append(cls_logits)
                all_labels.append(y_dir.to(self.device))

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Freeze everything except temperature
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.temperature.requires_grad = True

        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(
            [self.model.temperature], lr=0.01, max_iter=50
        )

        def _closure():
            optimizer.zero_grad()
            scaled = all_logits / self.model.temperature.clamp(min=0.01)
            loss = nll(scaled, all_labels)
            loss.backward()
            return loss

        optimizer.step(_closure)

        # Restore requires_grad
        for param in self.model.parameters():
            param.requires_grad = True

        temp = self.model.temperature.item()
        logger.info("Temperature calibrated to %.4f", temp)
        return temp

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y_dir, y_mag in loader:
            x = x.to(self.device)
            y_dir = y_dir.to(self.device)
            y_mag = y_mag.to(self.device)

            cls_logits, mag_pred = self.model(x)
            loss, _, _ = self.criterion(cls_logits, mag_pred, y_dir, y_mag)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        abs_errors: list[float] = []

        for x, y_dir, y_mag in loader:
            x = x.to(self.device)
            y_dir = y_dir.to(self.device)
            y_mag = y_mag.to(self.device)

            cls_logits, mag_pred = self.model(x)
            loss, _, _ = self.criterion(cls_logits, mag_pred, y_dir, y_mag)

            total_loss += loss.item() * x.size(0)
            preds = cls_logits.argmax(dim=1)
            correct += (preds == y_dir).sum().item()
            total_samples += x.size(0)
            abs_errors.append((mag_pred.squeeze(-1) - y_mag).abs().mean().item())

        n = max(total_samples, 1)
        return {
            "loss": total_loss / n,
            "cls_acc": correct / n,
            "reg_mae": float(np.mean(abs_errors)) if abs_errors else 0.0,
        }

    def save_final(
        self,
        path: Optional[str] = None,
        tag: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> str:
        """
        Save the current model weights with metadata for version verification.

        Parameters
        ----------
        path : str | None
            Explicit file path. Overrides tag.
        tag : str | None
            If set, saves as ``model_{tag}.pt`` (e.g. ``model_final_60.pt``
            for the 1h model). Ignored when *path* is provided.
        feature_cols : list[str] | None
            Feature column names used during training. Saved as a hash for
            mismatch detection at load time.
        """
        if path is None:
            filename = f"model_{tag}.pt" if tag else "model_final.pt"
            path = os.path.join(self.checkpoint_dir, filename)

        cols_hash = ""
        if feature_cols:
            cols_hash = hashlib.md5(",".join(feature_cols).encode()).hexdigest()

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "num_features": self.model.num_features,
            "hidden_dim": self.model.hidden_dim,
            "feature_cols_hash": cols_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(checkpoint, path)
        logger.info("Model saved to %s (num_features=%d, hidden=%d)",
                     path, self.model.num_features, self.model.hidden_dim)
        return path

    def evaluate(self, dataset: CryptoTimeSeriesDataset) -> Dict[str, float]:
        """
        Evaluate the current model on a dataset and return metrics.

        Returns dict with keys: loss, cls_acc, reg_mae.
        """
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return self._validate_epoch(loader)

    def load(self, path: str) -> None:
        """Load model weights from checkpoint, verifying metadata if present."""
        data = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(data, dict) and "model_state_dict" in data:
            ckpt_feats = data.get("num_features")
            ckpt_hidden = data.get("hidden_dim")
            if ckpt_feats and ckpt_feats != self.model.num_features:
                raise RuntimeError(
                    f"Checkpoint expects {ckpt_feats} features but model "
                    f"has {self.model.num_features}. Retrain required."
                )
            if ckpt_hidden and ckpt_hidden != self.model.hidden_dim:
                raise RuntimeError(
                    f"Checkpoint expects hidden_dim={ckpt_hidden} but model "
                    f"has {self.model.hidden_dim}. Config mismatch."
                )
            self.model.load_state_dict(data["model_state_dict"])
            logger.info(
                "Model loaded from %s (created=%s)",
                path, data.get("created_at", "unknown"),
            )
        else:
            # Legacy checkpoint: plain state_dict
            self.model.load_state_dict(data)
            logger.info("Model loaded from %s (legacy format)", path)

        self.model.eval()
