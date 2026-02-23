"""
Gradient Boosting Machine (GBM) baseline model.

Uses LightGBM for classification + regression as a complementary
model to neural approaches. Fast to train, strong baseline.

Falls back to scikit-learn's GradientBoostingClassifier/Regressor
if LightGBM is not installed.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.info("LightGBM not installed — falling back to sklearn GBM")

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder


@dataclass
class GBMConfig:
    """Configuration for GBM model."""
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 20
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 30
    random_state: int = 42


class GBMPredictor:
    """
    Gradient boosting model for direction classification and magnitude regression.

    Operates on flattened feature windows (window_size * num_features).
    """

    def __init__(self, config: Optional[GBMConfig] = None):
        self.config = config or GBMConfig()
        self.clf = None
        self.reg = None
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        self.is_fitted = False

    def _flatten_window(self, windows: np.ndarray) -> np.ndarray:
        """Flatten 3D (N, T, F) to 2D (N, T*F) and add aggregated features."""
        N, T, F = windows.shape
        flat = windows.reshape(N, -1)

        last_step = windows[:, -1, :]
        means = windows.mean(axis=1)
        stds = windows.std(axis=1)
        deltas = windows[:, -1, :] - windows[:, 0, :]

        return np.concatenate([flat, last_step, means, stds, deltas], axis=1)

    def fit(
        self,
        X_train: np.ndarray,
        y_cls_train: np.ndarray,
        y_reg_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_cls_val: Optional[np.ndarray] = None,
        y_reg_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train the GBM classifier and regressor.

        Parameters
        ----------
        X_train : (N, T, F) feature windows
        y_cls_train : (N,) direction labels (0=UP, 1=FLAT, 2=DOWN)
        y_reg_train : (N,) magnitude values
        X_val : optional validation features
        y_cls_val : optional validation labels
        y_reg_val : optional validation magnitudes

        Returns
        -------
        Dict with training/validation metrics.
        """
        X_flat = self._flatten_window(X_train)
        cfg = self.config
        metrics: Dict[str, float] = {}

        if HAS_LIGHTGBM:
            self.clf = lgb.LGBMClassifier(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                subsample=cfg.subsample,
                colsample_bytree=cfg.colsample_bytree,
                min_child_samples=cfg.min_child_samples,
                reg_alpha=cfg.reg_alpha,
                reg_lambda=cfg.reg_lambda,
                random_state=cfg.random_state,
                verbosity=-1,
            )
            self.reg = lgb.LGBMRegressor(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                subsample=cfg.subsample,
                random_state=cfg.random_state,
                verbosity=-1,
            )
        else:
            self.clf = GradientBoostingClassifier(
                n_estimators=min(cfg.n_estimators, 200),
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                subsample=cfg.subsample,
                random_state=cfg.random_state,
            )
            self.reg = GradientBoostingRegressor(
                n_estimators=min(cfg.n_estimators, 200),
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                subsample=cfg.subsample,
                random_state=cfg.random_state,
            )

        if X_val is not None and HAS_LIGHTGBM:
            X_val_flat = self._flatten_window(X_val)
            callbacks = [lgb.early_stopping(cfg.early_stopping_rounds)]
            self.clf.fit(
                X_flat, y_cls_train,
                eval_set=[(X_val_flat, y_cls_val)],
                callbacks=callbacks,
            )
            self.reg.fit(
                X_flat, y_reg_train,
                eval_set=[(X_val_flat, y_reg_val)],
                callbacks=callbacks,
            )
        else:
            self.clf.fit(X_flat, y_cls_train)
            self.reg.fit(X_flat, y_reg_train)

        train_acc = (self.clf.predict(X_flat) == y_cls_train).mean()
        metrics["train_cls_acc"] = float(train_acc)

        train_mae = np.abs(self.reg.predict(X_flat) - y_reg_train).mean()
        metrics["train_reg_mae"] = float(train_mae)

        if X_val is not None:
            X_val_flat = self._flatten_window(X_val)
            val_acc = (self.clf.predict(X_val_flat) == y_cls_val).mean()
            metrics["val_cls_acc"] = float(val_acc)
            val_mae = np.abs(self.reg.predict(X_val_flat) - y_reg_val).mean()
            metrics["val_reg_mae"] = float(val_mae)

        self.is_fitted = True
        logger.info("GBM training complete: %s", metrics)
        return metrics

    def predict(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions.

        Parameters
        ----------
        X : (N, T, F) feature windows

        Returns
        -------
        class_probs : (N, 3) probability per class
        predicted_class : (N,) class indices
        magnitude : (N,) predicted % change
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted — call fit() first")

        X_flat = self._flatten_window(X)
        class_probs = self.clf.predict_proba(X_flat)
        predicted_class = self.clf.predict(X_flat)
        magnitude = self.reg.predict(X_flat)

        return class_probs, predicted_class, magnitude

    def save(self, path: str) -> None:
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "clf": self.clf,
                "reg": self.reg,
                "config": self.config,
            }, f)
        logger.info("GBM model saved to %s", path)

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.clf = data["clf"]
        self.reg = data["reg"]
        self.config = data.get("config", GBMConfig())
        self.is_fitted = True
        logger.info("GBM model loaded from %s", path)

    def feature_importance(self) -> Optional[np.ndarray]:
        """Return feature importances from the classifier."""
        if self.clf is None:
            return None
        if hasattr(self.clf, "feature_importances_"):
            return self.clf.feature_importances_
        return None
