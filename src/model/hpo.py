"""
Hyperparameter optimization using Optuna.

Optimizes for validation Sharpe ratio (not accuracy) across
the LSTM, TFT, and GBM model families.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.pruners import MedianPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.info("Optuna not installed — HPO disabled")


def optimize_lstm_hyperparams(
    train_fn: Callable[[Dict[str, Any]], float],
    n_trials: int = 50,
    study_name: str = "lstm_hpo",
    storage_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Optimize LSTM hyperparameters using Optuna.

    Parameters
    ----------
    train_fn : callable
        Function that accepts a hyperparameter dict and returns a
        validation metric (higher is better, e.g., Sharpe ratio).
    n_trials : int
        Number of optimization trials.
    study_name : str
        Name for the Optuna study.
    storage_path : str | None
        SQLite path for study persistence.

    Returns
    -------
    Best hyperparameters as a dict.
    """
    if not HAS_OPTUNA:
        logger.warning("Optuna not installed — returning default hyperparameters")
        return _default_lstm_params()

    def objective(trial):
        params = {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "window_size": trial.suggest_categorical("window_size", [96, 168, 336]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
        return train_fn(params)

    storage = f"sqlite:///{storage_path}" if storage_path else None
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)

    logger.info("HPO complete. Best params: %s", study.best_params)
    logger.info("Best value: %.4f", study.best_value)
    return study.best_params


def optimize_tft_hyperparams(
    train_fn: Callable[[Dict[str, Any]], float],
    n_trials: int = 50,
) -> Dict[str, Any]:
    """Optimize TFT hyperparameters."""
    if not HAS_OPTUNA:
        return _default_tft_params()

    def objective(trial):
        params = {
            "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
            "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8]),
            "num_lstm_layers": trial.suggest_int("num_lstm_layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.05, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        }
        return train_fn(params)

    study = optuna.create_study(direction="maximize", pruner=MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def optimize_gbm_hyperparams(
    train_fn: Callable[[Dict[str, Any]], float],
    n_trials: int = 50,
) -> Dict[str, Any]:
    """Optimize GBM hyperparameters."""
    if not HAS_OPTUNA:
        return _default_gbm_params()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        return train_fn(params)

    study = optuna.create_study(direction="maximize", pruner=MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def _default_lstm_params() -> Dict[str, Any]:
    return {
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "window_size": 168,
        "batch_size": 64,
    }


def _default_tft_params() -> Dict[str, Any]:
    return {
        "d_model": 64,
        "num_heads": 4,
        "num_lstm_layers": 1,
        "dropout": 0.1,
        "learning_rate": 0.001,
    }


def _default_gbm_params() -> Dict[str, Any]:
    return {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }
