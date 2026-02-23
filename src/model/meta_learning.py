"""
Meta-learning for per-symbol, per-regime model selection.

Tracks which models perform best for which symbol/regime combinations
and dynamically selects the best model at inference time.
Also clusters symbols by behavior type.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceRecord:
    """Tracks a model's performance for a symbol-regime combination."""
    model_name: str
    symbol: str
    regime: int
    sharpe: float
    accuracy: float
    n_predictions: int
    updated_at: str = ""


class MetaLearner:
    """
    Learns which model performs best for each symbol-regime combination.

    Maintains a performance table and provides model selection at inference.
    """

    def __init__(self, decay_factor: float = 0.95):
        self.decay_factor = decay_factor
        self._performance: Dict[Tuple[str, str, int], PerformanceRecord] = {}
        self._cluster_labels: Dict[str, int] = {}

    def record_performance(
        self,
        model_name: str,
        symbol: str,
        regime: int,
        sharpe: float,
        accuracy: float,
        n_predictions: int,
        timestamp: str = "",
    ) -> None:
        """Record model performance for a symbol-regime pair."""
        key = (model_name, symbol, regime)
        existing = self._performance.get(key)

        if existing:
            existing.sharpe = (
                self.decay_factor * existing.sharpe +
                (1 - self.decay_factor) * sharpe
            )
            existing.accuracy = (
                self.decay_factor * existing.accuracy +
                (1 - self.decay_factor) * accuracy
            )
            existing.n_predictions += n_predictions
            existing.updated_at = timestamp
        else:
            self._performance[key] = PerformanceRecord(
                model_name=model_name,
                symbol=symbol,
                regime=regime,
                sharpe=sharpe,
                accuracy=accuracy,
                n_predictions=n_predictions,
                updated_at=timestamp,
            )

    def select_best_model(
        self,
        symbol: str,
        regime: int,
        available_models: List[str],
        min_predictions: int = 10,
    ) -> str:
        """
        Select the best model for the given symbol-regime combination.

        Falls back to the model with best overall performance if no
        specific data exists for this combination.
        """
        best_model = available_models[0]
        best_sharpe = -np.inf

        for model_name in available_models:
            key = (model_name, symbol, regime)
            record = self._performance.get(key)

            if record and record.n_predictions >= min_predictions:
                if record.sharpe > best_sharpe:
                    best_sharpe = record.sharpe
                    best_model = model_name

        if best_sharpe == -np.inf:
            best_model = self._get_best_overall(available_models, min_predictions)

        return best_model

    def _get_best_overall(
        self,
        available_models: List[str],
        min_predictions: int = 10,
    ) -> str:
        """Get the best overall model across all symbols and regimes."""
        model_sharpes: Dict[str, List[float]] = defaultdict(list)

        for (model_name, _, _), record in self._performance.items():
            if model_name in available_models and record.n_predictions >= min_predictions:
                model_sharpes[model_name].append(record.sharpe)

        best = available_models[0]
        best_avg = -np.inf
        for name, sharpes in model_sharpes.items():
            avg = np.mean(sharpes)
            if avg > best_avg:
                best_avg = avg
                best = name

        return best

    def cluster_symbols(
        self,
        returns_data: Dict[str, np.ndarray],
        n_clusters: int = 5,
    ) -> Dict[str, int]:
        """
        Cluster symbols by behavior type using K-means on return statistics.

        Features: mean return, volatility, skewness, kurtosis, autocorrelation.
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        symbols = []
        features = []

        for sym, returns in returns_data.items():
            if len(returns) < 50:
                continue

            r = returns[-500:] if len(returns) > 500 else returns
            feat = [
                np.mean(r),
                np.std(r),
                float(np.mean(r**3) / (np.std(r)**3 + 1e-10)),  # skewness
                float(np.mean(r**4) / (np.std(r)**4 + 1e-10)),  # kurtosis
                float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 2 else 0,
            ]
            symbols.append(sym)
            features.append(feat)

        if len(features) < n_clusters:
            return {s: 0 for s in symbols}

        X = StandardScaler().fit_transform(np.array(features))
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)

        self._cluster_labels = {sym: int(lbl) for sym, lbl in zip(symbols, labels)}

        for cluster_id in range(n_clusters):
            cluster_syms = [s for s, label in self._cluster_labels.items() if label == cluster_id]
            logger.info("Cluster %d: %d symbols — %s",
                        cluster_id, len(cluster_syms), cluster_syms[:5])

        return self._cluster_labels

    def get_cluster(self, symbol: str) -> int:
        """Get cluster assignment for a symbol."""
        return self._cluster_labels.get(symbol, -1)

    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Return summary of model performance across all conditions."""
        model_stats: Dict[str, List[float]] = defaultdict(list)

        for (model_name, _, _), record in self._performance.items():
            model_stats[model_name].append(record.sharpe)

        return {
            name: {
                "mean_sharpe": float(np.mean(sharpes)),
                "std_sharpe": float(np.std(sharpes)),
                "n_combinations": len(sharpes),
            }
            for name, sharpes in model_stats.items()
        }
