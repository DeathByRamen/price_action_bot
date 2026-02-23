"""
A/B testing framework for model comparison.

Runs a shadow model alongside the production model, compares predictions
without affecting live signals, and supports automated promotion.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    min_days_for_promotion: int = 7
    sharpe_improvement_threshold: float = 0.1
    min_predictions: int = 100
    gradual_rollout_steps: List[float] = field(
        default_factory=lambda: [0.10, 0.25, 0.50, 1.00]
    )


@dataclass
class ModelPerformance:
    """Tracks a model's performance during A/B testing."""
    model_name: str
    predictions: List[Dict] = field(default_factory=list)
    pnls: List[float] = field(default_factory=list)
    start_date: str = ""

    @property
    def n_predictions(self) -> int:
        return len(self.predictions)

    @property
    def sharpe(self) -> float:
        if len(self.pnls) < 2:
            return 0.0
        mean = np.mean(self.pnls)
        std = np.std(self.pnls, ddof=1)
        if std < 1e-10:
            return 0.0
        return float(mean / std * np.sqrt(365 * 24))

    @property
    def accuracy(self) -> float:
        correct = sum(1 for p in self.predictions if p.get("was_correct", False))
        return correct / max(len(self.predictions), 1)


class ABTestManager:
    """
    Manages A/B tests between production and shadow models.

    The production model generates live signals. The shadow model runs
    in parallel, generating predictions that are recorded but not acted on.
    After sufficient data is collected, the shadow model can be promoted
    if it demonstrates superior risk-adjusted returns.
    """

    def __init__(self, config: Optional[ABTestConfig] = None):
        self.config = config or ABTestConfig()
        self.production = ModelPerformance(model_name="production")
        self.shadow = ModelPerformance(model_name="shadow")
        self._rollout_step: int = 0
        self._promotion_in_progress: bool = False

    def start_test(
        self,
        production_name: str,
        shadow_name: str,
    ) -> None:
        """Start a new A/B test."""
        now = datetime.now(timezone.utc).isoformat()
        self.production = ModelPerformance(model_name=production_name, start_date=now)
        self.shadow = ModelPerformance(model_name=shadow_name, start_date=now)
        self._rollout_step = 0
        self._promotion_in_progress = False
        logger.info(
            "A/B test started: production='%s' vs shadow='%s'",
            production_name, shadow_name,
        )

    def record_production_prediction(
        self,
        prediction: Dict,
        actual_pnl: Optional[float] = None,
    ) -> None:
        """Record a prediction from the production model."""
        self.production.predictions.append(prediction)
        if actual_pnl is not None:
            self.production.pnls.append(actual_pnl)

    def record_shadow_prediction(
        self,
        prediction: Dict,
        actual_pnl: Optional[float] = None,
    ) -> None:
        """Record a prediction from the shadow model."""
        self.shadow.predictions.append(prediction)
        if actual_pnl is not None:
            self.shadow.pnls.append(actual_pnl)

    def evaluate(self) -> Dict[str, any]:
        """
        Evaluate the A/B test and determine if promotion is warranted.

        Returns
        -------
        Dict with keys: should_promote, production_sharpe, shadow_sharpe,
                        n_predictions, days_running, improvement.
        """
        result = {
            "should_promote": False,
            "production_sharpe": self.production.sharpe,
            "shadow_sharpe": self.shadow.sharpe,
            "production_accuracy": self.production.accuracy,
            "shadow_accuracy": self.shadow.accuracy,
            "production_n": self.production.n_predictions,
            "shadow_n": self.shadow.n_predictions,
        }

        if self.shadow.n_predictions < self.config.min_predictions:
            result["reason"] = (
                f"Not enough shadow predictions "
                f"({self.shadow.n_predictions}/{self.config.min_predictions})"
            )
            return result

        if self.production.start_date:
            start = datetime.fromisoformat(self.production.start_date)
            days = (datetime.now(timezone.utc) - start).days
        else:
            days = 0
        result["days_running"] = days

        if days < self.config.min_days_for_promotion:
            result["reason"] = (
                f"Need {self.config.min_days_for_promotion} days, only {days}"
            )
            return result

        improvement = self.shadow.sharpe - self.production.sharpe
        result["improvement"] = improvement

        if improvement >= self.config.sharpe_improvement_threshold:
            result["should_promote"] = True
            result["reason"] = (
                f"Shadow Sharpe ({self.shadow.sharpe:.3f}) beats "
                f"production ({self.production.sharpe:.3f}) by {improvement:.3f}"
            )
            logger.info(
                "A/B test result: PROMOTE shadow model '%s' "
                "(Sharpe: %.3f vs %.3f, improvement: +%.3f)",
                self.shadow.model_name,
                self.shadow.sharpe, self.production.sharpe, improvement,
            )
        else:
            result["reason"] = (
                f"Shadow improvement ({improvement:+.3f}) below "
                f"threshold ({self.config.sharpe_improvement_threshold})"
            )

        return result

    def get_rollout_fraction(self) -> float:
        """Get current rollout fraction for gradual promotion."""
        if self._rollout_step >= len(self.config.gradual_rollout_steps):
            return 1.0
        return self.config.gradual_rollout_steps[self._rollout_step]

    def advance_rollout(self) -> float:
        """Advance to next rollout step. Returns new fraction."""
        self._rollout_step = min(
            self._rollout_step + 1,
            len(self.config.gradual_rollout_steps) - 1,
        )
        frac = self.get_rollout_fraction()
        logger.info("Rollout advanced to %.0f%%", frac * 100)
        return frac
