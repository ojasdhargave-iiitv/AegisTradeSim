"""Continuous Regressive Loss and Reward Formulation for AegisTradeSim RL."""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


class RegressiveLossCalculator:
    """Calculates continuous regressive prediction error, directional penalty, and RL reward signal.
    
    Mathematical Formulation:
        L_regression = |R_hat - R_realized| + lambda * I(sign(R_hat) != sign(R_realized))
        Reward = sign(R_hat * R_realized) * |R_realized| - penalty
    """

    def __init__(self, direction_penalty_lambda: float = 0.05, error_tolerance: float = 0.03):
        self.direction_penalty_lambda = direction_penalty_lambda
        self.error_tolerance = error_tolerance

    def calculate_loss(
        self,
        predicted_return_pct: float,
        realized_return_pct: float,
        action: str = "HOLD",
    ) -> dict[str, Any]:
        """Compute continuous regressive loss, directional accuracy, and reward.
        
        Args:
            predicted_return_pct: Expected percentage return from decision (e.g. +5.0)
            realized_return_pct: Real-world forward market percentage return (e.g. -2.5)
            action: BUY, SELL, SHORT, or HOLD
        """
        # Convert percent to decimal
        r_hat = predicted_return_pct / 100.0
        r_real = realized_return_pct / 100.0

        # Absolute return discrepancy
        abs_error = abs(r_hat - r_real)

        # Directional alignment check
        # For HOLD: aligned if realized market move was mild (< 3% either way)
        directional_match = False
        if action == "HOLD":
            directional_match = abs(r_real) <= self.error_tolerance
        elif action in ("BUY", "LONG"):
            directional_match = r_real > 0.0
        elif action in ("SELL", "SHORT"):
            directional_match = r_real < 0.0
        else:
            directional_match = (r_hat * r_real) >= 0.0

        directional_penalty = 0.0 if directional_match else self.direction_penalty_lambda
        total_loss = abs_error + directional_penalty

        # Compute RL Reward signal [-1.0, +1.0]
        if directional_match:
            # Positive reinforcement scaled by realized alpha/return
            reward = max(0.1, min(1.0, 0.5 + abs(r_real) - (abs_error * 0.5)))
        else:
            # Negative reinforcement penalized by loss magnitude
            reward = max(-1.0, min(-0.1, -0.5 - directional_penalty - (abs_error * 0.5)))

        is_significant_failure = total_loss > (self.error_tolerance * 2) and not directional_match

        return {
            "predicted_return_pct": predicted_return_pct,
            "realized_return_pct": realized_return_pct,
            "absolute_error": round(abs_error, 4),
            "directional_match": directional_match,
            "directional_penalty": round(directional_penalty, 4),
            "total_regressive_loss": round(total_loss, 4),
            "rl_reward_score": round(reward, 4),
            "is_significant_failure": is_significant_failure,
            "direction_score": 1.0 if directional_match else 0.0,
        }
