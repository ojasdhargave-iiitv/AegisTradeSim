"""Dynamic Agent Credibility Matrix (Contextual Bandit) for AegisTradeSim RL."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ROLES = ["market_analyst", "fundamentals_analyst", "news_analyst", "sentiment_analyst"]


class AgentCredibilityBandit:
    """Tracks historical prediction accuracy for each agent role and generates dynamic credibility weights."""

    def __init__(self, persist_path: str | Path | None = None):
        if persist_path is None:
            home = os.path.expanduser("~")
            persist_path = Path(home) / ".aegistradesim" / "rl_store" / "agent_weights.json"
        self.persist_path = Path(persist_path).expanduser()
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.weights_data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.persist_path.exists():
            # Initialize with balanced default prior
            return {
                "global_weights": {role: 0.50 for role in DEFAULT_ROLES},
                "role_stats": {
                    role: {"correct": 5, "total": 10, "win_rate": 0.50} for role in DEFAULT_ROLES
                },
                "ticker_specific": {},
            }
        try:
            return json.loads(self.persist_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Could not read agent weights (%s); starting fresh.", e)
            return {
                "global_weights": {role: 0.50 for role in DEFAULT_ROLES},
                "role_stats": {
                    role: {"correct": 5, "total": 10, "win_rate": 0.50} for role in DEFAULT_ROLES
                },
                "ticker_specific": {},
            }

    def save(self) -> None:
        try:
            self.persist_path.write_text(json.dumps(self.weights_data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error("Failed to save agent credibility weights: %s", e)

    def update_role_credibility(
        self,
        role: str,
        is_successful: bool,
        ticker: str | None = None,
        learning_rate: float = 0.1,
    ) -> float:
        """Update historical win rate and credibility weight using exponential moving average (Bandit update)."""
        if role not in self.weights_data["role_stats"]:
            self.weights_data["role_stats"][role] = {"correct": 5, "total": 10, "win_rate": 0.50}
            self.weights_data["global_weights"][role] = 0.50

        stats = self.weights_data["role_stats"][role]
        stats["total"] += 1
        if is_successful:
            stats["correct"] += 1
        stats["win_rate"] = round(stats["correct"] / stats["total"], 4)

        # Update weight via exponential smoothing
        current_w = self.weights_data["global_weights"].get(role, 0.50)
        target = 0.90 if is_successful else 0.20
        new_w = round((1.0 - learning_rate) * current_w + (learning_rate * target), 4)
        new_w = max(0.10, min(0.95, new_w))
        self.weights_data["global_weights"][role] = new_w

        # If ticker provided, record ticker-specific credibility
        if ticker:
            ticker = ticker.upper()
            if ticker not in self.weights_data["ticker_specific"]:
                self.weights_data["ticker_specific"][ticker] = {}
            t_weights = self.weights_data["ticker_specific"][ticker]
            curr_t_w = t_weights.get(role, new_w)
            new_t_w = round((1.0 - learning_rate) * curr_t_w + (learning_rate * target), 4)
            t_weights[role] = max(0.10, min(0.95, new_t_w))

        self.save()
        return new_w

    def get_credibility_prompt(self, ticker: str | None = None) -> str:
        """Return a formatted prompt instruction for LLM debate & portfolio manager weighting."""
        weights = dict(self.weights_data.get("global_weights", {}))
        if ticker:
            ticker_weights = self.weights_data.get("ticker_specific", {}).get(ticker.upper(), {})
            weights.update(ticker_weights)

        # Sort roles by descending credibility
        sorted_roles = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_role, top_val = sorted_roles[0]
        lowest_role, low_val = sorted_roles[-1]

        lines = [
            "### [RL DYNAMIC AGENT CREDIBILITY WEIGHTS]",
            "Based on historical real-world trade tracking and regressive prediction error, weight agent arguments accordingly:",
        ]
        for role, weight in sorted_roles:
            role_title = role.replace("_", " ").title()
            lines.append(f"- **{role_title} Credibility Weight:** {weight:.2f}")

        lines.append(
            f"\n*Decision Directive:* Prioritize signals from **{top_role.replace('_', ' ').title()}** ({top_val:.2f}) "
            f"over speculative claims from **{lowest_role.replace('_', ' ').title()}** ({low_val:.2f})."
        )
        return "\n".join(lines)

    def get_weights_summary(self) -> dict[str, float]:
        """Return dict of current global credibility weights."""
        return self.weights_data.get("global_weights", {r: 0.5 for r in DEFAULT_ROLES})
