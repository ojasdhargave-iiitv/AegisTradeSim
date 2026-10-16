"""Unified RL Engine and Telemetry Audit System for AegisTradeSim."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any

from aegistradesim.rl.bandit_weights import AgentCredibilityBandit
from aegistradesim.rl.critic_agent import ReflexionCriticAgent
from aegistradesim.rl.extractor import DecisionExtractor
from aegistradesim.rl.outcome_tracker import OutcomeTracker
from aegistradesim.rl.regressive_loss import RegressiveLossCalculator

logger = logging.getLogger(__name__)


class RLEngine:
    """Unified Facade for Regressive Learning and Real-World RL Feedback."""

    def __init__(self):
        self.extractor = DecisionExtractor()
        self.tracker = OutcomeTracker()
        self.loss_calc = RegressiveLossCalculator()
        self.bandit = AgentCredibilityBandit()
        self.critic = ReflexionCriticAgent()

    def evaluate_report(
        self,
        report_dir: Path | str,
        horizon_days: int = 10,
    ) -> dict[str, Any]:
        """Perform end-to-end RL evaluation on a report:
        1. Extract decision, spot price, targets
        2. Track real-world market outcome via yfinance
        3. Compute continuous regressive loss & reward
        4. Update agent credibility bandit weights
        5. Trigger Reflexion Critic post-mortem if loss or directional failure occurred
        6. Generate 10-line quantitative audit report in aegistradesim/rl/reports/
        """
        report_dir = Path(report_dir)
        pred = self.extractor.extract_from_report_dir(report_dir)
        ticker = pred["ticker"]
        trade_date = pred["trade_date"]
        action = pred["action"]
        entry_price = pred["entry_price"]
        expected_ret = pred["expected_return_pct"]

        # 2. Track Real-World Outcome
        outcome = self.tracker.calculate_trade_outcome(
            ticker=ticker,
            trade_date=trade_date,
            horizon_days=horizon_days,
            provided_entry_price=entry_price,
        )
        realized_ret = outcome["realized_return_pct"]
        alpha = outcome["alpha_pct"]
        spy_ret = outcome["spy_benchmark_return_pct"]

        # 3. Compute Regressive Loss
        loss_info = self.loss_calc.calculate_loss(
            predicted_return_pct=expected_ret,
            realized_return_pct=realized_ret,
            action=action,
        )
        total_loss = loss_info["total_regressive_loss"]
        dir_match = loss_info["directional_match"]
        is_failure = loss_info["is_significant_failure"]

        # 4. Update Dynamic Agent Credibility Bandit
        # If market call was correct, reward technical market analyst; adjust others
        self.bandit.update_role_credibility("market_analyst", is_successful=dir_match, ticker=ticker)
        self.bandit.update_role_credibility("fundamentals_analyst", is_successful=(alpha >= 0), ticker=ticker)
        self.bandit.update_role_credibility("sentiment_analyst", is_successful=dir_match, ticker=ticker)

        # 5. Reflexion Post-Mortem if loss or failure
        critique_generated = 0
        if is_failure or not dir_match or abs(total_loss) > 0.05:
            self.critic.generate_post_mortem(
                ticker=ticker,
                trade_date=trade_date,
                predicted_action=action,
                predicted_return_pct=expected_ret,
                realized_return_pct=realized_ret,
                alpha_pct=alpha,
                regressive_loss=total_loss,
            )
            critique_generated = 1

        # 6. Generate 10-Line Quantitative Telemetry Report
        telemetry_file = self.generate_telemetry_report(
            ticker=ticker,
            trade_date=trade_date,
            predicted_return_pct=expected_ret,
            realized_return_pct=realized_ret,
            spy_return_pct=spy_ret,
            alpha_pct=alpha,
            regressive_loss=total_loss,
            directional_match=dir_match,
            critique_generated=critique_generated,
        )

        return {
            "ticker": ticker,
            "trade_date": trade_date,
            "action": action,
            "predicted_return_pct": expected_ret,
            "realized_return_pct": realized_ret,
            "alpha_pct": alpha,
            "regressive_loss": total_loss,
            "directional_match": dir_match,
            "critique_generated": critique_generated,
            "telemetry_file": str(telemetry_file),
        }

    def generate_telemetry_report(
        self,
        ticker: str,
        trade_date: str,
        predicted_return_pct: float,
        realized_return_pct: float,
        spy_return_pct: float,
        alpha_pct: float,
        regressive_loss: float,
        directional_match: bool,
        critique_generated: int,
    ) -> Path:
        """Produce a strict 10-line quantitative telemetry report in aegistradesim/rl/reports."""
        ticker = ticker.upper()
        now_ts = int(datetime.datetime.now().timestamp())
        ts_slug = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        weights = self.bandit.get_weights_summary()
        market_w = weights.get("market_analyst", 0.50)
        sentiment_w = weights.get("sentiment_analyst", 0.50)
        total_lessons = self.critic.get_lessons_count()
        dir_score = 1.0 if directional_match else 0.0

        lines = [
            f"# RL TELEMETRY AUDIT: {ticker} [{trade_date}]",
            f"1. RL_PREDICTED_RETURN_PCT: {predicted_return_pct:+.2f}%",
            f"2. REAL_WORLD_REALIZED_RETURN_PCT: {realized_return_pct:+.2f}%",
            f"3. BENCHMARK_SPY_RETURN_PCT: {spy_return_pct:+.2f}%",
            f"4. GENERATED_ALPHA_PCT: {alpha_pct:+.2f}%",
            f"5. REGRESSIVE_ERROR_LOSS: {regressive_loss:.4f}",
            f"6. DIRECTIONAL_ACCURACY_SCORE: {dir_score:.2f}",
            f"7. REFLEXION_CRITIQUE_GENERATED: {critique_generated}",
            f"8. AGENT_CREDIBILITY_MARKET_WEIGHT: {market_w:.2f}",
            f"9. AGENT_CREDIBILITY_SENTIMENT_WEIGHT: {sentiment_w:.2f}",
            f"10. TOTAL_RL_LESSONS_STORED: {total_lessons}",
        ]
        content = "\n".join(lines) + "\n"

        rl_reports_dir = Path(__file__).resolve().parent / "reports"
        rl_reports_dir.mkdir(parents=True, exist_ok=True)

        specific_file = rl_reports_dir / f"{ticker}_{ts_slug}_rl_telemetry.md"
        latest_file = rl_reports_dir / "latest_rl_telemetry.md"

        specific_file.write_text(content, encoding="utf-8")
        latest_file.write_text(content, encoding="utf-8")

        print("\n" + "=" * 55)
        print(f"📈 RL QUANTITATIVE TELEMETRY GENERATED -> aegistradesim/rl/reports/")
        for l in lines[1:]:
            print(f"   {l}")
        print("=" * 55 + "\n")

        return specific_file

    def get_agent_credibility_prompt(self, ticker: str | None = None) -> str:
        """Returns dynamic credibility prompt for LLM injection."""
        return self.bandit.get_credibility_prompt(ticker=ticker)


_GLOBAL_RL_ENGINE: RLEngine | None = None


def get_rl_engine() -> RLEngine:
    """Singleton getter for global RL Engine."""
    global _GLOBAL_RL_ENGINE
    if _GLOBAL_RL_ENGINE is None:
        _GLOBAL_RL_ENGINE = RLEngine()
    return _GLOBAL_RL_ENGINE
