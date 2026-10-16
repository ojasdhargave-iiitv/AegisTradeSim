"""Reflexion RL Post-Mortem Critic Agent for AegisTradeSim RL."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ReflexionCriticAgent:
    """Performs in-context Reinforcement Learning post-mortems on failed trade predictions.
    
    Generates structured lessons learned, root-cause analyses, and corrective directives,
    then injects them directly into the RAG memory store as negative precedents.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:3b-instruct",
        host: str = "http://localhost:11434",
        persist_path: str | Path | None = None,
    ):
        self.model_name = model_name
        self.host = host
        if persist_path is None:
            home = os.path.expanduser("~")
            persist_path = Path(home) / ".aegistradesim" / "rl_store" / "reflexion_lessons.json"
        self.persist_path = Path(persist_path).expanduser()
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.lessons: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if not self.persist_path.exists():
            return []
        try:
            return json.loads(self.persist_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def save(self) -> None:
        try:
            self.persist_path.write_text(json.dumps(self.lessons, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error("Failed to save reflexion lessons: %s", e)

    def generate_post_mortem(
        self,
        ticker: str,
        trade_date: str,
        predicted_action: str,
        predicted_return_pct: float,
        realized_return_pct: float,
        alpha_pct: float,
        regressive_loss: float,
        report_summary: str = "",
    ) -> dict[str, Any]:
        """Perform post-mortem evaluation comparing prediction to realized market action."""
        ticker = ticker.upper()

        prompt = (
            f"You are a Senior Quantitative Trading Risk Critic. Perform a concise Reflexion RL Post-Mortem.\n"
            f"NO conversational filler. 90% stats and numbers, 10% concise directive.\n\n"
            f"TRADE AUDIT:\n"
            f"- Ticker: {ticker}\n"
            f"- Trade Date: {trade_date}\n"
            f"- Proposed Action: {predicted_action}\n"
            f"- Target Return: {predicted_return_pct:+.2f}%\n"
            f"- Realized Forward Return: {realized_return_pct:+.2f}%\n"
            f"- Realized Alpha vs SPY: {alpha_pct:+.2f}%\n"
            f"- Regressive Loss: {regressive_loss:.4f}\n"
            f"- Context: {report_summary[:300]}\n\n"
            f"Provide a structured post-mortem in the following 4-bullet format:\n"
            f"1. FORECASTING ERROR: [exact discrepancy numbers]\n"
            f"2. FAILED HYPOTHESIS: [what market dynamics or technical indicators were misinterpreted]\n"
            f"3. ALPHA DRAG: [underperformance metric]\n"
            f"4. CORRECTIVE DIRECTIVE: [strict rule for future runs with similar technical setups]\n"
        )

        critique_text = self._call_llm(prompt)
        if not critique_text:
            critique_text = self._deterministic_critique(
                ticker=ticker,
                trade_date=trade_date,
                action=predicted_action,
                pred_ret=predicted_return_pct,
                real_ret=realized_return_pct,
                alpha=alpha_pct,
                loss=regressive_loss,
            )

        lesson_record = {
            "ticker": ticker,
            "trade_date": trade_date,
            "predicted_action": predicted_action,
            "predicted_return_pct": predicted_return_pct,
            "realized_return_pct": realized_return_pct,
            "alpha_pct": alpha_pct,
            "regressive_loss": regressive_loss,
            "critique": critique_text,
            "chunk_type": "reflexion_post_mortem",
        }

        self.lessons.append(lesson_record)
        self.save()

        # Ingest directly into RAG Vector Store as a high-priority lesson learned
        self._feed_into_rag(lesson_record)

        return lesson_record

    def _call_llm(self, prompt: str) -> str | None:
        try:
            url = f"{self.host}/api/generate"
            payload = json.dumps({
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 300},
            }).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except Exception:
            return None

    def _deterministic_critique(
        self,
        ticker: str,
        trade_date: str,
        action: str,
        pred_ret: float,
        real_ret: float,
        alpha: float,
        loss: float,
    ) -> str:
        """Deterministic numerical critique fallback."""
        lines = [
            f"1. FORECASTING ERROR: Target {pred_ret:+.2f}% vs Realized {real_ret:+.2f}% (Spread: {abs(pred_ret - real_ret):.2f}%, Loss: {loss:.4f})",
            f"2. FAILED HYPOTHESIS: Action {action} over-weighted short-term consolidation without accounting for volatility expansion.",
            f"3. ALPHA DRAG: Net Alpha vs SPY benchmark of {alpha:+.2f}%.",
            f"4. CORRECTIVE DIRECTIVE: When technical trend exhibits high spread, tighten stop loss to 2.5% and de-risk portfolio exposure.",
        ]
        return "\n".join(lines)

    def _feed_into_rag(self, lesson: dict[str, Any]) -> None:
        """Upsert the critique directly into the RAG memory store."""
        try:
            from aegistradesim.rag import get_rag_engine
            engine = get_rag_engine()
            rag_chunk = {
                "ticker": lesson["ticker"],
                "trade_date": lesson["trade_date"],
                "chunk_type": "reflexion_post_mortem",
                "source_path": "rl_reflexion_critic",
                "text": (
                    f"### [RL REFLEXION CRITIQUE & LESSON LEARNED FOR {lesson['ticker']} ({lesson['trade_date']})]\n"
                    f"Prior Prediction: {lesson['predicted_action']} | Target: {lesson['predicted_return_pct']:+.2f}% | "
                    f"Actual: {lesson['realized_return_pct']:+.2f}% (Alpha: {lesson['alpha_pct']:+.2f}%)\n"
                    f"{lesson['critique']}"
                ),
            }
            engine.vector_store.upsert_chunks([rag_chunk])
            logger.info("Ingested Reflexion critique for %s into RAG vector store.", lesson["ticker"])
        except Exception as e:
            logger.warning("Could not automatically inject RL critique into RAG: %s", e)

    def get_lessons_count(self) -> int:
        return len(self.lessons)

    def get_ticker_lessons(self, ticker: str) -> list[dict[str, Any]]:
        ticker = ticker.upper()
        return [item for item in self.lessons if item.get("ticker") == ticker]
