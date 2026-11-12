"""Augmentation Synthesizer for AegisTradeSim RAG.

Synthesizes longitudinal narrative, thesis drift, causal precedents,
and agent-specific lessons across sequential reports of a stock (e.g. AAPL).
Transforms simple chunk retrieval into genuine reasoning augmentation.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AugmentationSynthesizer:
    """Produces synthesized longitudinal context and thesis evolution for agent prompt injection."""

    def synthesize_augmentation(
        self,
        ticker: str,
        current_date: str,
        timeline: list[dict[str, Any]],
        similar_chunks: list[dict[str, Any]] | None = None,
        rl_context: str = "",
    ) -> dict[str, str]:
        """Synthesize specialized augmentations tailored to each agent category.
        
        Returns a dictionary mapping agent role to its augmented context block:
        - 'longitudinal_narrative': High-level narrative for Portfolio Manager
        - 'analyst_context': Macro/trend drift for Technical & Fundamental analysts
        - 'risk_context': Downside precedents & invalidation history for Risk team
        """
        ticker = ticker.upper()
        similar_chunks = similar_chunks or []

        if not timeline:
            return {
                "longitudinal_narrative": f"No previous reports indexed yet for {ticker}. Initiating baseline thesis.",
                "analyst_context": f"Baseline initiation for {ticker}.",
                "risk_context": f"Standard risk parameters applied for {ticker}.",
                "full_injection": f"### [RAG Knowledge Initiation for {ticker}]\nFirst session for {ticker}.",
            }

        # 1. Longitudinal Thesis Evolution (Portfolio Manager)
        num_past_reports = len(timeline)
        recent_timeline = timeline[-5:]  # last 5 reports
        ratings_seq = " -> ".join([r.get("rating", "HOLD") for r in recent_timeline])
        
        narrative_lines = [
            f"### [RAG Longitudinal Thesis Evolution for {ticker} across {num_past_reports} Reports]",
            f"- **Trajectory Path (Last {len(recent_timeline)} Sessions):** {ratings_seq}",
        ]

        # Thesis drift & shifts
        prev_report = timeline[-1]
        prev_date = prev_report.get("date", "previous")
        prev_rating = prev_report.get("rating", "HOLD")
        prev_thesis = prev_report.get("thesis") or prev_report.get("summary", "")

        narrative_lines.append(f"- **Most Recent Precedent ({prev_date}):** Stood at **{prev_rating}**.")
        if prev_thesis:
            narrative_lines.append(f"  *Prior Thesis Core:* {prev_thesis[:250]}")

        # Invalidation & Risk memory
        invalidation = prev_report.get("invalidation", "")
        if invalidation:
            narrative_lines.append(f"  *Watchout / Invalidation Criterion:* {invalidation}")

        longitudinal_narrative = "\n".join(narrative_lines)

        # 2. Analyst Context (Technical & Fundamental setups comparison)
        analyst_lines = [
            f"### [RAG Historical Indicator & Setup Precedents for {ticker}]",
        ]
        
        # Check historical indicator shifts if available
        for past_rep in reversed(recent_timeline[-3:]):
            p_date = past_rep.get("date", "")
            p_ind = past_rep.get("indicators", {})
            p_cat = past_rep.get("bull_catalyst", "")
            if p_ind:
                ind_str = ", ".join([f"{k.upper()}={v}" for k, v in p_ind.items()])
                analyst_lines.append(f"- **Setup on {p_date}:** {ind_str} | Catalyst: {p_cat[:150]}")

        if len(analyst_lines) == 1:
            analyst_lines.append(f"- Historical consensus previously maintained {prev_rating} stance on {prev_date}.")

        analyst_context = "\n".join(analyst_lines)

        # 3. Risk Management Context (Downside & Failure Modes)
        risk_lines = [
            f"### [RAG Downside Precedents & Stop-Loss Memory for {ticker}]",
            f"- In past sessions, risk was triggered primarily by: {invalidation or 'Momentum deceleration and moving average breaches.'}",
        ]

        if rl_context:
            risk_lines.append(f"- **RL Reflexion Negative Constraints:**\n{rl_context}")
        
        bear_risks = [r.get("bear_risk") for r in recent_timeline if r.get("bear_risk")]
        if bear_risks:
            risk_lines.append(f"- **Historical Bear Counter-Theses:** {bear_risks[-1][:200]}")

        risk_context = "\n".join(risk_lines)

        # 4. Full Comprehensive Injection
        full_parts = [
            longitudinal_narrative,
            "\n" + analyst_context,
            "\n" + risk_context,
        ]

        if rl_context:
            full_parts.append(f"\n### [RAG-RL Synergistic Reflexion Context]\n{rl_context}")

        if similar_chunks:
            full_parts.append(f"\n### [RAG Semantic Debate Precedents]")
            for chunk in similar_chunks[:2]:
                score = chunk.get("score", 0.0)
                sec_type = chunk.get("metadata", {}).get("chunk_type", "section")
                full_parts.append(f"- ({sec_type} | match {score:.2f}):\n{chunk['text'][:300]}...")

        return {
            "longitudinal_narrative": longitudinal_narrative,
            "analyst_context": analyst_context,
            "risk_context": risk_context,
            "full_injection": "\n".join(full_parts),
        }

