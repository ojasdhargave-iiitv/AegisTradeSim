"""Thesis and Causal Factor Extractor for AegisTradeSim RAG.

Extracts structured investment thesis, technical indicators, catalyst assumptions,
and invalidation triggers from reports to build causal knowledge graph nodes.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ThesisExtractor:
    """Extracts structured investment hypotheses and causal relations from reports."""

    def extract_from_report_dir(self, report_dir: Path | str) -> dict[str, Any]:
        """Parse completed report into structured investment thesis nodes."""
        report_dir = Path(report_dir)
        complete_file = report_dir / "complete_report.md"
        
        text = complete_file.read_text(encoding="utf-8") if complete_file.exists() else ""
        folder_name = report_dir.name

        clean_name = folder_name.replace("-", "_")
        parts = clean_name.split("_")
        ticker = parts[0].upper()

        date_match = re.search(r"(\d{4}[-_]?\d{2}[-_]?\d{2})", folder_name)
        trade_date = date_match.group(1).replace("_", "-") if date_match else "unknown"

        # 1. Rating & Target extraction
        rating = "HOLD"
        rating_match = re.search(r"\*\*Rating\*\*:\s*([A-Za-z]+)", text, re.IGNORECASE)
        if rating_match:
            rating = rating_match.group(1).upper()
        elif "**BUY**" in text or "FINAL TRANSACTION PROPOSAL: **BUY**" in text:
            rating = "BUY"
        elif "**SELL**" in text or "FINAL TRANSACTION PROPOSAL: **SELL**" in text:
            rating = "SELL"

        target_price = 0.0
        target_match = re.search(r"\*\*Price Target\*\*:\s*([0-9\.]+)", text, re.IGNORECASE)
        if target_match:
            try:
                target_price = float(target_match.group(1))
            except ValueError:
                pass

        # 2. Extract Key Indicators (RSI, MACD, SMAs)
        indicators: dict[str, float] = {}
        rsi_match = re.search(r"RSI.*?(\d{1,3}(?:\.\d+)?)", text, re.IGNORECASE)
        if rsi_match:
            try:
                indicators["rsi"] = float(rsi_match.group(1))
            except ValueError:
                pass

        sma200_match = re.search(r"(?:200[ -]day|close_200_sma).*?\$?([0-9]+\.[0-9]{2})", text, re.IGNORECASE)
        if sma200_match:
            try:
                indicators["sma_200"] = float(sma200_match.group(1))
            except ValueError:
                pass

        sma50_match = re.search(r"(?:50[ -]day|close_50_sma).*?\$?([0-9]+\.[0-9]{2})", text, re.IGNORECASE)
        if sma50_match:
            try:
                indicators["sma_50"] = float(sma50_match.group(1))
            except ValueError:
                pass

        atr_match = re.search(r"(?:ATR|Average True Range).*?\$?([0-9]+\.[0-9]{2})", text, re.IGNORECASE)
        if atr_match:
            try:
                indicators["atr"] = float(atr_match.group(1))
            except ValueError:
                pass

        # 3. Extract Core Thesis Statements
        thesis_text = ""
        thesis_match = re.search(r"\*\*Investment Thesis\*\*:\s*(.*?)(?=\n\n|\n\*\*|\Z)", text, re.DOTALL | re.IGNORECASE)
        if thesis_match:
            thesis_text = thesis_match.group(1).strip()
        elif "Rationale" in text:
            rat_match = re.search(r"\*\*Rationale\*\*:\s*(.*?)(?=\n\n|\n\*\*|\Z)", text, re.DOTALL | re.IGNORECASE)
            if rat_match:
                thesis_text = rat_match.group(1).strip()

        # 4. Extract Bull Catalyst vs Bear Risk
        bull_catalyst = ""
        bear_risk = ""
        
        bull_section = re.search(r"### Bull Researcher\s*(.*?)(?=### Bear Researcher|\n## |\Z)", text, re.DOTALL)
        if bull_section:
            bull_lines = [l.strip() for l in bull_section.group(1).splitlines() if len(l.strip()) > 30 and not l.strip().startswith("#")]
            if bull_lines:
                bull_catalyst = bull_lines[0][:250]

        bear_section = re.search(r"### Bear Researcher\s*(.*?)(?=### Research Manager|\n## |\Z)", text, re.DOTALL)
        if bear_section:
            bear_lines = [l.strip() for l in bear_section.group(1).splitlines() if len(l.strip()) > 30 and not l.strip().startswith("#")]
            if bear_lines:
                bear_risk = bear_lines[0][:250]

        # 5. Extract Invalidation Triggers / Stop-Loss Criteria
        invalidation = ""
        inval_match = re.search(r"(?:invalidate|stop-loss|invalidation|break below).*?([A-Z0-9\$\.\s]{20,150}\.)", text, re.IGNORECASE)
        if inval_match:
            invalidation = inval_match.group(0).strip()
        else:
            invalidation = f"Breach below 50 SMA support or negative momentum acceleration."

        return {
            "ticker": ticker,
            "trade_date": trade_date,
            "rating": rating,
            "target_price": target_price,
            "indicators": indicators,
            "thesis": thesis_text[:400] if thesis_text else "Consolidation and multi-horizon trend monitoring.",
            "bull_catalyst": bull_catalyst,
            "bear_risk": bear_risk,
            "invalidation": invalidation,
            "source_dir": str(report_dir),
        }

