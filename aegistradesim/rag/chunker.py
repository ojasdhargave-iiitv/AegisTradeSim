"""Semantic section-aware report chunker for AegisTradeSim RAG."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class ReportChunker:
    """Parses complete_report.md or individual report directories into typed chunks."""

    @staticmethod
    def parse_report_dir(report_dir: Path | str) -> list[dict[str, Any]]:
        """Parse a completed report directory into semantic chunks with metadata."""
        report_dir = Path(report_dir)
        complete_file = report_dir / "complete_report.md"

        if not complete_file.exists():
            return []

        text = complete_file.read_text(encoding="utf-8")
        folder_name = report_dir.name
        
        # Robust ticker and date extraction from folder name: e.g. TSLA_20261015_210644 or NVDA-20260917
        clean_name = folder_name.replace("-", "_")
        parts = clean_name.split("_")
        ticker = parts[0].upper()

        date_match = re.search(r"(\d{4}[-_]?\d{2}[-_]?\d{2})", folder_name)
        trade_date = date_match.group(1).replace("_", "-") if date_match else "unknown"

        chunks: list[dict[str, Any]] = []

        # Split report into markdown sections
        sections = re.split(r"\n##\s+", text)
        for sec in sections:
            lines = sec.strip().splitlines()
            if not lines:
                continue

            section_title = lines[0].replace("#", "").strip()
            body = "\n".join(lines[1:]).strip()
            if not body:
                continue

            # Classify chunk category
            chunk_type = "general"
            title_lower = section_title.lower()
            if "analyst" in title_lower:
                chunk_type = "analyst_reports"
            elif "research" in title_lower or "debate" in title_lower:
                chunk_type = "research_debate"
            elif "trading" in title_lower or "trader" in title_lower:
                chunk_type = "trading_plan"
            elif "risk" in title_lower:
                chunk_type = "risk_management"
            elif "portfolio" in title_lower or "decision" in title_lower:
                chunk_type = "final_decision"

            chunks.append({
                "chunk_id": f"{ticker}_{trade_date}_{chunk_type}",
                "ticker": ticker,
                "trade_date": trade_date,
                "section": section_title,
                "chunk_type": chunk_type,
                "text": f"### [{ticker} | {trade_date} | {section_title}]\n{body}",
                "source_path": str(report_dir),
            })

        return chunks
