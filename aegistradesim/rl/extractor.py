"""Decision, Target, and Entry Price Extractor for AegisTradeSim RL."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DecisionExtractor:
    """Extracts trade actions, price targets, entry price, and dates from completed reports."""

    def extract_from_report_dir(self, report_dir: Path | str) -> dict[str, Any]:
        """Parse a report directory and return structured decision and target parameters."""
        report_dir = Path(report_dir)
        if not report_dir.exists() or not report_dir.is_dir():
            raise FileNotFoundError(f"Report directory does not exist: {report_dir}")

        folder_name = report_dir.name
        ticker, date_str = self._parse_folder_name(folder_name)

        # 1. Read files if available
        portfolio_file = report_dir / "5_portfolio" / "decision.md"
        trader_file = report_dir / "3_trading" / "trader.md"
        market_file = report_dir / "1_analysts" / "market.md"
        complete_file = report_dir / "complete_report.md"

        portfolio_text = portfolio_file.read_text(encoding="utf-8") if portfolio_file.exists() else ""
        trader_text = trader_file.read_text(encoding="utf-8") if trader_file.exists() else ""
        market_text = market_file.read_text(encoding="utf-8") if market_file.exists() else ""
        complete_text = complete_file.read_text(encoding="utf-8") if complete_file.exists() else ""

        combined_text = f"{portfolio_text}\n{trader_text}\n{complete_text}"

        # 2. Extract Action / Decision
        action = self._extract_action(portfolio_text or trader_text or combined_text)

        # 3. Extract Entry Price (Spot Price)
        entry_price = self._extract_entry_price(market_text or combined_text)

        # 4. Extract Target Price and Stop Loss
        target_price, stop_loss = self._extract_target_and_stop(combined_text, entry_price, action)

        # 5. Calculate Expected Return (R_hat)
        expected_return_pct = 0.0
        if entry_price > 0 and target_price > 0:
            expected_return_pct = round(((target_price - entry_price) / entry_price) * 100.0, 2)
            if action in ("SELL", "SHORT") and expected_return_pct > 0:
                expected_return_pct = -expected_return_pct
        elif action == "BUY":
            expected_return_pct = 5.0  # standard baseline swing target
        elif action in ("SELL", "SHORT"):
            expected_return_pct = -5.0
        else:
            expected_return_pct = 0.0

        return {
            "ticker": ticker.upper(),
            "trade_date": date_str,
            "action": action,
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "expected_return_pct": expected_return_pct,
            "report_path": str(report_dir),
        }

    @staticmethod
    def _parse_folder_name(folder_name: str) -> tuple[str, str]:
        """Extract ticker and normalized date from folder name."""
        clean = folder_name.replace("-", "_")
        parts = clean.split("_")
        ticker = parts[0]
        date_raw = parts[1] if len(parts) > 1 else "unknown"

        # Format date into YYYY-MM-DD if possible
        if len(date_raw) >= 8 and date_raw[:8].isdigit():
            formatted_date = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
        else:
            formatted_date = date_raw

        return ticker, formatted_date

    @staticmethod
    def _extract_action(text: str) -> str:
        """Identify recommended action: BUY, SELL, SHORT, or HOLD."""
        text_upper = text.upper()
        if "**BUY**" in text_upper or "RATING: BUY" in text_upper or "ACTION: BUY" in text_upper:
            return "BUY"
        if "**SELL**" in text_upper or "RATING: SELL" in text_upper or "ACTION: SELL" in text_upper:
            return "SELL"
        if "**SHORT**" in text_upper or "SHORT-SELL" in text_upper:
            return "SHORT"
        if "**OVERWEIGHT**" in text_upper:
            return "BUY"
        if "**UNDERWEIGHT**" in text_upper:
            return "SELL"
        return "HOLD"

    @staticmethod
    def _extract_entry_price(text: str) -> float:
        """Extract spot/entry price from text."""
        # Common patterns: spot levels ($248.42), consolidating at $248.42, price ($248.42), etc.
        patterns = [
            r"spot\s*(?:levels|price|is)?\s*[:\-\(]*\s*\$([0-9]+\.[0-9]{2})",
            r"consolidating\s+at\s+\$([0-9]+\.[0-9]{2})",
            r"current\s+price\s*[:\-\(]*\s*\$([0-9]+\.[0-9]{2})",
            r"price\s*\(?\$([0-9]+\.[0-9]{2})\)?",
            r"\$([0-9]+\.[0-9]{2})",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                try:
                    val = float(match.group(1))
                    if 1.0 <= val <= 10000.0:
                        return val
                except ValueError:
                    continue
        return 0.0

    @staticmethod
    def _extract_target_and_stop(text: str, entry_price: float, action: str) -> tuple[float, float]:
        """Extract explicit target price and stop loss, or derive prudent defaults."""
        target_price = 0.0
        stop_loss = 0.0

        # Regex search for target and stop loss
        target_match = re.search(r"(?:target|take\s*profit)\s*(?:price)?\s*[:\-\(]*\s*\$([0-9]+\.[0-9]{2})", text, re.IGNORECASE)
        if target_match:
            try:
                target_price = float(target_match.group(1))
            except ValueError:
                pass

        stop_match = re.search(r"(?:stop\s*loss|stop)\s*(?:price)?\s*[:\-\(]*\s*\$([0-9]+\.[0-9]{2})", text, re.IGNORECASE)
        if stop_match:
            try:
                stop_loss = float(stop_match.group(1))
            except ValueError:
                pass

        # If not explicitly specified in markdown, calculate standard risk-reward targets based on entry
        if entry_price > 0:
            if target_price == 0.0:
                if action == "BUY":
                    target_price = round(entry_price * 1.05, 2)
                elif action in ("SELL", "SHORT"):
                    target_price = round(entry_price * 0.95, 2)
                else:
                    target_price = entry_price

            if stop_loss == 0.0:
                if action == "BUY":
                    stop_loss = round(entry_price * 0.97, 2)
                elif action in ("SELL", "SHORT"):
                    stop_loss = round(entry_price * 1.03, 2)
                else:
                    stop_loss = round(entry_price * 0.98, 2)

        return target_price, stop_loss
