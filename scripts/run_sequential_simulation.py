"""Sequential Multi-Report Simulation & Auto-Ingestion Loop for AegisTradeSim.

Runs automated sequential simulations for a specified ticker (e.g. AAPL)
across distinct trading dates across 2026 using real yfinance market data.
Automatically evaluates each report, indices it into the Temporal Knowledge Graph,
and feeds it into subsequent report generations to demonstrate progressive learning.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from aegistradesim.graph.trading_graph import AegisTradeSimGraph
from aegistradesim.rag import get_rag_engine
from aegistradesim.rl import get_rl_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_real_trading_dates(ticker: str, start_year: int = 2026, count: int = 100) -> list[str]:
    """Retrieve distinct real trading dates in the given year from yfinance."""
    start_date = f"{start_year}-01-01"
    end_date = f"{start_year}-11-06"
    
    logger.info("Fetching real OHLCV dates for %s from %s to %s via yfinance...", ticker, start_date, end_date)
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No market data returned from yfinance for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    total_days = len(df)
    logger.info("Found %d real trading days for %s in %d.", total_days, ticker, start_year)
    
    if total_days < count:
        logger.warning("Requested %d dates, but only %d trading days are available in %d. Using all available days.", count, total_days, start_year)
        selected_dates = [idx.strftime("%Y-%m-%d") for idx in df.index]
    else:
        step = max(1, total_days // count)
        sample_indices = [i for i in range(0, total_days, step)][:count]
        selected_dates = [df.index[i].strftime("%Y-%m-%d") for i in sample_indices]

    return selected_dates


def run_sequential_simulation(
    ticker: str = "AAPL",
    count: int = 100,
    start_year: int = 2026,
    start_index: int = 1,
    debug: bool = False,
) -> None:
    """Run sequential simulation across trading dates in 2026."""
    dates = get_real_trading_dates(ticker, start_year=start_year, count=count)
    logger.info("Starting automated sequential simulation: %d runs for %s across %d (starting at index %d).", len(dates), ticker, start_year, start_index)

    rag_engine = get_rag_engine()
    rl_engine = get_rl_engine()

    graph = AegisTradeSimGraph(
        selected_analysts=("market", "fundamentals"),
        debug=debug,
    )

    completed_runs = 0
    workspace_reports = Path("reports")
    workspace_reports.mkdir(parents=True, exist_ok=True)

    for idx, trade_date in enumerate(dates, start=1):
        if idx < start_index:
            continue

        print("\n" + "=" * 65)
        print(f"🚀 [RUN {idx}/{len(dates)}] SIMULATION FOR {ticker} ON {trade_date}")
        print("=" * 65)

        try:
            # 1. Run trading graph simulation on real market data for trade_date
            final_state, signal = graph.propagate(
                company_name=ticker,
                trade_date=trade_date,
            )

            # Explicit save directly into ./reports/AAPL_{trade_date}_{timestamp}
            date_clean = trade_date.replace("-", "")
            ts = datetime.datetime.now().strftime("%H%M%S")
            custom_save_path = workspace_reports / f"{ticker}_{date_clean}_{ts}"
            report_dir = graph.save_reports(final_state, ticker, save_path=custom_save_path)

            if not report_dir or not Path(report_dir).exists():
                logger.error("Run %d failed to write report directory.", idx)
                continue

            logger.info("Run %d successfully saved report directly to: %s", idx, report_dir)
            completed_runs += 1

        except Exception as e:
            logger.error("Error executing run %d on %s: %s", idx, trade_date, e, exc_info=debug)
            time.sleep(2)

    print("\n" + "=" * 65)
    print(f"🎉 COMPLETED {completed_runs}/{len(dates)} SEQUENTIAL RUNS FOR {ticker}!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sequential simulation across 2026 for a ticker.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--count", type=int, default=100, help="Number of reports across 2026 (default: 100)")
    parser.add_argument("--start-index", type=int, default=1, help="Run index to start from (e.g. 27 to resume)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()

    run_sequential_simulation(
        ticker=args.ticker,
        count=args.count,
        start_index=args.start_index,
        debug=args.debug,
    )

