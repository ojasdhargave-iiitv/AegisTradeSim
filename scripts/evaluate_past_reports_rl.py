#!/usr/bin/env python3
"""Batch evaluate existing historical reports using the RL Regressive Learning Loop."""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from aegistradesim.rl import get_rl_engine


def main():
    reports_base = root_dir / "reports"
    if not reports_base.exists():
        print("No reports/ directory found.")
        return

    report_dirs = [p for p in reports_base.iterdir() if p.is_dir() and not p.name.startswith(".")]
    report_dirs.sort()

    print(f"=======================================================")
    print(f"🚀 INGESTING & EVALUATING {len(report_dirs)} REPORTS IN RL ENGINE")
    print(f"=======================================================\n")

    rl_engine = get_rl_engine()
    total_evaluated = 0

    for rep in report_dirs:
        print(f"[*] Processing: {rep.name} ...")
        try:
            res = rl_engine.evaluate_report(rep, horizon_days=10)
            total_evaluated += 1
            print(
                f"    -> Ticker: {res['ticker']} | Action: {res['action']} | "
                f"Pred: {res['predicted_return_pct']:+.1f}% | Realized: {res['realized_return_pct']:+.1f}% | "
                f"Alpha: {res['alpha_pct']:+.1f}% | Loss: {res['regressive_loss']:.4f}"
            )
        except Exception as e:
            print(f"    [!] Error evaluating {rep.name}: {e}")

    print(f"\n=======================================================")
    print(f"✅ RL EVALUATION COMPLETE: {total_evaluated} reports processed.")
    print(f"📊 Quantitative Audits saved to: aegistradesim/rl/reports/")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
