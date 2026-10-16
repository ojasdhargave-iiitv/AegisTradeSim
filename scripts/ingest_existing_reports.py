"""Batch ingests all existing completed reports into the RAG Vector & Graph stores."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aegistradesim.rag import get_rag_engine


def main() -> None:
    reports_root = PROJECT_ROOT / "reports"
    if not reports_root.exists():
        print(f"Error: Reports directory not found at {reports_root}")
        return

    rag = get_rag_engine()
    report_dirs = [p for p in reports_root.iterdir() if p.is_dir()]

    if not report_dirs:
        print("No report directories found in reports/")
        return

    print(f"===========================================================")
    print(f" AegisTradeSim RAG: Ingesting {len(report_dirs)} Existing Report Folders")
    print(f"===========================================================")

    total_chunks = 0
    for r_dir in sorted(report_dirs):
        print(f"\n[+] Parsing report: {r_dir.name} ...")
        n_chunks = rag.index_report(r_dir)
        total_chunks += n_chunks
        print(f"    --> Successfully indexed {n_chunks} semantic chunks.")

    print(f"\n===========================================================")
    print(f"✓ All existing reports ingested successfully!")
    print(f"  Total semantic chunks indexed: {total_chunks}")
    print(f"===========================================================")


if __name__ == "__main__":
    main()
