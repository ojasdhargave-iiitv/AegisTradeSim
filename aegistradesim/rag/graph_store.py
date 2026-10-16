"""Relational Graph Memory Layer for Graph-RAG in AegisTradeSim."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False


class ReportGraphStore:
    """Relational knowledge graph connecting Tickers, Dates, Indicators, and Outcomes."""

    def __init__(self, persist_path: str | Path | None = None):
        if persist_path is None:
            home = os.path.expanduser("~")
            persist_path = Path(home) / ".aegistradesim" / "rag_store" / "rag_graph.json"
        self.persist_path = Path(persist_path).expanduser()
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        self._graph: Any = nx.DiGraph() if _HAS_NETWORKX else None
        self._raw_data: dict[str, Any] = {"nodes": {}, "edges": []}
        self._load()

    def _load(self) -> None:
        if not self.persist_path.exists():
            return
        try:
            content = self.persist_path.read_text(encoding="utf-8")
            data = json.loads(content)
            self._raw_data = data
            if _HAS_NETWORKX:
                self._graph = nx.node_link_graph(data, edges="edges")
        except Exception as e:
            logger.warning("Failed to load graph (%s); starting fresh.", e)
            if _HAS_NETWORKX:
                self._graph = nx.DiGraph()
            self._raw_data = {"nodes": {}, "edges": []}

    def save(self) -> None:
        """Persist graph structure to JSON."""
        try:
            if _HAS_NETWORKX and self._graph is not None:
                data = nx.node_link_data(self._graph, edges="edges")
                self._raw_data = data
            self.persist_path.write_text(json.dumps(self._raw_data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error("Failed to save graph store: %s", e)

    def add_report_node(
        self,
        ticker: str,
        trade_date: str,
        rating: str,
        summary: str = "",
        key_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Add or update an analysis report node and connect it to Ticker and Rating."""
        ticker = ticker.upper()
        ticker_node = f"TICKER:{ticker}"
        report_node = f"REPORT:{ticker}_{trade_date}"
        rating_node = f"RATING:{rating.upper()}"

        metrics = key_metrics or {}

        if _HAS_NETWORKX and self._graph is not None:
            self._graph.add_node(ticker_node, type="ticker", name=ticker)
            self._graph.add_node(
                report_node,
                type="report",
                date=trade_date,
                ticker=ticker,
                rating=rating.upper(),
                summary=summary[:300],
                **metrics,
            )
            self._graph.add_node(rating_node, type="rating", value=rating.upper())

            self._graph.add_edge(ticker_node, report_node, relation="HAS_REPORT")
            self._graph.add_edge(report_node, rating_node, relation="RECOMMENDED")
        else:
            # Fallback dict structure
            self._raw_data["nodes"][ticker_node] = {"type": "ticker", "name": ticker}
            self._raw_data["nodes"][report_node] = {
                "type": "report",
                "date": trade_date,
                "ticker": ticker,
                "rating": rating.upper(),
                "summary": summary[:300],
                **metrics,
            }
            self._raw_data["nodes"][rating_node] = {"type": "rating", "value": rating.upper()}

            self._add_raw_edge(ticker_node, report_node, "HAS_REPORT")
            self._add_raw_edge(report_node, rating_node, "RECOMMENDED")

        self.save()

    def _add_raw_edge(self, src: str, tgt: str, relation: str) -> None:
        for e in self._raw_data["edges"]:
            if e.get("source") == src and e.get("target") == tgt and e.get("relation") == relation:
                return
        self._raw_data["edges"].append({"source": src, "target": tgt, "relation": relation})

    def get_ticker_timeline(self, ticker: str) -> list[dict[str, Any]]:
        """Retrieve chronological history of reports, indicators, and decisions for a ticker."""
        ticker = ticker.upper()
        ticker_node = f"TICKER:{ticker}"
        reports: list[dict[str, Any]] = []

        if _HAS_NETWORKX and self._graph is not None:
            if ticker_node not in self._graph:
                return []
            for _, rep_node, data in self._graph.out_edges(ticker_node, data=True):
                if data.get("relation") == "HAS_REPORT":
                    node_info = dict(self._graph.nodes[rep_node])
                    reports.append({
                        "date": node_info.get("date", "unknown"),
                        "rating": node_info.get("rating", "HOLD"),
                        "summary": node_info.get("summary", ""),
                    })
        else:
            for edge in self._raw_data.get("edges", []):
                if edge.get("source") == ticker_node and edge.get("relation") == "HAS_REPORT":
                    tgt = edge.get("target")
                    node_info = self._raw_data.get("nodes", {}).get(tgt, {})
                    reports.append({
                        "date": node_info.get("date", "unknown"),
                        "rating": node_info.get("rating", "HOLD"),
                        "summary": node_info.get("summary", ""),
                    })

        reports.sort(key=lambda x: str(x["date"]))
        return reports

    def get_graph_stats(self) -> dict[str, int]:
        """Return counts of nodes, edges, and indexed reports."""
        if _HAS_NETWORKX and self._graph is not None:
            nodes_count = self._graph.number_of_nodes()
            edges_count = self._graph.number_of_edges()
            reports_count = sum(1 for _, d in self._graph.nodes(data=True) if d.get("type") == "report")
            return {
                "nodes": nodes_count,
                "edges": edges_count,
                "reports": reports_count,
            }
        nodes_dict = self._raw_data.get("nodes", {})
        edges_list = self._raw_data.get("edges", [])
        reports_count = sum(1 for d in nodes_dict.values() if d.get("type") == "report")
        return {
            "nodes": len(nodes_dict),
            "edges": len(edges_list),
            "reports": reports_count,
        }

