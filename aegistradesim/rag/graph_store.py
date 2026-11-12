"""Temporal and Causal Knowledge Graph Memory Layer for AegisTradeSim RAG.

Extends the graph representation to track causal relationships across sequential
reports for a company (e.g. AAPL), connecting:
(Report_t-1) -[:PRECEDED_BY]-> (Report_t)
(Report) -[:HAS_THESIS]-> (ThesisNode)
(Report) -[:HAS_CATALYST]-> (CatalystNode)
(Report) -[:TRIGGERED_INVALIDATION]-> (RiskCondition)
"""

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
    """Relational and Temporal knowledge graph connecting Tickers, Reports, Theses, and Outcomes."""

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
        thesis: str = "",
        bull_catalyst: str = "",
        bear_risk: str = "",
        invalidation: str = "",
        indicators: dict[str, Any] | None = None,
        key_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Add or update an analysis report node and connect it temporally to previous report."""
        ticker = ticker.upper()
        ticker_node = f"TICKER:{ticker}"
        report_node = f"REPORT:{ticker}_{trade_date}"
        rating_node = f"RATING:{rating.upper()}"

        metrics = key_metrics or {}
        indicators = indicators or {}

        # Look up previous report for this ticker to create temporal edge: prev -> curr
        prior_reports = self.get_ticker_timeline(ticker)
        prev_report_node = None
        if prior_reports:
            last_date = prior_reports[-1]["date"]
            if last_date != trade_date:
                prev_report_node = f"REPORT:{ticker}_{last_date}"

        node_attributes = {
            "type": "report",
            "date": trade_date,
            "ticker": ticker,
            "rating": rating.upper(),
            "summary": summary[:300],
            "thesis": thesis[:400] if thesis else summary[:300],
            "bull_catalyst": bull_catalyst[:250],
            "bear_risk": bear_risk[:250],
            "invalidation": invalidation[:250],
            "indicators": indicators,
            **metrics,
        }

        if _HAS_NETWORKX and self._graph is not None:
            self._graph.add_node(ticker_node, type="ticker", name=ticker)
            self._graph.add_node(report_node, **node_attributes)
            self._graph.add_node(rating_node, type="rating", value=rating.upper())

            self._graph.add_edge(ticker_node, report_node, relation="HAS_REPORT")
            self._graph.add_edge(report_node, rating_node, relation="RECOMMENDED")

            # Temporal causal connection
            if prev_report_node and prev_report_node in self._graph:
                self._graph.add_edge(prev_report_node, report_node, relation="PRECEDED_BY")
        else:
            # Fallback dict structure
            self._raw_data["nodes"][ticker_node] = {"type": "ticker", "name": ticker}
            self._raw_data["nodes"][report_node] = node_attributes
            self._raw_data["nodes"][rating_node] = {"type": "rating", "value": rating.upper()}

            self._add_raw_edge(ticker_node, report_node, "HAS_REPORT")
            self._add_raw_edge(report_node, rating_node, "RECOMMENDED")

            if prev_report_node and prev_report_node in self._raw_data["nodes"]:
                self._add_raw_edge(prev_report_node, report_node, "PRECEDED_BY")

        self.save()

    def add_failure_mode_node(
        self,
        ticker: str,
        trade_date: str,
        loss: float,
        critique: str,
    ) -> None:
        """Connect a report node to a high-priority RL failure mode node."""
        ticker = ticker.upper()
        report_node = f"REPORT:{ticker}_{trade_date}"
        failure_node = f"FAILURE_MODE:{ticker}_{trade_date}"

        node_attr = {
            "type": "failure_mode",
            "date": trade_date,
            "ticker": ticker,
            "loss": loss,
            "critique": critique[:300],
        }

        if _HAS_NETWORKX and self._graph is not None:
            self._graph.add_node(failure_node, **node_attr)
            if report_node in self._graph:
                self._graph.add_edge(report_node, failure_node, relation="TRIGGERED_FAILURE")
        else:
            self._raw_data["nodes"][failure_node] = node_attr
            if report_node in self._raw_data["nodes"]:
                self._add_raw_edge(report_node, failure_node, "TRIGGERED_FAILURE")

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
                        "thesis": node_info.get("thesis", ""),
                        "bull_catalyst": node_info.get("bull_catalyst", ""),
                        "bear_risk": node_info.get("bear_risk", ""),
                        "invalidation": node_info.get("invalidation", ""),
                        "indicators": node_info.get("indicators", {}),
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
                        "thesis": node_info.get("thesis", ""),
                        "bull_catalyst": node_info.get("bull_catalyst", ""),
                        "bear_risk": node_info.get("bear_risk", ""),
                        "invalidation": node_info.get("invalidation", ""),
                        "indicators": node_info.get("indicators", {}),
                    })

        reports.sort(key=lambda x: str(x["date"]))
        return reports

    def get_graph_stats(self) -> dict[str, int]:
        """Return counts of nodes, edges, and indexed reports."""
        if _HAS_NETWORKX and self._graph is not None:
            nodes_count = self._graph.number_of_nodes()
            edges_count = self._graph.number_of_edges()
            reports_count = sum(1 for _, d in self._graph.nodes(data=True) if d.get("type") == "report")
            temporal_edges = sum(1 for _, _, d in self._graph.edges(data=True) if d.get("relation") == "PRECEDED_BY")
            return {
                "nodes": nodes_count,
                "edges": edges_count,
                "reports": reports_count,
                "temporal_links": temporal_edges,
            }
        nodes_dict = self._raw_data.get("nodes", {})
        edges_list = self._raw_data.get("edges", [])
        reports_count = sum(1 for d in nodes_dict.values() if d.get("type") == "report")
        temporal_edges = sum(1 for e in edges_list if e.get("relation") == "PRECEDED_BY")
        return {
            "nodes": len(nodes_dict),
            "edges": len(edges_list),
            "reports": reports_count,
            "temporal_links": temporal_edges,
        }
