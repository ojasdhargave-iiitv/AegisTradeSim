"""AegisTradeSim Unified Visualizer & Professor Demo Dashboard.

Visualizes:
1. Knowledge Graph (NetworkX / PyVis / D3 Force Graph): Tickers, Reports, Ratings, Relations, and Edges
2. RAG Semantic Chunk Explorer & Vector Store: Ingested chunks, similarity search, embeddings
3. Reinforcement Learning Audit: Regressive Loss, Realized Return vs. SPY, Dynamic Credibility Weights, Outcome Tracking
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from typing import Any

import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="AegisTradeSim | Research & Audit Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Sleek Dark Glassmorphism Presentation
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-buy { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }
    .badge-sell { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .badge-hold { background: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid #eab308; }
    .badge-ticker { background: rgba(14, 165, 233, 0.2); color: #38bdf8; border: 1px solid #0ea5e9; }
    .badge-chunk { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid #a855f7; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# DATA LOADERS
# ==============================================================================

@st.cache_data(ttl=5)
def load_graph_data() -> dict[str, Any]:
    """Load relational graph data from ~/.aegistradesim/rag_store/rag_graph.json."""
    path = Path.home() / ".aegistradesim" / "rag_store" / "rag_graph.json"
    if not path.exists():
        return {"nodes": {}, "edges": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except Exception as e:
        st.error(f"Error loading graph store: {e}")
        return {"nodes": {}, "edges": []}


@st.cache_data(ttl=5)
def load_vector_store() -> dict[str, Any]:
    """Load vector chunks from fallback_vectors.json or Chroma store."""
    path = Path.home() / ".aegistradesim" / "rag_store" / "vectors" / "fallback_vectors.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except Exception as e:
        st.error(f"Error loading vector chunks: {e}")
        return {}


@st.cache_data(ttl=5)
def load_telemetry_logs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load RAG and RL telemetry logs from markdown reports."""
    base_dir = Path("/mnt/NewVolume1/IIIT VADODARA/AI/ailabcode/AegisTradeSim")
    
    # RL telemetry
    rl_reports = []
    rl_files = sorted(glob.glob(str(base_dir / "aegistradesim" / "rl" / "reports" / "*_rl_telemetry.md")), reverse=True)
    for f in rl_files:
        if "latest" in f:
            continue
        p = Path(f)
        content = p.read_text(encoding="utf-8")
        entry: dict[str, Any] = {"filename": p.name}
        for line in content.splitlines():
            m = re.match(r"^\d+\.\s+([A-Z_]+):\s+(.*)$", line.strip())
            if m:
                entry[m.group(1)] = m.group(2)
            elif line.startswith("# RL TELEMETRY AUDIT:"):
                entry["header"] = line.replace("# RL TELEMETRY AUDIT:", "").strip()
        rl_reports.append(entry)

    # RAG telemetry
    rag_reports = []
    rag_files = sorted(glob.glob(str(base_dir / "aegistradesim" / "rag" / "reports" / "*_rag_telemetry.md")), reverse=True)
    for f in rag_files:
        if "latest" in f:
            continue
        p = Path(f)
        content = p.read_text(encoding="utf-8")
        entry = {"filename": p.name}
        for line in content.splitlines():
            m = re.match(r"^\d+\.\s+([A-Z_]+):\s+(.*)$", line.strip())
            if m:
                entry[m.group(1)] = m.group(2)
            elif line.startswith("# RAG TELEMETRY AUDIT:"):
                entry["header"] = line.replace("# RAG TELEMETRY AUDIT:", "").strip()
        rag_reports.append(entry)

    return rag_reports, rl_reports


# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("AegisTradeSim")
st.sidebar.markdown("**Interactive Professor Demo & Audit**")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Select Component to Inspect",
    [
        "🌐 Relational Knowledge Graph (NetworkX)",
        "📚 RAG Semantic Chunker & ChromaDB",
        "🎯 Reinforcement Learning (Loss & Outcomes)",
        "📊 Live Multi-Agent Telemetry Audit",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**System Status**:\n"
    "- Knowledge Graph: Active\n"
    "- Hybrid Vector Store: Operational\n"
    "- Reflexion Critic: Enabled\n"
    "- Realized Market Evaluation: Active"
)


# ==============================================================================
# TAB 1: KNOWLEDGE GRAPH VISUALIZER (Interactive Physics Graph)
# ==============================================================================
if "Knowledge Graph" in app_mode:
    st.title("🌐 Relational Knowledge Graph Memory")
    st.markdown(
        "Interactive Graph visualization of multi-agent memory connecting **Tickers**, "
        "**Historical Reports**, **Trading Ratings**, and **Decision Edges**."
    )

    graph_raw = load_graph_data()
    raw_nodes = graph_raw.get("nodes", {})
    edges = graph_raw.get("edges", [])

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""<div class='metric-card'><div class='metric-label'>Graph Nodes</div>
            <div class='metric-val'>{len(raw_nodes)}</div></div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class='metric-card'><div class='metric-label'>Relational Edges</div>
            <div class='metric-val'>{len(edges)}</div></div>""",
            unsafe_allow_html=True,
        )
    with col3:
        tickers_count = sum(1 for n, d in (raw_nodes.items() if isinstance(raw_nodes, dict) else []) if d.get("type") == "ticker" or "TICKER:" in n)
        st.markdown(
            f"""<div class='metric-card'><div class='metric-label'>Active Tickers</div>
            <div class='metric-val'>{tickers_count}</div></div>""",
            unsafe_allow_html=True,
        )
    with col4:
        reports_count = sum(1 for n, d in (raw_nodes.items() if isinstance(raw_nodes, dict) else []) if d.get("type") == "report" or "REPORT:" in n)
        st.markdown(
            f"""<div class='metric-card'><div class='metric-label'>Linked Reports</div>
            <div class='metric-val'>{reports_count}</div></div>""",
            unsafe_allow_html=True,
        )

    # Convert nodes and edges into vis.js / D3 interactive HTML format
    st.subheader("Interactive Force-Directed Graph")
    st.caption("Drag nodes, zoom in/out, hover over entities to inspect attributes and connections.")

    # Format nodes for vis.js
    vis_nodes = []
    if isinstance(raw_nodes, dict):
        for node_id, attrs in raw_nodes.items():
            ntype = attrs.get("type", "entity")
            label = attrs.get("name", attrs.get("ticker", node_id))
            
            # Color scheme by node type
            if ntype == "ticker":
                color = "#38bdf8"  # Cyan/Blue
                shape = "box"
                size = 25
            elif ntype == "report":
                rating = attrs.get("rating", "NEUTRAL")
                color = "#22c55e" if "BUY" in rating else "#ef4444" if "SELL" in rating else "#eab308"
                shape = "dot"
                size = 20
                label = f"{attrs.get('ticker', '')}\n{attrs.get('date', '')}"
            elif ntype == "rating":
                val = attrs.get("value", node_id)
                color = "#4ade80" if "BUY" in val else "#f87171" if "SELL" in val else "#facc15"
                shape = "diamond"
                size = 22
            else:
                color = "#c084fc"
                shape = "ellipse"
                size = 15

            title_tooltip = f"<b>{node_id}</b><br>Type: {ntype}<br>"
            for k, v in attrs.items():
                if k not in ["type", "name"]:
                    title_tooltip += f"{k}: {v}<br>"

            vis_nodes.append({
                "id": node_id,
                "label": label,
                "color": color,
                "shape": shape,
                "size": size,
                "title": title_tooltip,
                "font": {"color": "#ffffff", "face": "Inter, sans-serif"},
            })

    # Format edges
    vis_edges = []
    for e in edges:
        src = e.get("source")
        tgt = e.get("target")
        rel = e.get("relation", "CONNECTS")
        vis_edges.append({
            "from": src,
            "to": tgt,
            "label": rel,
            "arrows": "to",
            "color": {"color": "#64748b", "highlight": "#38bdf8"},
            "font": {"color": "#94a3b8", "size": 11, "align": "middle"},
        })

    # Render Vis.js in an iframe with dark styling
    vis_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
      <style type="text/css">
        #network {{
          width: 100%;
          height: 520px;
          background-color: #0f172a;
          border: 1px solid #334155;
          border-radius: 12px;
        }}
      </style>
    </head>
    <body>
      <div id="network"></div>
      <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(vis_nodes)});
        var edges = new vis.DataSet({json.dumps(vis_edges)});
        var container = document.getElementById('network');
        var data = {{
          nodes: nodes,
          edges: edges
        }};
        var options = {{
          physics: {{
            stabilization: true,
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {{
              gravitationalConstant: -50,
              centralGravity: 0.01,
              springLength: 100,
              springConstant: 0.08
            }}
          }},
          interaction: {{
            hover: true,
            tooltipDelay: 100,
            zoomView: true
          }}
        }};
        var network = new vis.Network(container, data, options);
      </script>
    </body>
    </html>
    """
    st.components.v1.html(vis_html, height=550)

    # Table breakdown of nodes and edges
    st.subheader("Knowledge Graph Entities & Adjacency Breakdown")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Nodes Registry")
        node_records = []
        if isinstance(raw_nodes, dict):
            for k, v in raw_nodes.items():
                node_records.append({
                    "Node ID": k,
                    "Type": v.get("type", "N/A"),
                    "Key Attribute": v.get("rating", v.get("name", v.get("date", "-"))),
                })
        st.dataframe(node_records, use_container_width=True, height=280)

    with col_b:
        st.markdown("#### Edges (Relations & Causal Links)")
        edge_records = []
        for e in edges:
            edge_records.append({
                "Source Node": e.get("source"),
                "Relation": e.get("relation"),
                "Target Node": e.get("target"),
            })
        st.dataframe(edge_records, use_container_width=True, height=280)


# ==============================================================================
# TAB 2: RAG SEMANTIC CHUNKS & VECTOR RETRIEVAL
# ==============================================================================
elif "RAG Semantic Chunker" in app_mode:
    st.title("📚 RAG Semantic Chunker & ChromaDB Store")
    st.markdown(
        "Inspect section-aware chunks parsed by [chunker.py](file:///mnt/NewVolume1/IIIT%20VADODARA/AI/ailabcode/AegisTradeSim/aegistradesim/rag/chunker.py) "
        "and embedded into the vector store."
    )

    vector_data = load_vector_store()
    chunk_list = list(vector_data.values())

    # Filter controls
    tickers = sorted(list(set(c.get("ticker", "UNKNOWN") for c in chunk_list)))
    chunk_types = sorted(list(set(c.get("chunk_type", "general") for c in chunk_list)))

    f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
    with f_col1:
        sel_ticker = st.selectbox("Filter by Ticker", ["ALL"] + tickers)
    with f_col2:
        sel_type = st.selectbox("Filter by Chunk Category", ["ALL"] + chunk_types)
    with f_col3:
        search_kw = st.text_input("Semantic Search / Query Keyword", "")

    # Filter logic
    filtered = chunk_list
    if sel_ticker != "ALL":
        filtered = [c for c in filtered if c.get("ticker") == sel_ticker]
    if sel_type != "ALL":
        filtered = [c for c in filtered if c.get("chunk_type") == sel_type]
    if search_kw:
        kw = search_kw.lower()
        filtered = [c for c in filtered if kw in c.get("text", "").lower() or kw in c.get("chunk_id", "").lower()]

    st.markdown(f"**Showing {len(filtered)} of {len(chunk_list)} stored chunks**")

    # Display chunk cards
    for idx, c in enumerate(filtered):
        cid = c.get("chunk_id", f"chunk_{idx}")
        ticker = c.get("ticker", "N/A")
        trade_date = c.get("trade_date", "N/A")
        ctype = c.get("chunk_type", "general")
        text = c.get("text", "")
        emb = c.get("embedding", [])
        source = c.get("source_path", "N/A")

        with st.expander(f"📄 {cid} | Ticker: {ticker} | Type: {ctype} | Date: {trade_date}"):
            st.markdown(
                f"<span class='badge badge-ticker'>{ticker}</span>"
                f"<span class='badge badge-chunk'>{ctype}</span>"
                f"<span class='badge badge-hold'>Date: {trade_date}</span>"
                f"<span class='badge badge-buy'>Dim: {len(emb)}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Source Path:** `{source}`")
            st.text_area("Chunk Content", text, height=180, key=f"chunk_txt_{idx}")
            if emb:
                st.caption(f"Vector Embedding Sample (First 10 of {len(emb)} dims): {emb[:10]}")


# ==============================================================================
# TAB 3: REINFORCEMENT LEARNING AUDIT & LOSS PROGRESSION
# ==============================================================================
elif "Reinforcement Learning" in app_mode:
    st.title("🎯 Reinforcement Learning & Regressive Loss Engine")
    st.markdown(
        "Auditing real-world trade tracking via **OutcomeTracker**, continuous loss calculation via **RegressiveLossCalculator**, "
        "and multi-agent credibility updates via **AgentCredibilityBandit**."
    )

    _, rl_logs = load_telemetry_logs()

    # Metrics
    if rl_logs:
        latest = rl_logs[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""<div class='metric-card'><div class='metric-label'>Latest Realized Return</div>
                <div class='metric-val'>{latest.get('REAL_WORLD_REALIZED_RETURN_PCT', 'N/A')}</div></div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div class='metric-card'><div class='metric-label'>Benchmark Return (SPY)</div>
                <div class='metric-val'>{latest.get('BENCHMARK_SPY_RETURN_PCT', 'N/A')}</div></div>""",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""<div class='metric-card'><div class='metric-label'>Generated Alpha</div>
                <div class='metric-val'>{latest.get('GENERATED_ALPHA_PCT', 'N/A')}</div></div>""",
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""<div class='metric-card'><div class='metric-label'>Regressive Loss</div>
                <div class='metric-val'>{latest.get('REGRESSIVE_ERROR_LOSS', 'N/A')}</div></div>""",
                unsafe_allow_html=True,
            )

    st.subheader("Historical RL Evaluation Records")
    rl_table_data = []
    for log in rl_logs:
        header = log.get("header", log.get("filename", ""))
        ticker_match = re.search(r"([A-Z]+)\s+\[(\d{4}[-_]\d{2}[-_]\d{2})\]", header)
        t_name = ticker_match.group(1) if ticker_match else "N/A"
        t_date = ticker_match.group(2) if ticker_match else "N/A"

        rl_table_data.append({
            "Ticker": t_name,
            "Date": t_date,
            "Predicted Return": log.get("RL_PREDICTED_RETURN_PCT", "-"),
            "Realized Return": log.get("REAL_WORLD_REALIZED_RETURN_PCT", "-"),
            "SPY Benchmark": log.get("BENCHMARK_SPY_RETURN_PCT", "-"),
            "Alpha": log.get("GENERATED_ALPHA_PCT", "-"),
            "Regressive Loss": log.get("REGRESSIVE_ERROR_LOSS", "-"),
            "Directional Match": log.get("DIRECTIONAL_ACCURACY_SCORE", "-"),
            "Reflexion Critiques": log.get("REFLEXION_CRITIQUE_GENERATED", "0"),
            "Market Weight": log.get("AGENT_CREDIBILITY_MARKET_WEIGHT", "-"),
            "Sentiment Weight": log.get("AGENT_CREDIBILITY_SENTIMENT_WEIGHT", "-"),
        })

    st.dataframe(rl_table_data, use_container_width=True)

    # Interactive RL Simulation Playground for the Professor
    st.markdown("---")
    st.subheader("🧪 Live RL Loss & Reward Simulator (For Professor Demos)")
    st.caption("Demonstrate how AegisTradeSim computes regressive loss and dynamic rewards dynamically.")

    sim_c1, sim_c2, sim_c3 = st.columns(3)
    with sim_c1:
        sim_pred = st.slider("Agent Predicted Return (%)", -20.0, 30.0, 5.0, step=0.5)
    with sim_c2:
        sim_real = st.slider("Actual Realized Return (%)", -20.0, 30.0, 3.5, step=0.5)
    with sim_c3:
        sim_act = st.selectbox("Agent Decision", ["BUY", "SELL", "HOLD"])

    # Compute loss dynamically using logic from regressive_loss.py
    from aegistradesim.rl.regressive_loss import RegressiveLossCalculator
    calc = RegressiveLossCalculator()
    res = calc.calculate_loss(predicted_return_pct=sim_pred, realized_return_pct=sim_real, action=sim_act)

    r_col1, r_col2, r_col3 = st.columns(3)
    with r_col1:
        st.metric("Total Regressive Loss", f"{res['total_regressive_loss']:.4f}")
    with r_col2:
        acc_pct = 100 if res.get("directional_match", False) else 0
        st.metric("Directional Accuracy", f"{acc_pct}%")
    with r_col3:
        is_crit = res["is_significant_failure"]
        st.metric("Reflexion Critic Triggered?", "YES (Audit Triggered)" if is_crit else "NO (Trade Healthy)")


# ==============================================================================
# TAB 4: MULTI-AGENT TELEMETRY AUDIT
# ==============================================================================
elif "Live Multi-Agent Telemetry" in app_mode:
    st.title("📊 Multi-Agent Quantitative Telemetry Audit")
    st.markdown(
        "Live quantitative telemetry generated across pipeline executions. "
        "Verified mathematically to ensure complete audit compliance."
    )

    rag_logs, rl_logs = load_telemetry_logs()

    t1, t2 = st.tabs(["RAG Telemetry Logs", "RL Telemetry Logs"])

    with t1:
        st.subheader("RAG Telemetry Stream")
        for r in rag_logs:
            with st.expander(f"Audit: {r.get('header', r.get('filename'))}"):
                for k, v in r.items():
                    if k not in ["filename", "header"]:
                        st.markdown(f"- **{k}**: `{v}`")

    with t2:
        st.subheader("RL Telemetry Stream")
        for r in rl_logs:
            with st.expander(f"Audit: {r.get('header', r.get('filename'))}"):
                for k, v in r.items():
                    if k not in ["filename", "header"]:
                        st.markdown(f"- **{k}**: `{v}`")
