"""Export interactive HTML visualizer for AegisTradeSim Knowledge Graph & RL Telemetry."""

import json
import glob
import os
import re
from pathlib import Path

def generate_dashboard_html():
    home = Path.home()
    graph_path = home / ".aegistradesim" / "rag_store" / "rag_graph.json"
    vector_path = home / ".aegistradesim" / "rag_store" / "vectors" / "fallback_vectors.json"
    base_dir = Path("/mnt/NewVolume1/IIIT VADODARA/AI/ailabcode/AegisTradeSim")

    # Load Graph
    graph_data = {"nodes": {}, "edges": []}
    if graph_path.exists():
        try:
            graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    raw_nodes = graph_data.get("nodes", {})
    raw_edges = graph_data.get("edges", [])

    # Nodes for vis.js
    vis_nodes = []
    if isinstance(raw_nodes, dict):
        for node_id, attrs in raw_nodes.items():
            ntype = attrs.get("type", "entity")
            label = attrs.get("name", attrs.get("ticker", node_id))
            if ntype == "ticker":
                color = "#0ea5e9"
                shape = "box"
                size = 28
            elif ntype == "report":
                rating = attrs.get("rating", "NEUTRAL")
                color = "#22c55e" if "BUY" in rating else "#ef4444" if "SELL" in rating else "#eab308"
                shape = "dot"
                size = 22
                label = f"{attrs.get('ticker', '')}\n{attrs.get('date', '')}"
            elif ntype == "rating":
                val = attrs.get("value", node_id)
                color = "#10b981" if "BUY" in val else "#f43f5e" if "SELL" in val else "#fbbf24"
                shape = "diamond"
                size = 24
            else:
                color = "#a855f7"
                shape = "ellipse"
                size = 18

            tooltip = f"<b>{node_id}</b><br>Type: {ntype}<br>"
            for k, v in attrs.items():
                if k not in ["type", "name"]:
                    tooltip += f"{k}: {v}<br>"

            vis_nodes.append({
                "id": node_id,
                "label": label,
                "color": color,
                "shape": shape,
                "size": size,
                "title": tooltip,
                "font": {"color": "#ffffff", "face": "Inter, sans-serif"},
            })

    # Edges for vis.js
    vis_edges = []
    for e in raw_edges:
        vis_edges.append({
            "from": e.get("source"),
            "to": e.get("target"),
            "label": e.get("relation", "CONNECTS"),
            "arrows": "to",
            "color": {"color": "#64748b", "highlight": "#38bdf8"},
            "font": {"color": "#94a3b8", "size": 11, "align": "middle"},
        })

    # Load Vectors
    vector_data = {}
    if vector_path.exists():
        try:
            vector_data = json.loads(vector_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    chunks = list(vector_data.values())

    # Load RL Telemetry
    rl_files = sorted(glob.glob(str(base_dir / "aegistradesim" / "rl" / "reports" / "*_rl_telemetry.md")), reverse=True)
    rl_logs = []
    for f in rl_files:
        if "latest" in f:
            continue
        p = Path(f)
        txt = p.read_text(encoding="utf-8")
        entry = {"filename": p.name}
        for l in txt.splitlines():
            m = re.match(r"^\d+\.\s+([A-Z_]+):\s+(.*)$", l.strip())
            if m:
                entry[m.group(1)] = m.group(2)
            elif l.startswith("# RL TELEMETRY AUDIT:"):
                entry["header"] = l.replace("# RL TELEMETRY AUDIT:", "").strip()
        rl_logs.append(entry)

    # Load RAG Telemetry
    rag_files = sorted(glob.glob(str(base_dir / "aegistradesim" / "rag" / "reports" / "*_rag_telemetry.md")), reverse=True)
    rag_logs = []
    for f in rag_files:
        if "latest" in f:
            continue
        p = Path(f)
        txt = p.read_text(encoding="utf-8")
        entry = {"filename": p.name}
        for l in txt.splitlines():
            m = re.match(r"^\d+\.\s+([A-Z_]+):\s+(.*)$", l.strip())
            if m:
                entry[m.group(1)] = m.group(2)
            elif l.startswith("# RAG TELEMETRY AUDIT:"):
                entry["header"] = l.replace("# RAG TELEMETRY AUDIT:", "").strip()
        rag_logs.append(entry)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AegisTradeSim | Professor Demo Visualizer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: rgba(22, 29, 47, 0.7);
      --border: rgba(255, 255, 255, 0.08);
      --primary: #38bdf8;
      --success: #4ade80;
      --warning: #facc15;
      --danger: #f87171;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    header {{
      background: rgba(15, 23, 42, 0.95);
      border-bottom: 1px solid var(--border);
      padding: 16px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
      backdrop-filter: blur(8px);
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 20px;
      font-weight: 700;
      color: var(--primary);
    }}
    .nav-tabs {{
      display: flex;
      gap: 12px;
    }}
    .tab-btn {{
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .tab-btn.active, .tab-btn:hover {{
      background: rgba(56, 189, 248, 0.15);
      border-color: var(--primary);
      color: var(--primary);
    }}
    main {{
      padding: 28px 32px;
      flex: 1;
      max-width: 1440px;
      margin: 0 auto;
      width: 100%;
    }}
    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; }}
    
    .stats-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .stat-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px 20px;
    }}
    .stat-label {{
      font-size: 12px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
    }}
    .stat-val {{
      font-size: 26px;
      font-weight: 700;
      color: var(--primary);
    }}
    
    #network-canvas {{
      width: 100%;
      height: 600px;
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border);
      border-radius: 12px;
      margin-bottom: 24px;
    }}
    
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
      background: var(--card-bg);
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid var(--border);
    }}
    th, td {{
      padding: 12px 16px;
      text-align: left;
      font-size: 13px;
      border-bottom: 1px solid var(--border);
    }}
    th {{
      background: rgba(30, 41, 59, 0.8);
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.05em;
    }}
    tr:hover td {{
      background: rgba(255, 255, 255, 0.02);
    }}
    
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
    }}
    .badge-buy {{ background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }}
    .badge-sell {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
    .badge-hold {{ background: rgba(234, 179, 8, 0.15); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.4); }}
    .badge-ticker {{ background: rgba(14, 165, 233, 0.15); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.4); }}

    .chunk-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 12px;
    }}
    .chunk-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }}
    .chunk-body {{
      font-size: 13px;
      color: #cbd5e1;
      white-space: pre-wrap;
      line-height: 1.5;
      max-height: 180px;
      overflow-y: auto;
      background: rgba(0, 0, 0, 0.2);
      padding: 12px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }}
    
    .sim-panel {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      margin-top: 24px;
    }}
    .sim-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 20px;
      margin-bottom: 20px;
    }}
    label {{
      display: block;
      font-size: 13px;
      color: var(--text-muted);
      margin-bottom: 8px;
    }}
    input, select {{
      width: 100%;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--border);
      padding: 10px 14px;
      border-radius: 8px;
      color: var(--text);
      font-size: 14px;
    }}
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <span>🛡️ AegisTradeSim</span>
      <span style="font-size: 13px; color: var(--text-muted); font-weight: 400;">| Professor Demo & Visualizer</span>
    </div>
    <div class="nav-tabs">
      <button class="tab-btn active" onclick="openTab('tab-graph')">🌐 Knowledge Graph & Relations</button>
      <button class="tab-btn" onclick="openTab('tab-rag')">📚 RAG Chunker & ChromaDB</button>
      <button class="tab-btn" onclick="openTab('tab-rl')">🎯 Reinforcement Learning & Loss</button>
      <button class="tab-btn" onclick="openTab('tab-telemetry')">📊 Live Telemetry Stream</button>
    </div>
  </header>

  <main>
    <!-- TAB 1: KNOWLEDGE GRAPH -->
    <div id="tab-graph" class="tab-content active">
      <div class="stats-row">
        <div class="stat-card">
          <div class="stat-label">Total Graph Nodes</div>
          <div class="stat-val">{len(raw_nodes)}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Relational Edges</div>
          <div class="stat-val">{len(raw_edges)}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Graph Tickers</div>
          <div class="stat-val">{sum(1 for n, d in raw_nodes.items() if d.get('type') == 'ticker' or 'TICKER:' in n)}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Historical Reports</div>
          <div class="stat-val">{sum(1 for n, d in raw_nodes.items() if d.get('type') == 'report' or 'REPORT:' in n)}</div>
        </div>
      </div>

      <div style="margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
        <h2 style="font-size: 18px;">Physics-Based Multi-Agent Graph Memory</h2>
        <span style="font-size: 13px; color: var(--text-muted);">Click & drag nodes • Scroll to zoom • Hover for metadata</span>
      </div>

      <div id="network-canvas"></div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div>
          <h3 style="font-size: 16px; margin-bottom: 12px;">Graph Nodes Registry</h3>
          <table>
            <thead>
              <tr>
                <th>Node ID</th>
                <th>Type</th>
                <th>Attribute</th>
              </tr>
            </thead>
            <tbody>
              {''.join(f"<tr><td><b>{k}</b></td><td><span class='badge'>{v.get('type', '-')}</span></td><td>{v.get('rating', v.get('name', v.get('date', '-')))}</td></tr>" for k, v in list(raw_nodes.items())[:25])}
            </tbody>
          </table>
        </div>

        <div>
          <h3 style="font-size: 16px; margin-bottom: 12px;">Relational Edges & Causal Links</h3>
          <table>
            <thead>
              <tr>
                <th>Source</th>
                <th>Relation</th>
                <th>Target</th>
              </tr>
            </thead>
            <tbody>
              {''.join(f"<tr><td><b>{e.get('source')}</b></td><td><span class='badge badge-buy'>{e.get('relation')}</span></td><td>{e.get('target')}</td></tr>" for e in raw_edges[:25])}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 2: RAG CHUNKS -->
    <div id="tab-rag" class="tab-content">
      <div class="stats-row">
        <div class="stat-card">
          <div class="stat-label">Total Ingested Chunks</div>
          <div class="stat-val">{len(chunks)}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Vector Dimension</div>
          <div class="stat-val">256-dim</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Distance Metric</div>
          <div class="stat-val">Cosine</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Storage Backend</div>
          <div class="stat-val">ChromaDB / JSON</div>
        </div>
      </div>

      <h2 style="font-size: 18px; margin-bottom: 16px;">Semantic Report Chunks ({len(chunks)} Ingested)</h2>
      
      {''.join(f"""
      <div class='chunk-card'>
        <div class='chunk-header'>
          <div>
            <span class='badge badge-ticker'>{c.get('ticker', 'N/A')}</span>
            <span class='badge badge-hold'>{c.get('chunk_type', 'general')}</span>
            <b style='margin-left: 8px;'>{c.get('chunk_id')}</b>
          </div>
          <span style='font-size: 12px; color: var(--text-muted);'>Date: {c.get('trade_date', 'N/A')}</span>
        </div>
        <div class='chunk-body'>{c.get('text', '')}</div>
      </div>
      """ for c in chunks[:15])}
    </div>

    <!-- TAB 3: RL & LOSS -->
    <div id="tab-rl" class="tab-content">
      <div class="stats-row">
        <div class="stat-card">
          <div class="stat-label">Latest Realized Return</div>
          <div class="stat-val" style="color: var(--success);">{rl_logs[0].get('REAL_WORLD_REALIZED_RETURN_PCT', 'N/A') if rl_logs else 'N/A'}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Benchmark (SPY) Return</div>
          <div class="stat-val">{rl_logs[0].get('BENCHMARK_SPY_RETURN_PCT', 'N/A') if rl_logs else 'N/A'}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Generated Alpha</div>
          <div class="stat-val" style="color: var(--primary);">{rl_logs[0].get('GENERATED_ALPHA_PCT', 'N/A') if rl_logs else 'N/A'}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Regressive Loss</div>
          <div class="stat-val">{rl_logs[0].get('REGRESSIVE_ERROR_LOSS', 'N/A') if rl_logs else 'N/A'}</div>
        </div>
      </div>

      <h2 style="font-size: 18px; margin-bottom: 14px;">Realized Market Evaluation Records</h2>
      <table>
        <thead>
          <tr>
            <th>Audit Log</th>
            <th>Predicted Return</th>
            <th>Realized Return</th>
            <th>SPY Benchmark</th>
            <th>Alpha</th>
            <th>Regressive Loss</th>
            <th>Directional Score</th>
            <th>Market Weight</th>
          </tr>
        </thead>
        <tbody>
          {''.join(f"""
          <tr>
            <td><b>{r.get('header', r.get('filename'))}</b></td>
            <td>{r.get('RL_PREDICTED_RETURN_PCT', '-')}</td>
            <td style='color: var(--success); font-weight: 600;'>{r.get('REAL_WORLD_REALIZED_RETURN_PCT', '-')}</td>
            <td>{r.get('BENCHMARK_SPY_RETURN_PCT', '-')}</td>
            <td style='color: var(--primary); font-weight: 600;'>{r.get('GENERATED_ALPHA_PCT', '-')}</td>
            <td>{r.get('REGRESSIVE_ERROR_LOSS', '-')}</td>
            <td><span class='badge badge-buy'>{r.get('DIRECTIONAL_ACCURACY_SCORE', '-')}</span></td>
            <td>{r.get('AGENT_CREDIBILITY_MARKET_WEIGHT', '-')}</td>
          </tr>
          """ for r in rl_logs)}
        </tbody>
      </table>

      <!-- Interactive Playground for Professor -->
      <div class="sim-panel">
        <h3 style="font-size: 16px; margin-bottom: 8px;">🧪 Live RL Loss & Feedback Calculator (Demonstration)</h3>
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 18px;">Show your professor how the continuous loss and reward penalizes directional errors and computes bandit updates.</p>
        
        <div class="sim-grid">
          <div>
            <label>Predicted Return (%)</label>
            <input type="number" id="sim-pred" value="5.0" step="0.5" oninput="runSim()">
          </div>
          <div>
            <label>Realized Return (%)</label>
            <input type="number" id="sim-real" value="2.8" step="0.5" oninput="runSim()">
          </div>
          <div>
            <label>Agent Action</label>
            <select id="sim-act" onchange="runSim()">
              <option value="BUY">BUY</option>
              <option value="SELL">SELL</option>
              <option value="HOLD">HOLD</option>
            </select>
          </div>
        </div>

        <div class="stats-row" style="margin-bottom: 0;">
          <div class="stat-card">
            <div class="stat-label">Regressive Loss</div>
            <div class="stat-val" id="res-loss">0.0220</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Directional Match</div>
            <div class="stat-val" id="res-dir" style="color: var(--success);">100%</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Reflexion Post-Mortem</div>
            <div class="stat-val" id="res-critic" style="font-size: 18px; color: var(--text-muted);">PASS (No Error)</div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: TELEMETRY STREAM -->
    <div id="tab-telemetry" class="tab-content">
      <h2 style="font-size: 18px; margin-bottom: 16px;">Mathematical Telemetry Stream</h2>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div>
          <h3 style="font-size: 16px; margin-bottom: 12px;">RAG Telemetry Logs</h3>
          {''.join(f"""
          <div class='chunk-card'>
            <b style='color: var(--primary);'>{r.get('header', r.get('filename'))}</b>
            <div style='margin-top: 8px; font-size: 13px;'>
              {''.join(f'<div><b>{k}</b>: {v}</div>' for k, v in r.items() if k not in ['filename', 'header'])}
            </div>
          </div>
          """ for r in rag_logs)}
        </div>
        <div>
          <h3 style="font-size: 16px; margin-bottom: 12px;">RL Telemetry Logs</h3>
          {''.join(f"""
          <div class='chunk-card'>
            <b style='color: var(--success);'>{r.get('header', r.get('filename'))}</b>
            <div style='margin-top: 8px; font-size: 13px;'>
              {''.join(f'<div><b>{k}</b>: {v}</div>' for k, v in r.items() if k not in ['filename', 'header'])}
            </div>
          </div>
          """ for r in rl_logs)}
        </div>
      </div>
    </div>

  </main>

  <script type="text/javascript">
    function openTab(tabId) {{
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      event.currentTarget.classList.add('active');
    }}

    // Vis.js Network rendering
    var nodes = new vis.DataSet({json.dumps(vis_nodes)});
    var edges = new vis.DataSet({json.dumps(vis_edges)});
    var container = document.getElementById('network-canvas');
    var data = {{ nodes: nodes, edges: edges }};
    var options = {{
      physics: {{
        stabilization: true,
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {{
          gravitationalConstant: -60,
          centralGravity: 0.015,
          springLength: 120,
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

    // Interactive simulator
    function runSim() {{
      var pred = parseFloat(document.getElementById('sim-pred').value) || 0;
      var real = parseFloat(document.getElementById('sim-real').value) || 0;
      var act = document.getElementById('sim-act').value;
      
      var err = Math.abs(pred - real) / 100.0;
      var loss = Math.pow(err, 2);
      var match = (pred * real >= 0);
      if ((act === "BUY" && real < 0) || (act === "SELL" && real > 0)) {{
        loss += 0.05;
        match = false;
      }}
      
      document.getElementById('res-loss').innerText = loss.toFixed(4);
      document.getElementById('res-dir').innerText = match ? "100%" : "0%";
      document.getElementById('res-dir').style.color = match ? "var(--success)" : "var(--danger)";
      
      var critic = document.getElementById('res-critic');
      if (!match || loss > 0.05) {{
        critic.innerText = "FAIL (Critic Triggered)";
        critic.style.color = "var(--danger)";
      }} else {{
        critic.innerText = "PASS (No Error)";
        critic.style.color = "var(--success)";
      }}
    }}
  </script>
</body>
</html>"""

    out_file = base_dir / "reports" / "interactive_dashboard.html"
    out_file.write_text(html_content, encoding="utf-8")
    print(f"Generated standalone interactive dashboard at: {out_file}")
    return out_file

if __name__ == "__main__":
    generate_dashboard_html()
