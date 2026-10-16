# AegisTradeSim: Regressive Learning & Real-World RL Feedback Loop Documentation

## 1. Overview & Mathematical Formulation

The **Regressive Learning & Real-World RL Feedback Loop** eliminates LLM agent episode amnesia. It grounds multi-agent reasoning in actual market price outcomes after trade execution, maintaining dynamic agent credibility weights and performing automated post-mortem critiques.

```
+-----------------------------------------------------------------------------------+
|                         SIMULATION & RL FEEDBACK CYCLE                            |
|                                                                                   |
|  1. Pre-Run Dynamic Agent Weighting & RAG Precedents:                             |
|     - Contextual Bandit injects credibility weights into prompt                   |
|     - RAG injects past Reflexion post-mortem critiques into agent context         |
|                                                                                   |
|  2. Agent Pipeline Decision:                                                      |
|     - Proposed Action (BUY / HOLD / SELL)                                         |
|     - Target Price & Expected Return: \hat{R}_t                                   |
|                                                                                   |
|  3. Real-World Market Tracking (Forward Horizon k = 10 Days):                     |
|     - Realized Return via yfinance: R_t                                           |
|     - Benchmark SPY Return: R_{m,t} | Alpha: \alpha_t = R_t - R_{m,t}             |
|                                                                                   |
|  4. Regressive Loss Formulation:                                                  |
|     \mathcal{L} = |\hat{R}_t - R_t| + \lambda \cdot \mathbb{I}_{(\operatorname{sgn}(\hat{R}_t) \neq \operatorname{sgn}(R_t))} |
|                                                                                   |
|  5. Reflexion Critic & Memory Feedback:                                           |
|     - If \mathcal{L} > \tau: Automated 4-point post-mortem critique generated     |
|     - Ingests critique directly into RAG Vector Store as negative constraint      |
|     - Updates Dynamic Agent Credibility Bandit weights                            |
|     - Generates 10-Line Quantitative Telemetry Report in aegistradesim/rl/reports/|
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Package Architecture (`aegistradesim/rl/`)

| File | Component | Description |
| :--- | :--- | :--- |
| [`extractor.py`](file:///mnt/NewVolume1/IIIT%20VADODARA/AI/ailabcode/AegisTradeSim/aegistradesim/rl/extractor.py) | **Decision & Target Extractor** | Parses report markdown files to extract spot price, target price, stop-loss, recommended action, and expected return $\hat{R}_t$. |
| [`outcome_tracker.py`](file:///mnt/NewVolume1/IIIT%20VADODARA/AI/ailabcode/AegisTradeSim/aegistradesim/rl/outcome_tracker.py) | **Real-World Outcome Tracker** | Uses `yfinance` to download forward prices over trading horizon $k=10$, computing realized return $R_t$, benchmark `SPY` return, and net Alpha $\alpha_t$. |
| [`regressive_loss.py`](file:///mnt/NewVolume1/IIIT%20VADODARA/AI/ailabcode/AegisTradeSim/aegistradesim/rl/regressive_loss.py) | **Regressive Loss Calculator** | Computes continuous regressive error combining return discrepancy and directional penalty $\lambda$, plus reinforcement rewards $[-1.0, +1.0]$. |
| [`bandit_weights.py`](file:///mnt/NewVolume1/IIIT%20VADODARA/AI/ailabcode/AegisTradeSim/aegistradesim/rl/bandit_weights.py) | **Agent Credibility Bandit** | Contextual Bandit updating agent reliability weights (`Market`, `Fundamentals`, `Sentiment`, `News`) using historical prediction hit rates. |
| [`critic_agent.py`](file:///mnt/NewVolume1/IIIT%20VADODARA/AI/ailabcode/AegisTradeSim/aegistradesim/rl/critic_agent.py) | **Reflexion Critic Agent** | Generates in-context post-mortems for trades and feeds them into the RAG vector store as negative constraints for future runs. |
| [`engine.py`](file:///mnt/NewVolume1/IIIT%20VADODARA/AI/ailabcode/AegisTradeSim/aegistradesim/rl/engine.py) | **Unified RL Engine Facade** | Coordinates the entire evaluation loop, singleton `get_rl_engine()`, and quantitative 10-line telemetry generator. |

---

## 3. Project-Wide Integration

### Automatic Post-Report RL Evaluation
- **Location:** [`aegistradesim/reporting.py`](file:///mnt/NewVolume1/IIIT%20VADODARA/AI/ailabcode/AegisTradeSim/aegistradesim/reporting.py) (`write_report_tree`)
- Every time a trading report is written, `get_rl_engine().evaluate_report(save_path)` is automatically triggered alongside RAG indexing.

### In-Context Agent Weight Injection
- **Location:** [`aegistradesim/graph/trading_graph.py`](file:///mnt/NewVolume1/IIIT%20VADODARA/AI/ailabcode/AegisTradeSim/aegistradesim/graph/trading_graph.py) (`_run_graph`)
- Dynamically injects agent credibility weights into `past_context` so the Portfolio Manager weights reliable agents over underperforming ones:
  ```text
  ### [RL DYNAMIC AGENT CREDIBILITY WEIGHTS]
  - Market Analyst Credibility Weight: 0.58
  - Fundamentals Analyst Credibility Weight: 0.50
  - Sentiment Analyst Credibility Weight: 0.44
  *Decision Directive:* Prioritize signals from Market Analyst over Sentiment Analyst.
  ```

### Batch Evaluation Tool
- **Location:** [`scripts/evaluate_past_reports_rl.py`](file:///mnt/NewVolume1/IIIT%20VADODARA/AI/ailabcode/AegisTradeSim/scripts/evaluate_past_reports_rl.py)
- Runs batch evaluation across all reports stored in `reports/`, updating credibility weights and populating the Reflexion memory store.

---

## 4. How to Run RL

### Does RL start automatically?
**Yes! It runs automatically.**
Whenever you run an analysis via the CLI:
```bash
python -m cli.main
```
1. Credibility weights and past Reflexion critiques are **automatically retrieved and injected**.
2. When the report finishes, the RL engine **automatically tracks the market outcome, computes regressive loss, updates agent weights, and writes the audit report**.

### Manual Batch Evaluation of Past Reports
To evaluate all historical reports stored in `reports/`:
```bash
./venv/bin/python scripts/evaluate_past_reports_rl.py
```

---

## 5. Quantitative Telemetry & Proof of RL (For Evaluation / Professor)

Every RL evaluation produces a separate **10-line numerical audit report** in `aegistradesim/rl/reports/`.

### Storage Location
- Ticker-stamped archive: `aegistradesim/rl/reports/{TICKER}_{TIMESTAMP}_rl_telemetry.md`
- Latest run pointer: `aegistradesim/rl/reports/latest_rl_telemetry.md`

### Strict 10-Line Numerical Format
```markdown
# RL TELEMETRY AUDIT: TSLA [2026-10-15]
1. RL_PREDICTED_RETURN_PCT: +0.00%
2. REAL_WORLD_REALIZED_RETURN_PCT: -2.13%
3. BENCHMARK_SPY_RETURN_PCT: -1.11%
4. GENERATED_ALPHA_PCT: -1.02%
5. REGRESSIVE_ERROR_LOSS: 0.0213
6. DIRECTIONAL_ACCURACY_SCORE: 1.00
7. REFLEXION_CRITIQUE_GENERATED: 0
8. AGENT_CREDIBILITY_MARKET_WEIGHT: 0.55
9. AGENT_CREDIBILITY_SENTIMENT_WEIGHT: 0.51
10. TOTAL_RL_LESSONS_STORED: 5
```

### Terminal Banner Output
```text
=======================================================
📈 RL QUANTITATIVE TELEMETRY GENERATED -> aegistradesim/rl/reports/
   1. RL_PREDICTED_RETURN_PCT: +0.00%
   2. REAL_WORLD_REALIZED_RETURN_PCT: -2.13%
   3. BENCHMARK_SPY_RETURN_PCT: -1.11%
   4. GENERATED_ALPHA_PCT: -1.02%
   5. REGRESSIVE_ERROR_LOSS: 0.0213
   6. DIRECTIONAL_ACCURACY_SCORE: 1.00
   7. REFLEXION_CRITIQUE_GENERATED: 0
   8. AGENT_CREDIBILITY_MARKET_WEIGHT: 0.55
   9. AGENT_CREDIBILITY_SENTIMENT_WEIGHT: 0.51
   10. TOTAL_RL_LESSONS_STORED: 5
=======================================================
```
