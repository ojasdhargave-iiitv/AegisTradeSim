# AegisTradeSim: Comprehensive System Architecture and Execution Flow Report

## Executive Summary
This comprehensive document serves as the foundational architectural blueprint for the **AegisTradeSim** project. Based on the system's design diagram, this report details a highly sophisticated, multi-agent trading simulation system. The architecture relies on **LangGraph** for dynamic task orchestration and agent routing, utilizing a localized **Qwen 2 (1.5b)** model running via **Ollama** as the core reasoning engine. 

The system introduces a novel approach to financial analysis by establishing a rigorous debate framework between **Causal Agents** (designed to seek direct evidence) and **Counterfactual Agents** (designed to challenge assumptions and assess risk). Furthermore, the architecture is designed to be future-proofed with the integration of **Retrieval-Augmented Generation (RAG)** for unstructured data synthesis and **Reinforcement Learning (RL)** for continuous strategy optimization.

---

## 1. System Orchestration and Initialization

### 1.1 The User Interface Layer
The execution flow begins with the **User**, who sets the initial parameters, target assets (e.g., AAPL, TSLA), and the desired simulation timeframe. The user's input acts as the catalyst for the entire automated workflow, passing the initial configuration downstream to the orchestrator.

### 1.2 LangGraph Orchestrator
At the heart of the system's control flow is the **LangGraph Orchestrator**. Unlike traditional linear scripts, LangGraph creates a dynamic, stateful graph of agentic execution. 
*   **Graph Creation**: It maps out the dependencies between different tasks, ensuring that data flows logically from extraction to analysis, and finally to execution. 
*   **State Management**: It maintains the global state of the simulation, passing context between independent agents so that downstream nodes have access to all previously gathered intelligence.
*   **Causal Analysis Integration**: At this orchestration level, a preliminary **Causal Analysis** is conducted. This module assesses the macroeconomic environment or initial trigger conditions to determine the appropriate routing of tasks. For instance, in a high-volatility market, the orchestrator might dynamically allocate more compute time to the Risk Management node.

---

## 2. The Data Extraction Layer

Once orchestrated, the system branches into three specialized data-gathering pathways. Every agent in this layer operates on a fundamental paradigm: **(Prompt + Tool + Goal + LLM)**. This means each agent is initialized with a specific directive (Prompt), equipped with APIs (Tool), assigned a clear objective (Goal), and powered by the foundational model (LLM) to perform its individual report analysis.

### 2.1 Market Analyst Agent
The Market Analyst Agent is a strictly quantitative entity focused on price action, volume, and historical market behavior.
*   **Tools**: It primarily leverages the **Yahoo Finance API** (`yfinance.py`).
*   **Responsibilities**: It extracts end-of-day pricing, intraday tick data, moving averages, relative strength index (RSI), MACD, and other technical indicators. 
*   **Output**: Produces a structured technical analysis report detailing current market trends and momentum for the specified asset.

### 2.2 Sentiment & News Agent (with RAG Integration)
This agent is responsible for understanding the qualitative "mood" of the market by parsing vast amounts of unstructured text.
*   **Data Sources**: It pulls data from a diverse set of APIs:
    *   **Reddit API**: For retail sentiment and meme-stock velocity.
    *   **Stocktwits API**: For real-time trader sentiment and ticker momentum.
    *   **Polymarket API**: For evaluating predictive markets and betting odds on macroeconomic events.
    *   **Yahoo Finance News** (`y-finance-news.py`): For traditional financial journalism and press releases.
*   **RAG (Retrieval-Augmented Generation) Extension**: To prevent hallucination and improve contextual reasoning, this agent utilizes RAG. Instead of merely asking the LLM to summarize text, the agent embeds all fetched news articles and social posts into a vector database. When analyzing sentiment, the LLM retrieves the most semantically relevant chunks of text to ground its reasoning, ensuring its sentiment score is backed by verifiable, recent sources.

### 2.3 Fundamentals API Agent
This agent focuses on the intrinsic value of the asset and the broader macroeconomic backdrop.
*   **Tools**: 
    *   **FRED (Federal Reserve Economic Data) API**: Extracts macroeconomic indicators like inflation rates (CPI), interest rates, and employment data.
    *   **Alpha-Vantage API**: Extracts company-specific SEC filings, income statements, balance sheets, cash flow metrics, and earnings per share (EPS).
*   **Output**: Generates a fundamental health score, DCF (Discounted Cash Flow) models, and macroeconomic headwind/tailwind reports.

---

## 3. The Machine Learning (LLM) Reasoning Layer

The disparate reports from the Data Extraction Layer must be synthesized, analyzed, and debated. This requires immense reasoning capability.

### 3.1 Foundational Model: Local Qwen 2 (1.5b) via Ollama
Instead of relying on costly and latency-heavy cloud APIs, AegisTradeSim utilizes a localized instance of **Qwen 2 (1.5b)** running on **Ollama**.
*   **Efficiency**: The 1.5 billion parameter size allows for rapid inference on consumer hardware, enabling the massive number of sequential LLM calls required by the multi-agent debate structure.
*   **Privacy & Control**: Running locally ensures that sensitive trading strategies and proprietary prompts are not leaked to external providers.
*   **Functionality**: This foundational model serves as the "brain" for every subsequent agent in the pipeline. Each agent is essentially a different system prompt wrapped around this core reasoning engine.

---

## 4. Agentic Debate & Research Synthesis

This phase represents the core intellectual engine of AegisTradeSim. It employs a dialectical approach, forcing different AI personas to argue opposing viewpoints to arrive at a more robust conclusion.

### 4.1 Causal Agents: The Triad of Perspectives
The raw data is fed into three parallel **Causal Agents**. Their goal is not to find the "truth," but to construct the strongest possible argument for their assigned bias.
*   **Bull Agent**: Programmed to seek **Buy evidence**. It scans the data for bullish divergence, positive earnings surprises, and optimistic sentiment. It deliberately ignores bearish data to build a maximally optimistic thesis.
*   **Bear Agent**: Programmed to seek **Sell evidence**. It focuses on overbought indicators, macroeconomic headwinds, negative social sentiment, and declining revenue growth. It constructs a thesis for why the asset will decline.
*   **Neutral Agent**: Operates without directional bias. It focuses on volatility metrics, standard deviations, and market efficiency, providing a baseline expectation of flat movement or range-bound trading.

### 4.2 Counterfactual Analysis: The Research Manager
The polarized reports from the Bull, Bear, and Neutral agents are forwarded to the **Research Manager**. 
*   **Role**: The Research Manager operates as a **Counterfactual Agent**. Instead of just accepting the arguments, it challenges them. It asks "What if?" questions. (e.g., "If the Bull agent is right about earnings, what if the FRED data on inflation causes a broader market sell-off anyway?").
*   **Synthesis**: It weighs the strength of the evidence provided by the triad. It identifies logical fallacies or over-reliance on weak data points. The final output is a cohesive, highly scrutinized Research Document that outlines the most probable scenarios, removing the extreme biases of the subordinate agents.

---

## 5. Execution, Risk Management, and Portfolio Adjustments

The finalized research must now be translated into actionable financial decisions.

### 5.1 The Trader (Causal Agent)
The **Trader** acts as the primary execution strategist. 
*   **Function**: It ingests the Research Manager's report and formulates a specific trading plan. This includes proposing entry points, position sizing, stop-loss levels, and take-profit targets based on the synthesized intelligence.

### 5.2 The Risk Manager (Counterfactual Debate)
Before any trade is finalized, it must survive the scrutiny of the **Risk Manager**.
*   **Structure**: This is a complex Counterfactual Agent that utilizes an internal debate mechanism simulating three distinct risk profiles:
    *   **Aggressive**: Argues for maximum leverage and wider stop-losses to capture massive upside.
    *   **Conservative**: Argues for capital preservation, tight stop-losses, and minimal exposure.
    *   **Neutral Debate**: Mediates between the aggressive and conservative profiles based on the current VIX (volatility index) and the Research Manager's confidence score.
*   **Outcome**: The Risk Manager may approve, modify (e.g., reduce position size), or completely veto the Trader's proposed plan.

### 5.3 The Portfolio Manager
The finalized, risk-adjusted trade proposal is sent to the **Portfolio Manager**.
*   **Role**: This Counterfactual Agent views the new trade not in isolation, but in the context of the entire existing portfolio. 
*   **Action**: It assesses correlation risks (e.g., "Does this AAPL trade make us too exposed to the tech sector?"). It executes the final rebalancing, allocating capital to the new trade while potentially trimming other positions to maintain the target portfolio beta.

---

## 6. Final Reporting and Reinforcement Learning (RL) Optimization

### 6.1 Report Generation
The culmination of the entire LangGraph execution is the generation of a comprehensive Markdown report. This report details not just the final portfolio changes, but the entire "chain of thought"—documenting the Bull/Bear debate, the Risk Manager's vetoes, and the fundamental data that drove the decisions.

### 6.2 Reinforcement Learning (RL) Integration
The most advanced extension of the AegisTradeSim architecture is the planned integration of **Reinforcement Learning** at the Portfolio Manager level.
*   **The Feedback Loop**: In a standard LLM agent system, the agents do not "learn" from their mistakes across different executions. RL solves this.
*   **Mechanism**: The system will track the outcome of the Portfolio Manager's decisions over time (e.g., trailing 30-day PnL, maximum drawdown). 
*   **Reward Function**: Positive alpha generation will serve as a reward signal, while drawdowns will serve as penalties. 
*   **Optimization**: Over thousands of simulated trading days, the RL algorithm will fine-tune the weights and prompt-biases of the Portfolio Manager. For example, the RL agent might learn that during high-inflation regimes (flagged by the Fundamentals Agent), the Portfolio Manager should place higher trust in the Conservative Risk Manager's vetoes. This creates a continuously evolving, self-improving trading ecosystem.

---

## Conclusion
The AegisTradeSim architecture represents a cutting-edge fusion of agentic workflows, adversarial LLM debate, and classical financial analysis. By anchoring its reasoning in a local Qwen 2 model, expanding its contextual awareness through RAG, and planning for continuous self-improvement via RL, the system moves beyond simple rule-based trading into the realm of dynamic, reasoned, and risk-adjusted portfolio management.
