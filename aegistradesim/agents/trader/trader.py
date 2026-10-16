"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from aegistradesim.agents.schemas import TraderProposal, render_trader_proposal
from aegistradesim.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from aegistradesim.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]
        # The research plan digests the debate but loses exact price structure;
        # give the Trader the technical market report so entry/stop levels are
        # grounded in real ATR / support-resistance / current price (#1167). The
        # report is empty when the user did not select the market analyst, so
        # only offer it (and the grounding instruction) when it has content.
        market_report = (state["market_report"] or "").strip()

        if market_report:
            grounding = (
                "Ground concrete price levels (entry, stop-loss, position sizing) in the technical "
                "market report's price structure -- current price, support/resistance, ATR, and "
                "volatility -- and use the research plan for direction and strategy. "
                "CRITICAL: Use the EXACT Current Price from the Technical Market Report. "
                "Never invent placeholder numbers (like $181.50) or placeholder tickers (like TSLA.TO). "
            )
            report_section = f"Technical Market Report:\n{market_report}\n\n"
        else:
            grounding = ""
            report_section = ""


        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    + grounding
                    + " Format your proposal with: 90% numerical tables and parameters (Entry price, Stop loss, Target price, Position sizing %, ATR, Volatility %, Support/Resistance) and 10% concise strategy. "
                    + NO_EXTERNAL_TOOLS
                    + get_language_instruction()
                ),
            },

            {
                "role": "user",
                "content": (
                    f"Here is the research team's investment plan for {company_name}. "
                    f"{instrument_context}\n\n"
                    f"{report_section}"
                    f"Proposed Investment Plan:\n{investment_plan}\n\n"
                    f"Make an informed, strategic trading decision."
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
