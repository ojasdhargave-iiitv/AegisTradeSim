from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from aegistradesim.agents.utils.agent_utils import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_instrument_context_from_state,
    get_language_instruction,
)


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        analyst_context = state.get("analyst_context", "")

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            "You are a researcher tasked with analyzing fundamental information about a company. Please write a data-dense report consisting of 90% structured Markdown financial tables (Revenue, Net Income, EBITDA, Cash Flow, Debt-to-Equity, PE/PB ratios, Gross Margin) and 10% concise analytical takeaways. Do NOT output raw JSON tool call strings in your final deliverable; synthesize all tool results into organized financial tables."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
            + get_language_instruction(),
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{analyst_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(analyst_context=analyst_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content or ""

        # Guarantee real fundamental data if tool calling failed or returned raw tool json
        is_raw_tool_json = "get_fundamentals" in report or "get_balance_sheet" in report
        if not report or len(report.strip()) < 50 or is_raw_tool_json:
            symbol = state.get("company_of_interest", "")
            if symbol:
                try:
                    real_fund_data = get_fundamentals.invoke({"ticker": symbol, "curr_date": str(current_date)})
                    report = (
                        f"# Company Fundamentals Report: {symbol} ({current_date})\n\n"
                        f"### Verified Financial Metrics\n"
                        f"{real_fund_data}\n\n"
                        f"{report if not is_raw_tool_json else ''}"
                    )
                except Exception:
                    pass

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node

