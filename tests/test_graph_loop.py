"""Graph assembly + the bounded agent<->tools loop guard."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.config import SETTINGS
from src.nodes import agent
from src.tools import AGENT_TOOLS
from src.graph import GRAPH, should_continue


def test_graph_compiles_with_expected_nodes():
    nodes = set(GRAPH.get_graph().nodes)
    assert {"classify", "agent", "tools", "finalize", "reject", "clarify"} <= nodes


def _ai_with_tool_call() -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": "get_pokemon", "args": {"name_or_id": "pikachu"}, "id": "1"},
    ])


def _saturated() -> list[ToolMessage]:
    return [ToolMessage(content="{}", tool_call_id=str(i))
            for i in range(SETTINGS.max_tool_iterations)]


def test_tool_call_with_room_continues_to_tools():
    assert should_continue({"messages": [_ai_with_tool_call()]}) == "tools"


def test_no_tool_call_finalizes():
    assert should_continue({"messages": [AIMessage(content="done")]}) == "finalize"


def test_pending_tool_call_is_always_answered_even_at_cap():
    """A pending tool_call must NEVER be dropped — routing it to `finalize` would
    persist an assistant tool_calls message with no matching ToolMessage and break the
    next turn's history. Termination is the agent's job (it stops offering tools at the
    cap), not something we get by abandoning a call here."""
    assert should_continue({"messages": [*_saturated(), _ai_with_tool_call()]}) == "tools"


def test_agent_offers_tools_under_budget(scripted_llm, patched_tools):
    llm = scripted_llm(ai=[AIMessage(content="here you go")])
    agent({"messages": [HumanMessage(content="What moves can Lucario learn?")]})
    assert llm.bound_tools is AGENT_TOOLS  # tools offered while budget remains


def test_agent_withholds_tools_once_budget_is_spent(scripted_llm, patched_tools):
    """At the cap the model is invoked without tools, so it can't request another call
    we'd be unable to answer — the mechanism that stops the dangling-tool_call defect."""
    llm = scripted_llm(ai=[AIMessage(content="final answer from what I have")])
    agent({"messages": [HumanMessage(content="q"), *_saturated()]})
    assert llm.bound_tools is None  # no tools bound -> no new tool_calls can be dangled
