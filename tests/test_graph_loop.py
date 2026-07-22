"""Graph assembly + the bounded agent<->tools loop guard."""
from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from src.config import SETTINGS
from src.graph import GRAPH, should_continue


def test_graph_compiles_with_expected_nodes():
    nodes = set(GRAPH.get_graph().nodes)
    assert {"classify", "agent", "tools", "finalize", "reject", "clarify"} <= nodes


def _ai_with_tool_call() -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": "get_pokemon", "args": {"name_or_id": "pikachu"}, "id": "1"},
    ])


def test_tool_call_with_room_continues_to_tools():
    assert should_continue({"messages": [_ai_with_tool_call()]}) == "tools"


def test_no_tool_call_finalizes():
    assert should_continue({"messages": [AIMessage(content="done")]}) == "finalize"


def test_iteration_cap_forces_finalize():
    """Once the tool-round cap is hit, the loop must terminate even mid tool-call."""
    saturated = [ToolMessage(content="{}", tool_call_id=str(i))
                 for i in range(SETTINGS.max_tool_iterations)]
    assert should_continue({"messages": [*saturated, _ai_with_tool_call()]}) == "finalize"
