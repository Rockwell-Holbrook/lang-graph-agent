"""Graph assembly.

    START
      │
      ▼
   classify ──► route_after_classify ──┬─► escalate ─► END
                                        ├─► clarify  ─► END
                                        └─► agent ⇄ tools  (cycle)
                                               │
                                               ▼
                                            finalize ─► END

The agent<->tools cycle is the reason this is a *graph* and not a linear chain:
the model can call a tool, see the result, and decide again. `should_continue`
also enforces a hard iteration cap so the loop always terminates.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .config import SETTINGS
from .nodes import (
    agent,
    classify,
    clarify,
    escalate,
    finalize_agent_reply,
    route_after_classify,
)
from .state import AgentState
from .tools import AGENT_TOOLS


def should_continue(state: AgentState) -> str:
    """Loop guard for the agent<->tools cycle."""
    messages = state["messages"]
    last = messages[-1]
    tool_rounds = sum(1 for m in messages if getattr(m, "type", None) == "tool")
    has_tool_call = bool(getattr(last, "tool_calls", None))
    if has_tool_call and tool_rounds < SETTINGS.max_tool_iterations:
        return "tools"
    return "finalize"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("classify", classify)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(AGENT_TOOLS))
    builder.add_node("finalize", finalize_agent_reply)
    builder.add_node("escalate", escalate)
    builder.add_node("clarify", clarify)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {"respond": "agent", "escalate": "escalate", "clarify": "clarify"},
    )
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "finalize": "finalize"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("finalize", END)
    builder.add_edge("escalate", END)
    builder.add_edge("clarify", END)

    return builder.compile()


# Compiled once at import — cheap, and lets `main.py` and tests share it.
GRAPH = build_graph()
