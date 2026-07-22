"""Graph assembly.

    START
      │
      ▼
   classify ──► route_after_classify ──┬─► reject   ─► END
                                       ├─► clarify  ─► END
                                       └─► agent ⇄ tools  (cycle)
                                              │
                                              ▼
                                           finalize ─► END

The agent<->tools cycle is the reason this is a *graph* and not a linear chain:
the model can call a tool, see the result, and decide again. `should_continue`
enforces a hard iteration cap so the loop always terminates.

Compiled with a `MemorySaver` checkpointer so a `thread_id` gives the conversation
memory across turns — that persisted `messages` history is what lets the agent
resolve "it" -> Charizard. (No `interrupt_before`/HITL: read-only Q&A has no action
to gate.)
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .config import SETTINGS
from .nodes import (
    agent,
    classify,
    clarify,
    finalize_agent_reply,
    reject,
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


def build_graph(checkpointer: MemorySaver | None = None):
    builder = StateGraph(AgentState)

    builder.add_node("classify", classify)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(AGENT_TOOLS))
    builder.add_node("finalize", finalize_agent_reply)
    builder.add_node("reject", reject)
    builder.add_node("clarify", clarify)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {"answer": "agent", "reject": "reject", "clarify": "clarify"},
    )
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "finalize": "finalize"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("finalize", END)
    builder.add_edge("reject", END)
    builder.add_edge("clarify", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


# Compiled once at import — cheap, and lets `main.py`, the server, and tests share it.
GRAPH = build_graph()
