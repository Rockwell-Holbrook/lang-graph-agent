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
    """Loop guard for the agent<->tools cycle.

    Invariant: every assistant `tool_calls` message MUST be answered by a ToolNode
    round. Abandoning a pending call would persist an AIMessage whose tool_calls have
    no matching ToolMessage, and OpenAI rejects that history on the next turn ("tool_
    call_ids did not have response messages"). So pending tool-calls always route to
    `tools`. The iteration budget is NOT enforced here by dropping a call — it is
    enforced in the `agent` node, which stops *offering* tools once the cap is reached,
    so the model simply can't request another round.
    """
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
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
