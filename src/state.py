"""Graph state.

The single source of truth that flows through every node. Nodes read from it and
return partial updates; LangGraph merges those updates. `messages` uses the
`add_messages` reducer so tool-calling turns accumulate correctly, and — combined
with the `MemorySaver` checkpointer in `graph.py` — persists across turns of a
conversation so the agent can resolve "it" -> Charizard.
"""
from __future__ import annotations

from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .schemas import Classification


class AgentState(TypedDict, total=False):
    # --- Conversation / tool-calling channel ---
    # New user text enters as a HumanMessage appended here each turn. The reducer
    # appends instead of overwriting — this is what makes both the agent<->tools
    # cycle and multi-turn memory work.
    messages: Annotated[list, add_messages]

    # --- Derived by the graph ---
    classification: Optional[Classification]
    final_reply: Optional[str]     # what we'd send back this turn
    handled_by: Optional[str]      # "agent" | "clarify" | "rejected" — observability
