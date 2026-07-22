"""Graph state.

The single source of truth that flows through every node. Nodes read from it and
return partial updates; LangGraph merges those updates. `messages` uses the
`add_messages` reducer so tool-calling turns accumulate correctly.
"""
from __future__ import annotations

from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .schemas import Classification


class AgentState(TypedDict, total=False):
    # --- Input ---
    inbound_message: str           # the raw customer message
    customer_id: Optional[str]     # known customer, if any

    # --- Conversation / tool-calling channel ---
    # Reducer appends new messages instead of overwriting. This is what makes the
    # agent<->tools cycle work.
    messages: Annotated[list, add_messages]

    # --- Derived by the graph ---
    classification: Optional[Classification]
    final_reply: Optional[str]     # what we'd send back
    handled_by: Optional[str]      # "agent" | "human" | "clarify" — for observability
