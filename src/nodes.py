"""Graph nodes.

Each node is a pure-ish function: (state) -> partial state update. Keeping nodes
small and single-purpose is what makes the graph readable and testable.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import get_llm
from .schemas import Classification, DraftReply, Route
from .state import AgentState
from .tools import AGENT_TOOLS, create_ticket

log = logging.getLogger("podium.nodes")

# --------------------------------------------------------------------------- #
# 1. CLASSIFY — turn a raw message into typed, branchable data.
# --------------------------------------------------------------------------- #
CLASSIFY_SYSTEM = (
    "You are a triage assistant for a local business's customer inbox. "
    "Classify the inbound message. Choose route=respond for straightforward "
    "questions you can answer, route=escalate for complaints/billing disputes or "
    "high urgency, and route=clarify when the message is too vague to act on."
)


def classify(state: AgentState) -> AgentState:
    """LLM step with structured output -> Classification."""
    llm = get_llm().with_structured_output(Classification)
    result: Classification = llm.invoke(
        [
            SystemMessage(content=CLASSIFY_SYSTEM),
            HumanMessage(content=state["inbound_message"]),
        ]
    )
    log.info("classified intent=%s route=%s urgency=%s",
             result.intent, result.route, result.urgency)
    return {"classification": result}


# --------------------------------------------------------------------------- #
# Conditional edge function: read classification, decide the branch.
# This is the "router" — pure logic, no LLM call, fully deterministic.
# --------------------------------------------------------------------------- #
def route_after_classify(state: AgentState) -> str:
    classification: Classification = state["classification"]
    # Safety override: never let the model auto-respond to high urgency.
    if classification.urgency >= 4 and classification.route == Route.RESPOND:
        return Route.ESCALATE.value
    return classification.route.value


# --------------------------------------------------------------------------- #
# 2a. AGENT — tool-calling node. May loop with the ToolNode (see graph.py).
# --------------------------------------------------------------------------- #
AGENT_SYSTEM = (
    "You are a helpful customer-support agent for a local business. "
    "Use tools to look up account details when relevant. Keep replies concise, "
    "friendly, and accurate. When you have enough information, write the final "
    "reply directly with no tool call."
)


def agent(state: AgentState) -> AgentState:
    """Bind tools and let the model either call a tool or produce a final reply."""
    llm = get_llm().bind_tools(AGENT_TOOLS)

    # Seed the message channel on first entry.
    messages = state.get("messages") or []
    if not messages:
        messages = [
            SystemMessage(content=AGENT_SYSTEM),
            HumanMessage(
                content=(
                    f"Customer (id={state.get('customer_id')}) says:\n"
                    f"{state['inbound_message']}"
                )
            ),
        ]

    ai_msg = llm.invoke(messages)
    return {"messages": [*(messages if not state.get("messages") else []), ai_msg]}


def finalize_agent_reply(state: AgentState) -> AgentState:
    """Pull the last AI message out of the tool loop as the final reply."""
    last = state["messages"][-1]
    return {"final_reply": last.content, "handled_by": "agent"}


# --------------------------------------------------------------------------- #
# 2b. ESCALATE — open a ticket, tell the customer a human will follow up.
# --------------------------------------------------------------------------- #
def escalate(state: AgentState) -> AgentState:
    c: Classification = state["classification"]
    priority = "high" if c.urgency >= 4 else "normal"
    ticket = create_ticket.invoke(
        {"summary": f"[{c.intent.value}] {state['inbound_message'][:80]}",
         "priority": priority}
    )
    reply = (
        "Thanks for reaching out — I've flagged this to a team member who will "
        f"follow up shortly (ref {ticket['ticket_id']})."
    )
    log.info("escalated -> %s", ticket["ticket_id"])
    return {"final_reply": reply, "handled_by": "human"}


# --------------------------------------------------------------------------- #
# 2c. CLARIFY — ask one focused follow-up question.
# --------------------------------------------------------------------------- #
def clarify(state: AgentState) -> AgentState:
    llm = get_llm().with_structured_output(DraftReply)
    draft: DraftReply = llm.invoke(
        [
            SystemMessage(
                content="The customer's message is too vague to act on. Write ONE "
                "short, friendly clarifying question to move things forward."
            ),
            HumanMessage(content=state["inbound_message"]),
        ]
    )
    return {"final_reply": draft.body, "handled_by": "clarify"}
