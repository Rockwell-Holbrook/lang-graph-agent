"""Structured output schemas.

We force the LLM into these shapes with `.with_structured_output(...)` instead of
parsing free text. This is the single biggest reliability win in an agent: the
model's decision becomes typed data the graph can branch on deterministically.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    QUESTION = "question"          # customer wants info
    COMPLAINT = "complaint"        # something went wrong
    SCHEDULING = "scheduling"      # book / reschedule / cancel
    BILLING = "billing"            # payments, invoices
    SPAM = "spam"                  # not a real lead
    OTHER = "other"


class Route(str, Enum):
    """Where the graph should send this message after classification."""
    RESPOND = "respond"    # agent drafts a reply (may use tools)
    ESCALATE = "escalate"  # hand to a human / open a ticket
    CLARIFY = "clarify"    # not enough info; ask a follow-up


class Classification(BaseModel):
    """Typed result of the triage step."""
    intent: Intent = Field(description="Primary intent of the inbound message.")
    sentiment: str = Field(description="One of: positive, neutral, negative.")
    urgency: int = Field(ge=1, le=5, description="1 = trivial, 5 = urgent.")
    route: Route = Field(description="Recommended next action for the graph.")
    reason: str = Field(description="One short sentence justifying the route.")


class DraftReply(BaseModel):
    """A drafted customer-facing message."""
    body: str = Field(description="The reply text to send to the customer.")
    requires_review: bool = Field(
        description="True if a human should approve before sending."
    )
