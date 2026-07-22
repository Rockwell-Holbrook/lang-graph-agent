"""Tools the agent can call.

Each tool is a plain, typed function with a clear docstring — the docstring IS the
spec the LLM sees, so it must describe when to use the tool and what it returns.
These are mocked (no real backend) so the scaffold runs offline-safe; swap the
bodies for real API calls on the day.
"""
from __future__ import annotations

from langchain_core.tools import tool

# --- Fake datastore so the scaffold is self-contained -----------------------
_FAKE_CUSTOMERS = {
    "cust_123": {"name": "Dana Lee", "plan": "Pro", "open_invoices": 0},
    "cust_456": {"name": "Sam Ruiz", "plan": "Starter", "open_invoices": 2},
}


@tool
def lookup_customer(customer_id: str) -> dict:
    """Look up a customer's account details by their customer_id.

    Use this before drafting a reply when you need the customer's name, plan, or
    billing status. Returns a dict of account fields, or an 'error' key if the id
    is unknown.
    """
    record = _FAKE_CUSTOMERS.get(customer_id)
    if record is None:
        return {"error": f"No customer found for id {customer_id!r}"}
    return {"customer_id": customer_id, **record}


@tool
def create_ticket(summary: str, priority: str = "normal") -> dict:
    """Open a support ticket for a human to handle.

    Use this when a message must be escalated (angry customer, billing dispute,
    anything you cannot safely resolve). `priority` is one of low|normal|high.
    Returns the created ticket id.
    """
    # In reality this would POST to a ticketing system.
    ticket_id = f"TCK-{abs(hash(summary)) % 100000:05d}"
    return {"ticket_id": ticket_id, "priority": priority, "summary": summary}


@tool
def send_message(customer_id: str, body: str) -> dict:
    """Send an SMS/message to the customer.

    Use ONLY once you have a final, approved reply. Returns a delivery receipt.
    """
    # In reality this would POST to the messaging API.
    return {"status": "sent", "customer_id": customer_id, "chars": len(body)}


# Tools the tool-calling agent node is allowed to use.
AGENT_TOOLS = [lookup_customer, send_message]

# Registry (handy for the escalate node and for tests).
ALL_TOOLS = {t.name: t for t in [lookup_customer, create_ticket, send_message]}
