"""No-API structural smoke test.

Proves the graph is wired correctly WITHOUT spending a token or needing a key:
  - the graph compiles and has the expected nodes
  - the deterministic router branches correctly
  - the loop guard terminates
  - tools and the LLM-free escalate node work end to end

Run: python smoke_test.py
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from src.graph import GRAPH, should_continue
from src.nodes import escalate, route_after_classify
from src.schemas import Classification, Intent, Route
from src.tools import create_ticket, lookup_customer


def check(name: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def main() -> None:
    print("1. Graph compiles with expected nodes")
    nodes = set(GRAPH.get_graph().nodes)
    for n in ["classify", "agent", "tools", "finalize", "escalate", "clarify"]:
        check(f"node '{n}' present", n in nodes)

    print("2. Router branches deterministically")
    base = dict(intent=Intent.QUESTION, sentiment="neutral", reason="x")
    respond = Classification(**base, urgency=2, route=Route.RESPOND)
    check("normal respond -> respond",
          route_after_classify({"classification": respond}) == "respond")
    urgent = Classification(**base, urgency=5, route=Route.RESPOND)
    check("high-urgency respond is overridden -> escalate",
          route_after_classify({"classification": urgent}) == "escalate")
    vague = Classification(**base, urgency=1, route=Route.CLARIFY)
    check("clarify -> clarify",
          route_after_classify({"classification": vague}) == "clarify")

    print("3. Loop guard terminates")
    ai_with_call = AIMessage(content="", tool_calls=[
        {"name": "lookup_customer", "args": {"customer_id": "cust_123"}, "id": "1"}
    ])
    check("tool call with room -> tools",
          should_continue({"messages": [ai_with_call]}) == "tools")
    # Simulate max rounds reached
    many = [ToolMessage(content="{}", tool_call_id=str(i)) for i in range(10)]
    check("cap reached -> finalize",
          should_continue({"messages": [*many, ai_with_call]}) == "finalize")
    check("no tool call -> finalize",
          should_continue({"messages": [AIMessage(content="done")]}) == "finalize")

    print("4. Tools work")
    check("lookup known customer",
          lookup_customer.invoke({"customer_id": "cust_123"})["name"] == "Dana Lee")
    check("lookup unknown -> error",
          "error" in lookup_customer.invoke({"customer_id": "nope"}))
    check("create_ticket returns id",
          create_ticket.invoke({"summary": "x", "priority": "high"})["ticket_id"]
          .startswith("TCK-"))

    print("5. LLM-free escalate node runs end to end")
    out = escalate({"classification": urgent, "inbound_message": "bad!"})
    check("escalate sets handled_by=human", out["handled_by"] == "human")
    check("escalate drafts a reply", bool(out["final_reply"]))

    print("\nALL SMOKE CHECKS PASSED — graph wiring is sound (no API used).")


if __name__ == "__main__":
    main()
