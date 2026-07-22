"""Offline streaming tests — no API key, no tokens spent.

Two things are unverified in the streaming path and this file pins both down:

  PART 1 — MY translation logic in server._event_stream:
      Does it turn LangGraph's (mode, data) stream into the right SSE events?
      (token forwarding, the classification chip, the escalate/clarify fallback,
      and graceful error handling.) We drive it with SYNTHETIC tuples shaped
      exactly like real LangGraph output, so it's deterministic.

  PART 2 — LangGraph's streaming CONTRACT that my code assumes:
      With stream_mode=["messages","values"], does it really yield (mode, data)
      tuples, and does messages-mode data carry a `langgraph_node` in metadata?
      We check this against a real tiny StateGraph + a fake streaming model.

Run: python test_streaming.py
"""
from __future__ import annotations

import json
from typing import Annotated

from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

import server
from server import ChatRequest
from src.schemas import Classification, Intent, Route


# Defined at module level so its annotations resolve (Python 3.14 evaluates
# forward refs against module globals, not function locals).
class _ContractState(TypedDict):
    messages: Annotated[list, add_messages]


def _collect(req: ChatRequest) -> list[dict]:
    """Run the SSE generator and parse each event into a dict."""
    events = []
    for chunk in server._event_stream(req):
        assert chunk.startswith("data: ") and chunk.endswith("\n\n"), chunk
        events.append(json.loads(chunk[len("data: "):].strip()))
    return events


def check(name: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


class FakeGraph:
    """Stand-in for the compiled graph: replays a scripted stream."""
    def __init__(self, script):
        self._script = script

    def stream(self, payload, stream_mode=None):
        for item in self._script:
            if isinstance(item, Exception):
                raise item
            yield item


def _classification(route: Route) -> Classification:
    return Classification(intent=Intent.QUESTION, sentiment="neutral",
                          urgency=2, route=route, reason="test")


# --------------------------------------------------------------------------- #
# PART 1: translation logic
# --------------------------------------------------------------------------- #
def test_respond_path_streams_tokens(monkey_restore):
    print("PART 1a: respond path streams agent tokens (no duplicate final)")
    meta = {"langgraph_node": "agent"}
    script = [
        ("values", {"classification": _classification(Route.RESPOND)}),
        ("messages", (AIMessageChunk(content="Hello"), meta)),
        ("messages", (AIMessageChunk(content=" there"), meta)),
        # tokens from OTHER nodes must be ignored:
        ("messages", (AIMessageChunk(content="IGNORE"), {"langgraph_node": "classify"})),
        ("values", {"final_reply": "Hello there"}),
    ]
    server.GRAPH = FakeGraph(script)
    events = _collect(ChatRequest(message="hi"))

    tokens = [e["text"] for e in events if e["type"] == "token"]
    metas = [e for e in events if e["type"] == "meta"]
    check("tokens are the two agent chunks, in order", tokens == ["Hello", " there"])
    check("classify-node token was filtered out", "IGNORE" not in "".join(tokens))
    check("exactly one meta chip emitted", len(metas) == 1)
    check("meta chip carries route/intent/urgency",
          metas[0]["route"] == "respond" and metas[0]["urgency"] == 2)
    check("no duplicate final reply appended", "".join(tokens) == "Hello there")
    check("stream ends with done", events[-1]["type"] == "done")


def test_escalate_path_fallback(monkey_restore):
    print("PART 1b: escalate/clarify path (no agent tokens) falls back to final_reply")
    script = [
        ("values", {"classification": _classification(Route.ESCALATE)}),
        ("values", {"final_reply": "A teammate will follow up (ref TCK-00042)."}),
    ]
    server.GRAPH = FakeGraph(script)
    events = _collect(ChatRequest(message="this is terrible"))

    tokens = [e["text"] for e in events if e["type"] == "token"]
    check("no tokens streamed mid-flight", len(tokens) <= 1)
    check("final reply emitted exactly once via fallback",
          tokens == ["A teammate will follow up (ref TCK-00042)."])
    check("stream ends with done", events[-1]["type"] == "done")


def test_error_is_surfaced(monkey_restore):
    print("PART 1c: an exception mid-stream becomes an error event, not a hang")
    server.GRAPH = FakeGraph([RuntimeError("boom")])
    events = _collect(ChatRequest(message="hi"))
    check("error event emitted", any(e["type"] == "error" for e in events))
    check("error text carries the message",
          any("boom" in e.get("text", "") for e in events))


# --------------------------------------------------------------------------- #
# PART 2: LangGraph streaming contract (real graph, fake streaming model)
# --------------------------------------------------------------------------- #
def test_langgraph_stream_contract():
    print("PART 2: LangGraph honors the (mode, data) + langgraph_node contract")
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langgraph.graph import END, START, StateGraph

    fake = GenericFakeChatModel(messages=iter([AIMessage(content="hi from fake model")]))

    def node(state: _ContractState):
        return {"messages": [fake.invoke(state["messages"])]}

    g = StateGraph(_ContractState)
    g.add_node("agent", node)
    g.add_edge(START, "agent")
    g.add_edge("agent", END)
    compiled = g.compile()

    saw_tuple = False
    saw_messages_mode = False
    saw_langgraph_node = False
    saw_values_mode = False

    for item in compiled.stream({"messages": [("user", "hi")]},
                                stream_mode=["messages", "values"]):
        check_tuple = isinstance(item, tuple) and len(item) == 2
        saw_tuple = saw_tuple or check_tuple
        mode, data = item
        if mode == "messages":
            saw_messages_mode = True
            _msg, metadata = data
            if isinstance(metadata, dict) and "langgraph_node" in metadata:
                saw_langgraph_node = True
        elif mode == "values":
            saw_values_mode = True

    check("stream yields (mode, data) tuples", saw_tuple)
    check("messages mode present", saw_messages_mode)
    check("messages metadata contains 'langgraph_node'", saw_langgraph_node)
    check("values mode present", saw_values_mode)


# --------------------------------------------------------------------------- #
def main() -> None:
    original = server.GRAPH
    try:
        test_respond_path_streams_tokens(None)
        test_escalate_path_fallback(None)
        test_error_is_surfaced(None)
    finally:
        server.GRAPH = original  # restore the real graph

    test_langgraph_stream_contract()
    print("\nALL STREAMING CHECKS PASSED — no API key used.")


if __name__ == "__main__":
    main()
