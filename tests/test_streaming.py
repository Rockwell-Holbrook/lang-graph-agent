"""Offline streaming tests — no API key, no tokens spent.

PART 1 pins the SSE translation in `server._event_stream` (token forwarding, the
classification chip, the reject/clarify fallback, graceful error handling) by
driving it with synthetic (mode, data) tuples shaped exactly like LangGraph output.

PART 2 pins the LangGraph streaming CONTRACT our code assumes: with
stream_mode=["messages","values"] it yields (mode, data) tuples and messages-mode
metadata carries `langgraph_node`.
"""
from __future__ import annotations

import json
from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

import server
from server import ChatRequest
from src.schemas import Classification, QueryType, Route


class _ContractState(TypedDict):
    messages: Annotated[list, add_messages]


def _collect(req: ChatRequest) -> list[dict]:
    events = []
    for chunk in server._event_stream(req):
        assert chunk.startswith("data: ") and chunk.endswith("\n\n"), chunk
        events.append(json.loads(chunk[len("data: "):].strip()))
    return events


class FakeGraph:
    """Stand-in for the compiled graph: replays a scripted stream. Accepts (and
    ignores) the `config` the server now threads in for the checkpointer."""

    def __init__(self, script):
        self._script = script
        self.last_payload = None

    def stream(self, payload, config=None, stream_mode=None):
        self.last_payload = payload
        for item in self._script:
            if isinstance(item, Exception):
                raise item
            yield item


def _classification(query_type: QueryType, route: Route, followup: bool = False) -> Classification:
    return Classification(query_type=query_type, route=route, is_followup=followup, reason="test")


@pytest.fixture(autouse=True)
def _restore_graph():
    original = server.GRAPH
    yield
    server.GRAPH = original


# --------------------------------------------------------------------------- #
# PART 1: translation logic
# --------------------------------------------------------------------------- #
def test_answer_path_streams_agent_tokens_and_one_chip():
    meta = {"langgraph_node": "agent"}
    server.GRAPH = FakeGraph([
        ("values", {"classification": _classification(QueryType.POKEMON_INFO, Route.ANSWER, True)}),
        ("messages", (AIMessageChunk(content="Charizard"), meta)),
        ("messages", (AIMessageChunk(content=" is Fire/Flying"), meta)),
        # tokens from other nodes must be ignored:
        ("messages", (AIMessageChunk(content="IGNORE"), {"langgraph_node": "classify"})),
        ("values", {"final_reply": "Charizard is Fire/Flying"}),
    ])
    events = _collect(ChatRequest(message="tell me about charizard", thread_id="t"))

    tokens = [e["text"] for e in events if e["type"] == "token"]
    metas = [e for e in events if e["type"] == "meta"]
    assert tokens == ["Charizard", " is Fire/Flying"]
    assert "IGNORE" not in "".join(tokens)
    assert len(metas) == 1
    assert metas[0]["route"] == "answer"
    assert metas[0]["query_type"] == "pokemon_info"
    assert metas[0]["is_followup"] is True
    assert events[-1]["type"] == "done"


def test_reject_path_falls_back_to_final_reply():
    server.GRAPH = FakeGraph([
        ("values", {"classification": _classification(QueryType.NOT_POKEMON, Route.REJECT)}),
        ("values", {"final_reply": "I'm a Pokémon assistant."}),
    ])
    events = _collect(ChatRequest(message="what's the weather?", thread_id="t"))
    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert tokens == ["I'm a Pokémon assistant."]
    assert events[-1]["type"] == "done"


def test_new_turn_resets_persisted_derived_state():
    """Regression: `classification` and `final_reply` are per-turn derived channels the
    checkpointer persists. If a turn doesn't reset them, the stream's initial emission
    hands back the PREVIOUS turn's values — the chip lags a turn and a no-stream turn
    echoes the last reply. The request payload must clear them."""
    g = FakeGraph([("values", {"final_reply": "ok"})])
    server.GRAPH = g
    _collect(ChatRequest(message="hi", thread_id="t"))
    assert g.last_payload["classification"] is None
    assert g.last_payload["final_reply"] is None


def test_error_mid_stream_becomes_error_event():
    server.GRAPH = FakeGraph([RuntimeError("boom")])
    events = _collect(ChatRequest(message="hi", thread_id="t"))
    assert any(e["type"] == "error" and "boom" in e.get("text", "") for e in events)


def test_missing_thread_id_is_tolerated():
    server.GRAPH = FakeGraph([("values", {"final_reply": "ok"})])
    events = _collect(ChatRequest(message="hi"))  # no thread_id supplied
    assert events[-1]["type"] == "done"


# --------------------------------------------------------------------------- #
# PART 2: LangGraph streaming contract (real graph, fake streaming model)
# --------------------------------------------------------------------------- #
def test_langgraph_stream_contract():
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

    saw_tuple = saw_messages = saw_node = saw_values = False
    for item in compiled.stream({"messages": [("user", "hi")]},
                                stream_mode=["messages", "values"]):
        saw_tuple = saw_tuple or (isinstance(item, tuple) and len(item) == 2)
        mode, data = item
        if mode == "messages":
            saw_messages = True
            _msg, metadata = data
            saw_node = saw_node or (isinstance(metadata, dict) and "langgraph_node" in metadata)
        elif mode == "values":
            saw_values = True

    assert saw_tuple and saw_messages and saw_node and saw_values
