"""Node behavior: the LLM-free reject node, finalize, and classify wiring."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.nodes import classify, finalize_agent_reply, reject
from src.schemas import Classification, QueryType, Route


def test_reject_is_deterministic_and_uses_no_llm():
    """reject must produce a canned decline without touching the model — no
    scripted_llm fixture is installed, so any LLM call would raise."""
    out = reject({"messages": [HumanMessage(content="what's the weather?")]})
    assert out["handled_by"] == "rejected"
    assert "Pokémon" in out["final_reply"]
    assert isinstance(out["messages"][0], AIMessage)


def test_finalize_pulls_last_message_as_reply():
    state = {"messages": [HumanMessage(content="hi"), AIMessage(content="Pikachu is Electric.")]}
    out = finalize_agent_reply(state)
    assert out["final_reply"] == "Pikachu is Electric." and out["handled_by"] == "agent"


def test_classify_returns_structured_classification(scripted_llm):
    scripted_llm(structured=[
        Classification(query_type=QueryType.POKEMON_INFO, route=Route.ANSWER, reason="x"),
    ])
    out = classify({"messages": [HumanMessage(content="What type is Pikachu?")]})
    assert out["classification"].route == Route.ANSWER


def test_classify_resets_final_reply_each_turn(scripted_llm):
    """A new turn must not inherit the previous turn's reply from the persisted state."""
    scripted_llm(structured=[
        Classification(query_type=QueryType.POKEMON_INFO, route=Route.ANSWER, reason="x"),
    ])
    out = classify({"messages": [HumanMessage(content="What type is Pikachu?")],
                    "final_reply": "stale answer from a previous turn"})
    assert out["final_reply"] is None
