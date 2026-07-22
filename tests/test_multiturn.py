"""End-to-end multi-turn test — the headline requirement.

With a MemorySaver checkpointer and a fixed thread_id, turn 1's conversation must
persist so turn 2 ("what abilities can it have?") is answered in context. We drive
the REAL compiled graph (real router, real ToolNode, real tools over the mock
PokéAPI) and only script the model, so this exercises the actual memory plumbing.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.graph import build_graph
from src.schemas import Classification, QueryType, Route


def _tool_call(name_or_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": "get_pokemon", "args": {"name_or_id": name_or_id}, "id": "t1"},
    ])


def test_followup_resolves_against_persisted_history(patched_tools, scripted_llm):
    # Turn 1: classify -> answer; agent calls get_pokemon, then replies.
    # Turn 2: classify -> answer (follow-up); agent calls get_pokemon again, replies.
    scripted_llm(
        structured=[
            Classification(query_type=QueryType.POKEMON_INFO, route=Route.ANSWER,
                           is_followup=False, reason="asks about Charizard"),
            Classification(query_type=QueryType.POKEMON_INFO, route=Route.ANSWER,
                           is_followup=True, reason="'it' = Charizard"),
        ],
        ai=[
            _tool_call("charizard"),
            AIMessage(content="Charizard is a Fire/Flying-type Pokémon."),
            _tool_call("charizard"),  # scripted model resolves "it" -> charizard
            AIMessage(content="Charizard can have Blaze, or Solar Power as a hidden ability."),
        ],
    )

    graph = build_graph()  # fresh checkpointer, isolated from other tests
    config = {"configurable": {"thread_id": "conv-1"}}

    turn1 = graph.invoke({"messages": [HumanMessage(content="Tell me about Charizard")]}, config)
    assert turn1["handled_by"] == "agent"
    assert "Fire/Flying" in turn1["final_reply"]

    turn2 = graph.invoke({"messages": [HumanMessage(content="What abilities can it have?")]}, config)
    assert turn2["classification"].is_followup is True
    assert "Blaze" in turn2["final_reply"]

    # Memory proof: turn 1's user message is still in the persisted thread at turn 2.
    history = graph.get_state(config).values["messages"]
    human_turns = [m.content for m in history if isinstance(m, HumanMessage)]
    assert human_turns == ["Tell me about Charizard", "What abilities can it have?"]


def test_separate_threads_do_not_share_memory(patched_tools, scripted_llm):
    scripted_llm(
        structured=[
            Classification(query_type=QueryType.POKEMON_INFO, route=Route.ANSWER, reason="x"),
        ],
        ai=[AIMessage(content="Pikachu is an Electric-type.")],
    )
    graph = build_graph()
    graph.invoke({"messages": [HumanMessage(content="What type is Pikachu?")]},
                 {"configurable": {"thread_id": "A"}})

    # A different thread starts empty.
    assert graph.get_state({"configurable": {"thread_id": "B"}}).values == {}
