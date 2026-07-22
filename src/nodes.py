"""Graph nodes.

Each node is a pure-ish function: (state) -> partial state update. Keeping nodes
small and single-purpose is what makes the graph readable and testable.

The routing judgment is deliberately generic — the classifier answers one question
(*can the agent give a concrete, defensible answer now?*) and the graph branches on
the typed `Route`. Specific Pokémon phrasings live only as calibration examples in
the prompt, never as branches in code.
"""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .config import SETTINGS
from .llm import get_llm
from .schemas import Classification, QueryType, Route
from .state import AgentState
from .tools import AGENT_TOOLS

log = logging.getLogger("agent.nodes")

# How many recent messages the classifier sees for follow-up context. Enough to
# resolve "it"/"that one" without feeding it long tool-result blobs.
CLASSIFY_CONTEXT_TURNS = 6


# --------------------------------------------------------------------------- #
# 1. CLASSIFY — one generic judgment -> typed, branchable data.
# --------------------------------------------------------------------------- #
CLASSIFY_SYSTEM = (
    "You classify a user's latest turn in a conversation about Pokémon, using the "
    "PokéAPI as the source of truth. Make ONE judgment and set `route`:\n"
    "- answer: you could give a concrete, grounded answer now — a reasonable default "
    "exists even if the scope is broad. (e.g. 'What type is Pikachu?', 'Which Pokémon "
    "are weak to electric?' -> the metric is objective, just broad; 'What abilities can "
    "it have?' when a Pokémon was named earlier.)\n"
    "- clarify: a required choice is undefined with no sensible default, OR a reference "
    "cannot be resolved. (e.g. 'Which is stronger, Dragonite or Salamence?' -> 'stronger' "
    "is undefined; 'Tell me about it' with no earlier Pokémon.)\n"
    "- reject: the turn is not about Pokémon at all.\n"
    "Judge the KIND of ambiguity, not the breadth of the answer. Broad-but-objective is "
    "answer; undefined-criterion or unresolvable-reference is clarify."
)


def _recent_dialogue(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Recent human/AI turns only — skip tool traffic so the classifier sees clean
    conversational context for resolving follow-ups."""
    dialogue = [m for m in messages if isinstance(m, (HumanMessage, AIMessage)) and m.content]
    return dialogue[-CLASSIFY_CONTEXT_TURNS:]


def classify(state: AgentState) -> AgentState:
    """LLM step with structured output -> Classification (the routing decision).

    Also clears `final_reply`: it is a per-turn derived value, but the checkpointer
    persists every channel, so without a reset a turn that never sets a fresh reply
    (an error path, or a consumer reading the rolling state early) would surface the
    PREVIOUS turn's answer. classify runs first on every turn, so it's the natural
    place to mark "no reply produced yet".
    """
    llm = get_llm().with_structured_output(Classification)
    context = _recent_dialogue(state["messages"])
    result: Classification = llm.invoke([SystemMessage(content=CLASSIFY_SYSTEM), *context])
    log.info(
        "classified query_type=%s route=%s followup=%s reason=%s",
        result.query_type, result.route, result.is_followup, result.reason,
    )
    return {"classification": result, "final_reply": None}


# --------------------------------------------------------------------------- #
# Conditional edge: read classification, decide the branch. Pure, no LLM call.
# --------------------------------------------------------------------------- #
def route_after_classify(state: AgentState) -> str:
    classification: Classification = state["classification"]
    # Scope override: anything the model tags as off-topic is rejected outright,
    # regardless of the route it suggested. Determinism where it counts.
    if classification.query_type == QueryType.NOT_POKEMON:
        return Route.REJECT.value
    return classification.route.value


# --------------------------------------------------------------------------- #
# 2a. AGENT — tool-calling node. Loops with the ToolNode (see graph.py).
# --------------------------------------------------------------------------- #
AGENT_SYSTEM = (
    "You are a friendly, knowledgeable Pokémon assistant grounded in the PokéAPI.\n"
    "GROUNDING: Answer from tool results, not memory. For anything the tools cover — types, "
    "abilities, base stats, moves, evolutions, species data — call the tool and answer only "
    "from what it returns; never recite these from memory or invent Pokémon or move names. "
    "Re-fetch on follow-ups too (e.g. 'is that all its abilities?') rather than trusting an "
    "earlier turn. Only when a question falls OUTSIDE the API — competitive tiers, which "
    "Pokémon is 'strongest'/'best', lore or anime opinions — may you answer from general "
    "knowledge, and then say so briefly first (\"There's no official PokéAPI data on this, "
    "but generally...\").\n"
    "Resolve follow-ups from the conversation: 'it' or 'that one' means the Pokémon discussed "
    "earlier. Call tools as many times as needed before answering.\n"
    "ABILITIES vs MOVES: `get_pokemon` returns `abilities` (1-3 passive traits) alongside "
    "`movepool` (the count of attacks it learns). Report them together — never give the "
    "abilities without the movepool, so a user never mistakes a Pokémon's two abilities for "
    "the sum of what it can do.\n"
    "LIST SIZE: default to 5 examples and always state the true total count. If the user "
    "asks for a specific number ('top 10') or 'all', pass the tool's `limit` argument (a "
    "number, or None for all) — don't truncate silently or pad beyond what was asked.\n"
    "Be concise. Explain findings in plain language — never dump raw JSON — and state each "
    "fact only ONCE (don't list a set of types and then repeat it with examples). For broad "
    "type questions (e.g. 'which Pokémon are weak to electric?'), give the type-level answer "
    "with a few examples, then offer to check a specific Pokémon.\n"
    "If a tool returns an error, say so plainly (e.g. a possible misspelling) rather than "
    "inventing an answer."
)


TOOL_BUDGET_NUDGE = (
    "You have reached the tool-call limit for this turn. Do NOT request more data — "
    "answer now from the tool results already gathered, and say so briefly if something "
    "is incomplete."
)


def agent(state: AgentState) -> AgentState:
    """Let the model either call a tool or write the final reply.

    The tool budget is enforced HERE, not in the loop guard. Once `max_tool_iterations`
    tool rounds have run we invoke the model WITHOUT tools, so it cannot request another
    call we'd have to drop — dropping one would persist a tool_calls message with no
    matching ToolMessage and break the next turn. Withholding tools guarantees the loop
    terminates (no new tool_calls -> finalize) while keeping the history valid.

    The system prompt is injected at invoke time (not stored in state), so it never
    duplicates across turns of a persisted conversation.
    """
    messages = state["messages"]
    tool_rounds = sum(1 for m in messages if getattr(m, "type", None) == "tool")
    prompt: list[BaseMessage] = [SystemMessage(content=AGENT_SYSTEM)]

    llm = get_llm()
    if tool_rounds < SETTINGS.max_tool_iterations:
        llm = llm.bind_tools(AGENT_TOOLS)
    else:
        prompt.append(SystemMessage(content=TOOL_BUDGET_NUDGE))

    ai_msg = llm.invoke([*prompt, *messages])
    return {"messages": [ai_msg]}


def finalize_agent_reply(state: AgentState) -> AgentState:
    """Pull the last AI message out of the tool loop as the final reply."""
    last = state["messages"][-1]
    return {"final_reply": last.content, "handled_by": "agent"}


# --------------------------------------------------------------------------- #
# 2b. REJECT — out-of-scope. Deterministic, no LLM, no tokens spent.
# --------------------------------------------------------------------------- #
REJECT_REPLY = (
    "I'm a Pokémon assistant — I can answer questions about Pokémon, their types, "
    "abilities, stats, moves, and evolutions using live PokéAPI data. Ask me something "
    "like \"What are Charizard's base stats?\" or \"What does thunderbolt do?\""
)


def reject(state: AgentState) -> AgentState:
    """Politely decline anything that isn't about Pokémon."""
    log.info("rejected out-of-scope turn")
    return {
        "messages": [AIMessage(content=REJECT_REPLY)],
        "final_reply": REJECT_REPLY,
        "handled_by": "rejected",
    }


# --------------------------------------------------------------------------- #
# 2c. CLARIFY — ask one focused question when a required choice is undefined.
# --------------------------------------------------------------------------- #
CLARIFY_SYSTEM = (
    "The user's Pokémon question is under-specified: a required choice has no obvious "
    "default (e.g. 'stronger' could mean base stats, type advantage, or battle "
    "viability), or a reference can't be resolved. Ask ONE short, friendly question that "
    "would let you answer. Do not attempt to answer yet."
)


def clarify(state: AgentState) -> AgentState:
    """Generate a single clarifying question, grounded in the conversation."""
    llm = get_llm()
    context = _recent_dialogue(state["messages"])
    question = llm.invoke([SystemMessage(content=CLARIFY_SYSTEM), *context])
    return {
        "messages": [question],
        "final_reply": question.content,
        "handled_by": "clarify",
    }
