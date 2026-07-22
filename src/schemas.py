"""Structured output schemas.

We force the LLM into these shapes with `.with_structured_output(...)` instead of
parsing free text. This is the single biggest reliability win in an agent: the
model's decision becomes typed data the graph can branch on deterministically.

The classifier makes ONE generic judgment per turn — *can the agent produce a
concrete, defensible answer from the tools with what it knows right now?* — and
the graph routes on the resulting `Route` enum. See `nodes.CLASSIFY_SYSTEM`.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Coarse kind of the user's turn. Drives the meta chip / observability and
    the deterministic `not_pokemon -> reject` scope override; it does NOT encode
    control flow on its own (that is `Route`)."""

    POKEMON_INFO = "pokemon_info"      # type / abilities / stats of a specific Pokémon
    EVOLUTION = "evolution"            # evolution chain or "what evolves from X"
    TYPE_MATCHUP = "type_matchup"      # type strengths / weaknesses, "all X-type"
    MOVE = "move"                      # a move's effect / power / accuracy
    COMPARISON = "comparison"          # compare two or more Pokémon
    ABILITY_SEARCH = "ability_search"  # which Pokémon have ability X
    OTHER_POKEMON = "other_pokemon"    # Pokémon-related but uncategorized
    NOT_POKEMON = "not_pokemon"        # out of scope


class Route(str, Enum):
    """Where the graph sends this turn after classification."""

    ANSWER = "answer"      # agent can produce a grounded answer (may use tools)
    CLARIFY = "clarify"    # a required choice is undefined; ask ONE focused question
    REJECT = "reject"      # not about Pokémon; decline politely


class Classification(BaseModel):
    """Typed result of the classify step — the graph branches on `route`."""

    query_type: QueryType = Field(description="Coarse kind of the user's turn.")
    route: Route = Field(
        description=(
            "answer = you can give a concrete, defensible answer now (a reasonable "
            "default exists even if scope is broad). clarify = a required choice is "
            "undefined with no sensible default, or a reference like 'it' cannot be "
            "resolved. reject = the turn is not about Pokémon."
        )
    )
    is_followup: bool = Field(
        default=False,
        description="True if this turn depends on earlier turns (e.g. 'it', 'that one').",
    )
    reason: str = Field(description="One short sentence justifying the route.")
