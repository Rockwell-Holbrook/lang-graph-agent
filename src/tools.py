"""Tools the agent can call — the LLM-facing surface over the PokéAPI.

Each tool is a small, typed function whose docstring IS the spec the model reads,
so it must say *when* to use the tool and *what* it returns. The heavy lifting
(HTTP, caching, parsing, joining) lives in `pokeapi.py`; these wrappers just call
it and convert `PokeApiError` into a `{"error": ...}` result so a bad name or a
network blip degrades gracefully instead of throwing.

The module-level `_CLIENT` is the single shared client (so its cache is reused
across calls). Tests swap it for one backed by an `httpx.MockTransport`.
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from .pokeapi import PokeApiClient, PokeApiError

# One shared client per process — reused so the response cache actually helps.
_CLIENT = PokeApiClient()


def _safe(fn, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run a client call, turning a PokéAPI failure into a fail-safe error dict."""
    try:
        return fn(*args, **kwargs)
    except PokeApiError as exc:
        return {"error": str(exc)}


@tool
def get_pokemon(name_or_id: str) -> dict:
    """Get a Pokémon's core data: types, abilities, base stats, height and weight.

    Use this for questions about a specific Pokémon ("What type is Pikachu?",
    "Charizard's base stats", "What abilities can Bulbasaur have?"). Accepts a name
    or numeric id; spelling/casing is normalized. Returns a dict, or an 'error' key
    if the Pokémon is not found.
    """
    return _safe(_CLIENT.pokemon, name_or_id)


@tool
def get_pokemon_species(name_or_id: str) -> dict:
    """Get a Pokémon's SPECIES data: Pokédex flavor text, generation, legendary/
    mythical/baby flags, and what it evolves from.

    Species data differs from `get_pokemon` (which is the battle/form data) — use
    this for lore, rarity, or "what games/generation" style questions. Returns a
    dict, or an 'error' key if not found.
    """
    return _safe(_CLIENT.species, name_or_id)


@tool
def get_evolution_chain(name_or_id: str) -> dict:
    """Get a Pokémon's full evolution chain as a tree (handles branching lines).

    Use for "What is Squirtle's evolution chain?" or "Which Pokémon evolve from
    Eevee?". Follows the species -> evolution-chain link for you and returns a
    nested {name, via, evolves_to[]} structure where `via` is how each stage
    evolves (e.g. "level 16", "use water-stone", "trade"). 'error' key if not found.
    """
    return _safe(_CLIENT.evolution_chain, name_or_id)


@tool
def get_type_matchups(type_name: str) -> dict:
    """Get a type's damage relations and the Pokémon that have that type.

    Use for "Which Pokémon are weak to electric?" (answer at the type level:
    `strong_against` lists the types this type beats) and "Show all fire-type
    Pokémon" (`pokemon_of_type`). Directional: `strong_against` = deals 2x (those
    types are weak to it); `weak_to` = takes 2x. 'error' key if the type is unknown.
    """
    return _safe(_CLIENT.type_matchups, type_name)


@tool
def get_move_details(move_name: str) -> dict:
    """Get a move's type, power, accuracy, PP, damage class, and effect text.

    Use for "What is the effect of thunderbolt?" or "How strong is earthquake?".
    Returns a dict, or an 'error' key if the move is unknown.
    """
    return _safe(_CLIENT.move, move_name)


@tool
def search_pokemon_by_ability(ability_name: str) -> dict:
    """Find which Pokémon can have a given ability, plus what the ability does.

    Use for "Which Pokémon have the ability intimidate?". Returns the ability's
    effect and a (possibly truncated) list of Pokémon with a total count. 'error'
    key if the ability is unknown.
    """
    return _safe(_CLIENT.ability, ability_name)


@tool
def compare_pokemon(names: list[str]) -> dict:
    """Compare two or more Pokémon side by side on their base stats.

    Use for "Compare the base stats of Gengar and Alakazam." Fetches each Pokémon
    and returns their stats together with base-stat totals so you can explain the
    difference. Any name that fails to resolve appears under 'errors'.
    """
    compared: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name in names:
        result = _safe(_CLIENT.pokemon, name)
        if "error" in result:
            errors[name] = result["error"]
        else:
            compared[result["name"]] = {
                "types": result["types"],
                "base_stats": result["base_stats"],
                "base_stat_total": result["base_stat_total"],
            }
    out: dict[str, Any] = {"compared": compared}
    if errors:
        out["errors"] = errors
    return out


# Tools the tool-calling agent node is allowed to use.
AGENT_TOOLS = [
    get_pokemon,
    get_pokemon_species,
    get_evolution_chain,
    get_type_matchups,
    get_move_details,
    search_pokemon_by_ability,
    compare_pokemon,
]

# Registry (handy for tests and introspection).
ALL_TOOLS = {t.name: t for t in AGENT_TOOLS}
