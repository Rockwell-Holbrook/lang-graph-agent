"""Shared test fixtures.

Two seams make the whole suite deterministic and offline (no API key, no network):

  1. `mock_client` / `patched_tools` — a `PokeApiClient` over an `httpx.MockTransport`
     that serves canned PokéAPI JSON, so client + tool tests exercise real parsing
     against realistic payloads.
  2. `ScriptedLLM` — a duck-typed stand-in for the chat model. `GenericFakeChatModel`
     cannot do `.with_structured_output()` or `.bind_tools()`, so we supply exactly
     what the nodes call: `.with_structured_output(M).invoke()` pops a scripted
     Pydantic object, `.bind_tools(t).invoke()` / `.invoke()` pop a scripted AIMessage.
"""
from __future__ import annotations

from typing import Any

import httpx
import pytest

from src.pokeapi import PokeApiClient

BASE_URL = "https://pokeapi.co/api/v2/"


# --------------------------------------------------------------------------- #
# Canned PokéAPI payloads (trimmed to the fields our parsers read).
# --------------------------------------------------------------------------- #
def _pokemon(name: str, id: int, types: list[str],
             abilities: list[tuple[str, bool]], stats: dict[str, int],
             height: int, weight: int, moves: list[str] | None = None) -> dict[str, Any]:
    return {
        "name": name, "id": id, "height": height, "weight": weight,
        "types": [{"slot": i + 1, "type": {"name": t}} for i, t in enumerate(types)],
        "abilities": [{"ability": {"name": a}, "is_hidden": hidden, "slot": i + 1}
                      for i, (a, hidden) in enumerate(abilities)],
        "stats": [{"base_stat": v, "effort": 0, "stat": {"name": k}}
                  for k, v in stats.items()],
        "moves": [{"move": {"name": m}, "version_group_details": []}
                  for m in (moves or [])],
    }


def _species(name: str, id: int, chain_url: str, *, evolves_from: str | None = None,
             legendary: bool = False, flavor: str = "A Pokémon.") -> dict[str, Any]:
    return {
        "name": name, "id": id, "is_legendary": legendary, "is_mythical": False,
        "is_baby": False, "generation": {"name": "generation-i"},
        "evolves_from_species": {"name": evolves_from} if evolves_from else None,
        "flavor_text_entries": [
            {"flavor_text": f"{flavor}\n\fWith­ line breaks.", "language": {"name": "en"}},
            {"flavor_text": "ignorer", "language": {"name": "fr"}},
        ],
        "evolution_chain": {"url": chain_url},
    }


def _node(name: str, evolves_to: list[dict], *, via: dict | None = None) -> dict[str, Any]:
    return {
        "species": {"name": name},
        "evolution_details": [via] if via else [],
        "evolves_to": evolves_to,
    }


_STATS = {"hp": 0, "attack": 0, "defense": 0, "special-attack": 0,
          "special-defense": 0, "speed": 0}


def _stats(**kw: int) -> dict[str, int]:
    return {**_STATS, **kw}


POKEAPI_FIXTURES: dict[str, dict[str, Any]] = {
    "pokemon/pikachu": _pokemon(
        "pikachu", 25, ["electric"], [("static", False), ("lightning-rod", True)],
        _stats(hp=35, attack=55, defense=40, **{"special-attack": 50, "special-defense": 50}, speed=90),
        4, 60),
    "pokemon/charizard": _pokemon(
        "charizard", 6, ["fire", "flying"], [("blaze", False), ("solar-power", True)],
        _stats(hp=78, attack=84, defense=78, **{"special-attack": 109, "special-defense": 85}, speed=100),
        17, 905, moves=["flamethrower", "fly", "dragon-claw", "air-slash", "fire-spin", "wing-attack"]),
    "pokemon/bulbasaur": _pokemon(
        "bulbasaur", 1, ["grass", "poison"], [("overgrow", False), ("chlorophyll", True)],
        _stats(hp=45, attack=49, defense=49, **{"special-attack": 65, "special-defense": 65}, speed=45),
        7, 69),
    "pokemon/gengar": _pokemon(
        "gengar", 94, ["ghost", "poison"], [("cursed-body", False)],
        _stats(hp=60, attack=65, defense=60, **{"special-attack": 130, "special-defense": 75}, speed=110),
        15, 405),
    "pokemon/alakazam": _pokemon(
        "alakazam", 65, ["psychic"], [("synchronize", False), ("inner-focus", False)],
        _stats(hp=55, attack=50, defense=45, **{"special-attack": 135, "special-defense": 95}, speed=120),
        15, 480),
    "pokemon/eevee": _pokemon(
        "eevee", 133, ["normal"], [("run-away", False), ("adaptability", False), ("anticipation", True)],
        _stats(hp=55, attack=55, defense=50, **{"special-attack": 45, "special-defense": 65}, speed=55),
        3, 65),

    "pokemon-species/charizard": _species(
        "charizard", 6, f"{BASE_URL}evolution-chain/2/", evolves_from="charmeleon",
        flavor="Spits fire hot enough to melt boulders."),
    "pokemon-species/eevee": _species("eevee", 133, f"{BASE_URL}evolution-chain/67/"),
    "pokemon-species/squirtle": _species("squirtle", 7, f"{BASE_URL}evolution-chain/3/"),

    # Charmander line (linear): charmander -> charmeleon(16) -> charizard(36).
    "evolution-chain/2": {"id": 2, "chain": _node(
        "charmander",
        [_node("charmeleon",
               [_node("charizard", [], via={"min_level": 36, "trigger": {"name": "level-up"}})],
               via={"min_level": 16, "trigger": {"name": "level-up"}})],
    )},
    # Eevee line (branching): several stone/friendship evolutions.
    "evolution-chain/67": {"id": 67, "chain": _node(
        "eevee",
        [
            _node("vaporeon", [], via={"item": {"name": "water-stone"}, "trigger": {"name": "use-item"}}),
            _node("jolteon", [], via={"item": {"name": "thunder-stone"}, "trigger": {"name": "use-item"}}),
            _node("flareon", [], via={"item": {"name": "fire-stone"}, "trigger": {"name": "use-item"}}),
            _node("espeon", [], via={"min_happiness": 160, "trigger": {"name": "level-up"}}),
        ],
    )},
    # Squirtle line (linear).
    "evolution-chain/3": {"id": 3, "chain": _node(
        "squirtle",
        [_node("wartortle",
               [_node("blastoise", [], via={"min_level": 36, "trigger": {"name": "level-up"}})],
               via={"min_level": 16, "trigger": {"name": "level-up"}})],
    )},

    "type/electric": {
        "name": "electric",
        "damage_relations": {
            "double_damage_to": [{"name": "water"}, {"name": "flying"}],
            "double_damage_from": [{"name": "ground"}],
            "half_damage_to": [{"name": "electric"}, {"name": "grass"}, {"name": "dragon"}],
            "half_damage_from": [{"name": "electric"}, {"name": "flying"}, {"name": "steel"}],
            "no_damage_to": [{"name": "ground"}],
            "no_damage_from": [],
        },
        "pokemon": [{"slot": 1, "pokemon": {"name": n}} for n in ["pikachu", "raichu", "voltorb"]],
    },

    "move/thunderbolt": {
        "name": "thunderbolt", "power": 90, "accuracy": 100, "pp": 15,
        "type": {"name": "electric"}, "damage_class": {"name": "special"},
        "effect_chance": 10,
        "effect_entries": [{
            "language": {"name": "en"},
            "short_effect": "Has a $effect_chance% chance to paralyze the target.",
            "effect": "Inflicts regular damage. Has a $effect_chance% chance to paralyze.",
        }],
    },

    "ability/intimidate": {
        "name": "intimidate",
        "effect_entries": [{
            "language": {"name": "en"},
            "short_effect": "Lowers the target's Attack by one stage on entry.",
            "effect": "On entry, lowers the Attack of adjacent opponents by one stage.",
        }],
        "pokemon": [{"is_hidden": False, "slot": 1, "pokemon": {"name": n}}
                    for n in ["growlithe", "arcanine", "gyarados"]],
    },
}


# --------------------------------------------------------------------------- #
# HTTP mocking
# --------------------------------------------------------------------------- #
def _handler(request: httpx.Request) -> httpx.Response:
    key = request.url.path.replace("/api/v2/", "").strip("/")
    if key in POKEAPI_FIXTURES:
        return httpx.Response(200, json=POKEAPI_FIXTURES[key])
    return httpx.Response(404, json={"detail": "Not Found."})


@pytest.fixture
def mock_client() -> PokeApiClient:
    """A PokeApiClient backed by canned fixtures — no network, no key."""
    http = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(_handler))
    return PokeApiClient(client=http)


@pytest.fixture
def patched_tools(monkeypatch, mock_client) -> PokeApiClient:
    """Point the tool layer's shared client at the mock, for tool/graph tests."""
    import src.tools as tools_mod
    monkeypatch.setattr(tools_mod, "_CLIENT", mock_client)
    return mock_client


# --------------------------------------------------------------------------- #
# Scripted chat model
# --------------------------------------------------------------------------- #
class _ScriptedRunnable:
    """What `.with_structured_output(M)` / `.bind_tools(t)` return: pops a queue."""

    def __init__(self, queue: list[Any], label: str) -> None:
        self._queue = queue
        self._label = label

    def invoke(self, *_args: Any, **_kwargs: Any) -> Any:
        assert self._queue, f"ScriptedLLM ran out of {self._label} responses"
        return self._queue.pop(0)


class ScriptedLLM:
    """Duck-typed chat model. Feed it scripted structured outputs (for `classify`)
    and AIMessages (for `agent` tool calls / final replies and `clarify`)."""

    def __init__(self, *, structured: list[Any] | None = None,
                 ai: list[Any] | None = None) -> None:
        self._structured = list(structured or [])
        self._ai = list(ai or [])
        self.bound_tools: Any = None  # records the last .bind_tools() argument, for tests

    def with_structured_output(self, _model: Any) -> _ScriptedRunnable:
        return _ScriptedRunnable(self._structured, "structured")

    def bind_tools(self, tools: Any) -> _ScriptedRunnable:
        self.bound_tools = tools
        return _ScriptedRunnable(self._ai, "AI message")

    def invoke(self, *_args: Any, **_kwargs: Any) -> Any:  # plain path (clarify node)
        assert self._ai, "ScriptedLLM ran out of AI messages"
        return self._ai.pop(0)


@pytest.fixture
def scripted_llm(monkeypatch):
    """Install a ScriptedLLM as the model factory for the node layer."""
    def _install(**kwargs: Any) -> ScriptedLLM:
        llm = ScriptedLLM(**kwargs)
        import src.nodes as nodes_mod
        monkeypatch.setattr(nodes_mod, "get_llm", lambda: llm)
        return llm
    return _install
