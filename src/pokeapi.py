"""PokéAPI client layer.

This is the transport + parsing boundary, deliberately separate from `tools.py`
(the LLM-facing surface) so it can be unit-tested without LangChain: construct a
`PokeApiClient` over an `httpx.MockTransport` and assert on cleaned output.

Responsibilities (per AGENT_IDENTITY §3):
  - request construction + input normalization ("Mr Mime" -> "mr-mime")
  - fetching from the relevant endpoints, following related URLs (evolution chains)
  - response parsing / cleaning into small, user-friendly dicts (never raw JSON)
  - caching (PokéAPI is near-static and asks clients to cache) + intra-turn dedupe
  - fail-safe errors: HTTP/network failures raise `PokeApiError`, which the tool
    layer turns into `{"error": ...}` — nothing hangs or leaks a stack trace.
"""
from __future__ import annotations

import re
from typing import Any, Optional

import httpx

from .config import SETTINGS

# Cap embedded lists so a "all fire-type" style answer stays digestible; the full
# count is always reported alongside the (possibly truncated) sample.
MAX_LIST_ITEMS = 40


class PokeApiError(Exception):
    """A PokéAPI request failed (not found, network error, timeout, bad status)."""


def normalize_name(name_or_id: str | int) -> str:
    """Normalize user input into a PokéAPI resource key.

    PokéAPI keys are lowercase, hyphen-separated ("mr-mime", "ho-oh"). We lowercase,
    trim, and collapse spaces/underscores to hyphens so "Mr Mime", "charizard " and
    "MEWTWO" all resolve. Numeric ids pass through unchanged.
    """
    text = str(name_or_id).strip().lower()
    text = re.sub(r"[\s_]+", "-", text)
    return text


class PokeApiClient:
    """Thin, cached wrapper over the PokéAPI.

    The underlying `httpx.Client` is injectable so tests can supply an
    `httpx.MockTransport`; in production it is built from `SETTINGS`.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._base_url = base_url or SETTINGS.pokeapi_base_url
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self._base_url,
            timeout=timeout if timeout is not None else SETTINGS.http_timeout,
            headers={"User-Agent": "langgraph-pokemon-agent"},
        )
        # Per-instance cache keyed by request path/URL. Successes only, so a
        # transient failure is retried on the next call rather than memoized.
        self._cache: dict[str, dict[str, Any]] = {}

    # -- low-level fetch ---------------------------------------------------- #
    def get_json(self, path_or_url: str) -> dict[str, Any]:
        """GET a path (relative to base) or absolute URL and return parsed JSON.

        Raises `PokeApiError` on 404, other bad status, network error, or timeout.
        Caches successful responses by key.
        """
        if path_or_url in self._cache:
            return self._cache[path_or_url]
        try:
            resp = self._client.get(path_or_url)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise PokeApiError(
                    f"Not found: {path_or_url!r}. Check the spelling — PokéAPI uses "
                    f"lowercase hyphenated names (e.g. 'mr-mime')."
                ) from exc
            raise PokeApiError(
                f"PokéAPI returned HTTP {exc.response.status_code} for {path_or_url!r}."
            ) from exc
        except httpx.RequestError as exc:  # includes timeouts and connection errors
            raise PokeApiError(f"Could not reach PokéAPI for {path_or_url!r}: {exc}") from exc

        self._cache[path_or_url] = data
        return data

    def clear_cache(self) -> None:
        """Drop the response cache (used by tests to keep runs independent)."""
        self._cache.clear()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # -- cleaned, domain-level reads --------------------------------------- #
    def pokemon(self, name_or_id: str | int) -> dict[str, Any]:
        """`/pokemon/{}` -> types, abilities, base stats, physical attributes."""
        raw = self.get_json(f"pokemon/{normalize_name(name_or_id)}")
        stats = {s["stat"]["name"]: s["base_stat"] for s in raw.get("stats", [])}
        return {
            "name": raw["name"],
            "id": raw["id"],
            "types": [t["type"]["name"] for t in raw.get("types", [])],
            "abilities": [
                {"name": a["ability"]["name"], "is_hidden": a["is_hidden"]}
                for a in raw.get("abilities", [])
            ],
            "base_stats": stats,
            "base_stat_total": sum(stats.values()),
            "height_m": raw.get("height", 0) / 10,   # decimetres -> metres
            "weight_kg": raw.get("weight", 0) / 10,   # hectograms -> kilograms
        }

    def species(self, name_or_id: str | int) -> dict[str, Any]:
        """`/pokemon-species/{}` -> flavor text, rarity flags, evolution-chain link."""
        raw = self.get_json(f"pokemon-species/{normalize_name(name_or_id)}")
        return {
            "name": raw["name"],
            "id": raw["id"],
            "is_legendary": raw.get("is_legendary", False),
            "is_mythical": raw.get("is_mythical", False),
            "is_baby": raw.get("is_baby", False),
            "generation": (raw.get("generation") or {}).get("name"),
            "evolves_from": (raw.get("evolves_from_species") or {}).get("name")
            if raw.get("evolves_from_species")
            else None,
            "flavor_text": _first_english_flavor(raw.get("flavor_text_entries", [])),
            "evolution_chain_url": (raw.get("evolution_chain") or {}).get("url"),
        }

    def evolution_chain(self, name_or_id: str | int) -> dict[str, Any]:
        """Species -> follow the evolution-chain URL -> parse the branching tree.

        The chain is a recursive `evolves_to[]` tree, not a flat list — Eevee, for
        instance, branches into many forms. We return a nested, readable structure
        that preserves branches, plus the base species name.
        """
        species = self.species(name_or_id)
        url = species.get("evolution_chain_url")
        if not url:
            raise PokeApiError(f"No evolution chain for {species['name']!r}.")
        raw = self.get_json(url)
        tree = _parse_chain_node(raw["chain"], via=None)
        return {"base": tree["name"], "chain": tree}

    def type_matchups(self, type_name: str) -> dict[str, Any]:
        """`/type/{}` -> damage relations (both directions) + the Pokémon of that type.

        Naming is from this type's perspective: `strong_against` are the types it
        deals 2x to (so those types are *weak to* it); `weak_to` are the types that
        deal 2x to it. `pokemon_of_type` answers "show all X-type Pokémon" directly.
        """
        raw = self.get_json(f"type/{normalize_name(type_name)}")
        rel = raw.get("damage_relations", {})
        names = lambda key: [d["name"] for d in rel.get(key, [])]  # noqa: E731
        pokemon = [p["pokemon"]["name"] for p in raw.get("pokemon", [])]
        return {
            "type": raw["name"],
            "strong_against": names("double_damage_to"),   # these types are weak to it
            "resisted_by": names("half_damage_to"),
            "no_effect_against": names("no_damage_to"),
            "weak_to": names("double_damage_from"),         # these types beat it
            "resists": names("half_damage_from"),
            "immune_to": names("no_damage_from"),
            "pokemon_of_type": pokemon[:MAX_LIST_ITEMS],
            "pokemon_of_type_count": len(pokemon),
        }

    def move(self, move_name: str) -> dict[str, Any]:
        """`/move/{}` -> type, power, accuracy, pp, damage class, effect."""
        raw = self.get_json(f"move/{normalize_name(move_name)}")
        return {
            "name": raw["name"],
            "type": (raw.get("type") or {}).get("name"),
            "power": raw.get("power"),
            "accuracy": raw.get("accuracy"),
            "pp": raw.get("pp"),
            "damage_class": (raw.get("damage_class") or {}).get("name"),
            "effect": _first_english_effect(
                raw.get("effect_entries", []), raw.get("effect_chance")
            ),
        }

    def ability(self, ability_name: str) -> dict[str, Any]:
        """`/ability/{}` -> effect text + the Pokémon that can have it."""
        raw = self.get_json(f"ability/{normalize_name(ability_name)}")
        pokemon = [p["pokemon"]["name"] for p in raw.get("pokemon", [])]
        return {
            "name": raw["name"],
            "effect": _first_english_effect(raw.get("effect_entries", []), None),
            "pokemon": pokemon[:MAX_LIST_ITEMS],
            "pokemon_count": len(pokemon),
        }


# --------------------------------------------------------------------------- #
# Parsing helpers (module-level, pure — easy to unit test in isolation).
# --------------------------------------------------------------------------- #
def _clean_text(text: str) -> str:
    """PokéAPI flavor/effect text is littered with hard line breaks and form feeds."""
    return re.sub(r"\s+", " ", re.sub("­", "", text)).strip()


def _first_english_flavor(entries: list[dict[str, Any]]) -> Optional[str]:
    for entry in entries:
        if (entry.get("language") or {}).get("name") == "en":
            return _clean_text(entry["flavor_text"])
    return None


def _first_english_effect(entries: list[dict[str, Any]], effect_chance: Optional[int]) -> Optional[str]:
    for entry in entries:
        if (entry.get("language") or {}).get("name") == "en":
            text = _clean_text(entry.get("short_effect") or entry.get("effect", ""))
            if effect_chance is not None:
                text = text.replace("$effect_chance", str(effect_chance))
            return text
    return None


def _parse_chain_node(node: dict[str, Any], *, via: Optional[str]) -> dict[str, Any]:
    """Recursively turn a PokéAPI evolution-chain node into a readable tree.

    `via` describes how this node evolves from its parent (e.g. "level 16",
    "use water-stone", "trade"); it is None for the base species.
    """
    return {
        "name": node["species"]["name"],
        "via": via,
        "evolves_to": [
            _parse_chain_node(child, via=_evolution_trigger(child.get("evolution_details", [])))
            for child in node.get("evolves_to", [])
        ],
    }


def _evolution_trigger(details: list[dict[str, Any]]) -> Optional[str]:
    """Summarize the first evolution trigger into a short human phrase."""
    if not details:
        return None
    d = details[0]
    if d.get("min_level"):
        return f"level {d['min_level']}"
    if d.get("item"):
        return f"use {d['item']['name']}"
    trigger = (d.get("trigger") or {}).get("name")
    if trigger == "trade":
        return "trade"
    if d.get("min_happiness"):
        return "high friendship"
    return trigger.replace("-", " ") if trigger else None
