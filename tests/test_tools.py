"""The @tool wrappers: clean output, fail-safe errors, and multi-endpoint joins."""
from __future__ import annotations

import pytest

from src.tools import (
    ALL_TOOLS,
    compare_pokemon,
    get_evolution_chain,
    get_move_details,
    get_pokemon,
    get_type_matchups,
    search_pokemon_by_ability,
)

# Every test here needs the tool layer pointed at the mock PokéAPI.
pytestmark = pytest.mark.usefixtures("patched_tools")


def test_registry_lists_all_agent_tools():
    assert set(ALL_TOOLS) == {
        "get_pokemon", "get_pokemon_species", "get_evolution_chain",
        "get_type_matchups", "get_move_details", "search_pokemon_by_ability",
        "compare_pokemon",
    }


def test_get_pokemon_returns_clean_dict():
    out = get_pokemon.invoke({"name_or_id": "pikachu"})
    assert out["types"] == ["electric"]
    assert out["base_stat_total"] == 320


def test_unknown_pokemon_yields_error_not_exception():
    out = get_pokemon.invoke({"name_or_id": "missingno"})
    assert "error" in out and "Not found" in out["error"]


def test_evolution_chain_preserves_branches():
    """Eevee is the branching case the parser must not flatten."""
    out = get_evolution_chain.invoke({"name_or_id": "eevee"})
    assert out["base"] == "eevee"
    branches = {c["name"]: c["via"] for c in out["chain"]["evolves_to"]}
    assert branches["vaporeon"] == "use water-stone"
    assert branches["espeon"] == "high friendship"
    assert len(out["chain"]["evolves_to"]) == 4


def test_evolution_chain_linear_line_reads_in_order():
    out = get_evolution_chain.invoke({"name_or_id": "squirtle"})
    wartortle = out["chain"]["evolves_to"][0]
    assert wartortle["name"] == "wartortle" and wartortle["via"] == "level 16"
    assert wartortle["evolves_to"][0]["name"] == "blastoise"


def test_type_matchups_exposes_relations_and_type_roster():
    out = get_type_matchups.invoke({"type_name": "electric"})
    assert out["strong_against"] == ["water", "flying"]
    assert out["pokemon_of_type"][0] == "pikachu"


def test_move_details_reads_effect_and_stats():
    out = get_move_details.invoke({"move_name": "Thunderbolt"})
    assert out["type"] == "electric" and out["power"] == 90
    assert "paralyze" in out["effect"]


def test_search_pokemon_by_ability_lists_holders():
    out = search_pokemon_by_ability.invoke({"ability_name": "intimidate"})
    assert "gyarados" in out["pokemon"]
    assert out["pokemon_count"] == 3


def test_compare_pokemon_puts_stats_side_by_side():
    out = compare_pokemon.invoke({"names": ["gengar", "alakazam"]})
    assert set(out["compared"]) == {"gengar", "alakazam"}
    assert out["compared"]["gengar"]["base_stat_total"] == 500
    assert out["compared"]["alakazam"]["base_stats"]["speed"] == 120
    assert "errors" not in out


def test_compare_pokemon_reports_bad_names_without_failing_the_rest():
    out = compare_pokemon.invoke({"names": ["gengar", "notamon"]})
    assert "gengar" in out["compared"]
    assert "notamon" in out["errors"]
