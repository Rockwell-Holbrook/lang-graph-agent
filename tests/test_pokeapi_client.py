"""PokeApiClient: request construction, parsing/cleaning, caching, and fail-safe errors."""
from __future__ import annotations

import httpx
import pytest

from src.pokeapi import PokeApiClient, PokeApiError, normalize_name

from .conftest import BASE_URL


def test_normalize_name_handles_spacing_and_case():
    assert normalize_name("Mr Mime") == "mr-mime"
    assert normalize_name("  CHARIZARD ") == "charizard"
    assert normalize_name("ho_oh") == "ho-oh"
    assert normalize_name(6) == "6"


def test_pokemon_is_parsed_into_clean_shape(mock_client):
    poke = mock_client.pokemon("Charizard")  # casing normalized on the way in
    assert poke["name"] == "charizard"
    assert poke["types"] == ["fire", "flying"]
    assert poke["base_stats"]["special-attack"] == 109
    assert poke["base_stat_total"] == 534
    assert poke["height_m"] == 1.7 and poke["weight_kg"] == 90.5
    assert {"name": "solar-power", "is_hidden": True} in poke["abilities"]


def test_species_flavor_text_is_cleaned(mock_client):
    species = mock_client.species("charizard")
    # Line breaks / form feeds / soft hyphens collapsed to single spaces.
    assert "\n" not in species["flavor_text"] and "\f" not in species["flavor_text"]
    assert species["flavor_text"] == "Spits fire hot enough to melt boulders. With line breaks."
    assert species["evolves_from"] == "charmeleon"


def test_move_effect_substitutes_effect_chance(mock_client):
    move = mock_client.move("thunderbolt")
    assert move["power"] == 90 and move["damage_class"] == "special"
    assert move["effect"] == "Has a 10% chance to paralyze the target."


def test_type_matchups_direction_and_pokemon_list(mock_client):
    m = mock_client.type_matchups("electric")
    assert m["strong_against"] == ["water", "flying"]   # these types are weak to electric
    assert m["weak_to"] == ["ground"]
    assert "pikachu" in m["pokemon_of_type"]
    assert m["pokemon_of_type_count"] == 3


def test_404_raises_pokeapi_error_with_helpful_message(mock_client):
    with pytest.raises(PokeApiError, match="Not found"):
        mock_client.pokemon("charizrd")  # misspelled -> 404


def test_network_error_becomes_pokeapi_error():
    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow")

    http = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(boom))
    client = PokeApiClient(client=http)
    with pytest.raises(PokeApiError, match="Could not reach PokéAPI"):
        client.pokemon("pikachu")


def test_successful_responses_are_cached(mock_client):
    calls = {"n": 0}
    original = mock_client._client.get

    def counting_get(url):
        calls["n"] += 1
        return original(url)

    mock_client._client.get = counting_get  # type: ignore[assignment]
    mock_client.pokemon("pikachu")
    mock_client.pokemon("pikachu")
    assert calls["n"] == 1  # second call served from cache

    mock_client.clear_cache()
    mock_client.pokemon("pikachu")
    assert calls["n"] == 2  # cache cleared -> fetched again
