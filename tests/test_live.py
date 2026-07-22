"""Live grounding checks against the real PokéAPI.

Deselected by default (see pyproject `addopts = -m 'not live'`). Run explicitly:

    pytest -m live

These catch drift between our parsers and the real API without making the default
suite network-dependent.
"""
from __future__ import annotations

import pytest

from src.pokeapi import PokeApiClient

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live_client():
    client = PokeApiClient()  # real httpx client from SETTINGS
    yield client
    client.close()


def test_pikachu_is_electric(live_client):
    assert live_client.pokemon("pikachu")["types"] == ["electric"]


def test_charizard_stat_total(live_client):
    assert live_client.pokemon("charizard")["base_stat_total"] == 534


def test_eevee_chain_branches(live_client):
    chain = live_client.evolution_chain("eevee")
    assert chain["base"] == "eevee"
    assert len(chain["chain"]["evolves_to"]) >= 5  # eeveelutions
