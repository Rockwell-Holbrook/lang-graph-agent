"""The deterministic router: branch selection + the not_pokemon scope override.

The classifier's *criterion-vs-scope* judgment is exercised with scripted
Classifications so these tests stay deterministic (no LLM). The point is that once
the LLM has judged, the graph routes correctly and generically on the enum.
"""
from __future__ import annotations

import pytest

from src.nodes import route_after_classify
from src.schemas import Classification, QueryType, Route


def _cls(query_type: QueryType, route: Route) -> Classification:
    return Classification(query_type=query_type, route=route, reason="test")


def test_answerable_query_routes_to_agent():
    cls = _cls(QueryType.POKEMON_INFO, Route.ANSWER)
    assert route_after_classify({"classification": cls}) == "answer"


def test_clarify_routes_to_clarify():
    cls = _cls(QueryType.OTHER_POKEMON, Route.CLARIFY)
    assert route_after_classify({"classification": cls}) == "clarify"


def test_not_pokemon_is_forced_to_reject_even_if_model_said_answer():
    """Deterministic safety override: off-topic never reaches the agent."""
    cls = _cls(QueryType.NOT_POKEMON, Route.ANSWER)
    assert route_after_classify({"classification": cls}) == "reject"


@pytest.mark.parametrize(
    "route, expected",
    [
        # Objective metric, broad scope ("weak to electric") -> answer.
        (Route.ANSWER, "answer"),
        # Undefined criterion ("which is stronger?") -> clarify.
        (Route.CLARIFY, "clarify"),
    ],
)
def test_comparison_criterion_vs_scope(route, expected):
    """A `comparison` turn can be either answer or clarify depending on whether the
    criterion is defined — the router honors whichever the classifier chose."""
    cls = _cls(QueryType.COMPARISON, route)
    assert route_after_classify({"classification": cls}) == expected
