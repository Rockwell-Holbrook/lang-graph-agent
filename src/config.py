"""Central configuration.

One place to control model, temperature, limits, and the PokéAPI endpoint so
nothing is hard-coded inside the graph or tools. Everything reads from environment
(.env), with sane defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env once, at import time.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    model: str
    temperature: float
    max_tool_iterations: int
    pokeapi_base_url: str
    http_timeout: float

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key)


def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        max_tool_iterations=int(os.getenv("MAX_TOOL_ITERATIONS", "5")),
        pokeapi_base_url=os.getenv("POKEAPI_BASE_URL", "https://pokeapi.co/api/v2/"),
        http_timeout=float(os.getenv("HTTP_TIMEOUT", "15")),
    )


SETTINGS = load_settings()
