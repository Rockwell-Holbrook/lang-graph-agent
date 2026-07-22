"""Central configuration.

One place to control model, temperature, and limits so nothing is hard-coded
inside the graph. Everything reads from environment (.env), with sane defaults.
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

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key)


def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        max_tool_iterations=int(os.getenv("MAX_TOOL_ITERATIONS", "5")),
    )


SETTINGS = load_settings()
