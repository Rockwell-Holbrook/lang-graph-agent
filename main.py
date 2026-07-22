"""CLI entry point — chat with the Pokémon agent.

Usage:
    python main.py                      # interactive multi-turn REPL
    python main.py "What type is Pikachu?"   # one-shot, then exit

The REPL keeps a single thread_id for the whole session, so follow-ups work:
    you> Tell me about Charizard
    you> What abilities can it have?     # "it" -> Charizard

Structural wiring can be checked without a key via `pytest` (no API calls).
"""
from __future__ import annotations

import logging
import sys
import uuid

from langchain_core.messages import HumanMessage

from src.config import SETTINGS
from src.graph import GRAPH

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def ask(message: str, thread_id: str) -> str:
    """Send one turn through the graph and return the agent's reply."""
    result = GRAPH.invoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result.get("final_reply") or "(no reply)"


def repl() -> None:
    thread_id = uuid.uuid4().hex
    print("Pokémon agent — ask me anything about Pokémon. Ctrl-D or 'quit' to exit.\n")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue
        if message.lower() in {"quit", "exit"}:
            return
        print(f"bot> {ask(message, thread_id)}\n")


def main() -> int:
    if not SETTINGS.has_api_key:
        print(
            "ERROR: OPENAI_API_KEY not set. Copy .env.example to .env and add your key. "
            "(To verify wiring without a key, run `pytest`.)",
            file=sys.stderr,
        )
        return 1

    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        print(ask(message, uuid.uuid4().hex))
    else:
        repl()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
