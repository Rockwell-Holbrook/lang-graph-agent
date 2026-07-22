"""Entry point. Run the triage agent against a message.

Usage:
    python main.py                      # runs a few built-in demo messages
    python main.py "your message here"  # runs one message
    python main.py "..." --customer cust_456
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.config import SETTINGS
from src.graph import GRAPH

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

DEMOS = [
    ("What are your hours on Saturday?", None),
    ("This is the THIRD time my invoice is wrong. Fix it now.", "cust_456"),
    ("hey", None),
]


def run_one(message: str, customer_id: str | None) -> None:
    print("\n" + "=" * 70)
    print(f"INBOUND: {message!r}  (customer={customer_id})")
    result = GRAPH.invoke(
        {"inbound_message": message, "customer_id": customer_id}
    )
    c = result.get("classification")
    if c:
        print(f"  intent={c.intent.value} sentiment={c.sentiment} "
              f"urgency={c.urgency} route={c.route.value}")
    print(f"  handled_by: {result.get('handled_by')}")
    print(f"  REPLY: {result.get('final_reply')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("message", nargs="?", help="Inbound customer message.")
    parser.add_argument("--customer", default=None, help="Customer id, e.g. cust_456")
    args = parser.parse_args()

    if not SETTINGS.has_api_key:
        print("ERROR: OPENAI_API_KEY not set. Copy .env.example to .env and add "
              "your key. (To verify wiring without a key, run smoke_test.py.)",
              file=sys.stderr)
        return 1

    if args.message:
        run_one(args.message, args.customer)
    else:
        for msg, cust in DEMOS:
            run_one(msg, cust)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
