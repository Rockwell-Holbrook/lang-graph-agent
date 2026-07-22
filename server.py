"""FastAPI transport for the agent.

This is the architectural boundary: the graph knows nothing about HTTP or the UI.
The endpoint streams the agent's response token-by-token over Server-Sent Events
(SSE), so any client — the bundled index.html, curl, or something else — can attach
without touching agent code.

Run:
    uvicorn server:app --reload
    # then open http://localhost:8000
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from src.graph import GRAPH

app = FastAPI(title="LangGraph Agent")

WEB_DIR = Path(__file__).parent / "web"


class ChatRequest(BaseModel):
    message: str
    customer_id: Optional[str] = None


def _sse(obj: dict) -> str:
    """Encode one Server-Sent Event."""
    return f"data: {json.dumps(obj)}\n\n"


def _event_stream(req: ChatRequest) -> Iterator[str]:
    """Drive the graph and translate its stream into SSE events.

    We use two LangGraph stream modes at once:
      - "messages": token-level LLM output (for live typing)
      - "values":   the rolling state (for the classification chip + final reply)

    Only free-text from the `agent` node is streamed as tokens; the escalate/clarify
    paths produce their reply without a token stream, so we emit it once at the end.
    """
    payload = {"inbound_message": req.message, "customer_id": req.customer_id}
    streamed_any = False
    final_reply: Optional[str] = None
    meta_sent = False

    try:
        for mode, data in GRAPH.stream(payload, stream_mode=["messages", "values"]):
            if mode == "messages":
                msg, metadata = data
                if metadata.get("langgraph_node") == "agent":
                    content = getattr(msg, "content", "") or ""
                    if content:
                        streamed_any = True
                        yield _sse({"type": "token", "text": content})

            elif mode == "values":
                classification = data.get("classification")
                if classification is not None and not meta_sent:
                    yield _sse({
                        "type": "meta",
                        "intent": classification.intent.value,
                        "route": classification.route.value,
                        "urgency": classification.urgency,
                    })
                    meta_sent = True
                if data.get("final_reply"):
                    final_reply = data["final_reply"]

        # escalate / clarify paths (no token stream): emit the reply now.
        if not streamed_any and final_reply:
            yield _sse({"type": "token", "text": final_reply})

        yield _sse({"type": "done"})

    except Exception as exc:  # surface errors to the UI instead of hanging
        yield _sse({"type": "error", "text": f"{type(exc).__name__}: {exc}"})


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(_event_stream(req), media_type="text/event-stream")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
