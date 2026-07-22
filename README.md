# LangGraph Agent Scaffold

A small, production-shaped LangGraph agent you can reuse as a starting point for an
open-ended "build an agent" exercise. The domain here is **inbound customer-message
triage** (a common real-world domain), but the architecture is the point — swap the
schemas, tools, and prompts to retarget it in minutes.

## Why a graph and not a chain

A linear chain can't *decide* and can't *loop*. This task needs both:

- **Branch** after classification — respond vs. escalate vs. ask a clarifying question.
- **Cycle** in the response path — the agent calls a tool, sees the result, and
  decides again, until it has enough to answer.

That branching + cycling is exactly what `StateGraph` gives you, with a typed state
object as the single source of truth flowing through every node.

## Architecture

```
START
  │
  ▼
classify ──► route_after_classify ──┬─► escalate ─► END      (open ticket, human follows up)
 (LLM,       (deterministic,        ├─► clarify  ─► END      (ask one follow-up question)
  typed       no LLM call)          └─► agent ⇄ tools (cycle) ─► finalize ─► END
  output)                                (tool-calling loop with hard iteration cap)
```

| Concern | Where | Why it matters |
|---|---|---|
| **Typed state** | `src/state.py` | One `AgentState` TypedDict flows everywhere; `messages` uses the `add_messages` reducer so tool turns accumulate. |
| **Structured output** | `src/schemas.py` + `classify`/`clarify` nodes | The model returns a validated `Classification`, not free text — the graph branches on typed data. |
| **Deterministic routing** | `route_after_classify` in `src/nodes.py` | Control flow is code, not vibes. Includes a safety override: high-urgency never auto-responds. |
| **Tools** | `src/tools.py` | Plain typed functions; the docstring is the LLM-facing spec. Mocked so the scaffold runs offline. |
| **Terminating loop** | `should_continue` in `src/graph.py` | Agent⇄tools cycle with a hard `MAX_TOOL_ITERATIONS` cap — never loops forever. |
| **Swappable model** | `src/llm.py` + `src/config.py` | Provider/model/temperature live in one place, read from env. |
| **Observability** | `logging` in nodes | Each node logs its decision. Flip on LangSmith via env for full traces. |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on mac/linux
pip install -r requirements.txt
copy .env.example .env            # then paste your OPENAI_API_KEY
```

## Run

```bash
# No key needed — proves the wiring is sound (structure, routing, loop, tools):
python smoke_test.py

# CLI, with a key in .env:
python main.py                                   # runs 3 demo messages
python main.py "What are your Saturday hours?"
python main.py "My invoice is wrong again!" --customer cust_456
```

### Chat UI

The graph is exposed over an HTTP boundary so any client can attach. `server.py` is
a FastAPI app that **streams the reply token-by-token over SSE**; `web/index.html` is
a dependency-free full-screen chat that consumes it.

```bash
uvicorn server:app --reload
# open http://localhost:8000
```

The agent knows nothing about HTTP or the UI — that separation is the point. The
same endpoint is drivable from curl:

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"My invoice is wrong again!\",\"customer_id\":\"cust_456\"}"
```

Without a key the endpoint streams a graceful `error` event rather than crashing.

## Extending it on the day

1. **Retarget the domain** — rewrite `schemas.py` (the decision shape) and the
   node system prompts. The graph topology usually survives unchanged.
2. **Add a tool** — write a `@tool` function in `tools.py`, add it to `AGENT_TOOLS`.
3. **Add a branch** — add a node + one entry in the `add_conditional_edges` map.
4. **Add memory/human-in-the-loop** — compile with a `MemorySaver` checkpointer and
   `interrupt_before=["finalize"]` to require approval before sending.

## Talking points (for the architecture discussion)

- State-first design: nodes are pure `(state) -> partial update`, trivially testable.
- Structured output is the reliability lever — decisions are typed, not parsed.
- Determinism where it counts: routing and the safety override are plain code.
- The loop is bounded by construction; failure modes (bad id, timeout) degrade
  gracefully (tool returns an `error`, LLM client retries transient failures).
- Everything is swappable behind `llm.py` / `config.py` — provider, model, limits.
```
