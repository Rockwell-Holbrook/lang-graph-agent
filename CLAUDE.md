# CLAUDE.md — working agreement for this repo

Guidance for any Claude Code session working in this repository. Read it fully
before making changes, and follow it over your defaults.

## Context

This is a **LangGraph agent** project. **Architecture and engineering judgment
matter more than UI** — prioritize accordingly. Do not trade correctness or design
quality for speed. Hold a **high bar on Python idiom and LangGraph patterns**; call
out and fix anything sub-standard rather than matching a lower bar.

## Prime directive

**Favor the correct architectural decision over speed or convenience — always.**

Balance it with two guardrails, because good architecture is *right-sized*, not
maximal:
1. A working, narrow solution beats a broken, ambitious one. Ship the happy path,
   then extend. Correctness first, breadth second.
2. Do **not** over-engineer. No speculative abstraction. Reach for a heavier
   pattern (MCP, multi-agent, persistent memory, custom infra) **only** when the
   task actually requires it — never "in case." Knowing when *not* to is the skill.

When a shortcut is tempting, stop and state the tradeoff explicitly, then take the
principled path.

## Architectural principles (the design this repo commits to)

- **Typed state is the single source of truth.** One state object flows through the
  graph; nodes are pure `(state) -> partial update`.
- **Structured output for every decision.** Use Pydantic + `.with_structured_output()`.
  Never parse free text to make a control-flow decision.
- **Determinism lives in code.** Routing, guards, and safety overrides are plain
  Python. LLMs provide judgment; they do not decide control flow implicitly.
- **Every loop is bounded.** Cyclic paths (agent⇄tools) must have a hard iteration
  cap and a guaranteed termination edge.
- **Tools are small, typed functions.** The docstring is the LLM-facing spec —
  describe when to use the tool and what it returns.
- **Decouple transport from the agent.** The graph knows nothing about HTTP or the
  UI; the server knows nothing about graph internals. Communicate over a clean API.
- **Fail safe.** Tools return `{"error": ...}` instead of throwing; the LLM client
  retries transient errors; errors surface to the caller — nothing hangs silently.
- **Config and secrets from environment only.** Nothing hard-coded. Model, keys,
  and limits come from `.env` via `src/config.py`.

## Python standards (hold the line)

- `from __future__ import annotations` at the top of every module.
- **Type-hint every function** — parameters and return type. No untyped signatures.
- Use **Pydantic models** for data shapes and **frozen dataclasses** for config.
- Prefer **pure functions**; avoid hidden mutable state.
- **No bare `except:`** and never silently swallow exceptions — catch specific types
  and surface or log them.
- No mutable default arguments. Use `pathlib` over `os.path`. Use f-strings.
- Explicit imports only (no `import *`). Keep imports ordered/grouped.
- Small, single-purpose functions with docstrings.
- **Match the surrounding code's style and patterns** — don't introduce a new idiom
  where an existing one fits.

## LangGraph / LangChain standards

- **Layering:** LangGraph is the orchestration backbone (state, nodes, edges,
  cycles, streaming, checkpointing). LangChain supplies components (models, `@tool`,
  parsers) called *inside* nodes. LangGraph calls LangChain, not the reverse.
- **Use `StateGraph` directly**, not the prebuilt `create_agent` — we want control
  flow that is explicit and inspectable, so it can be explained and reviewed.
- **Graph vs. chain:** use LangGraph the moment you need cycles, branching, shared
  state, or human-in-the-loop. A plain chain is only for linear, single-pass flows.
- Use the `add_messages` reducer for the message channel.
- Agent⇄tools loops use `ToolNode` + an explicit loop-guard edge with an iteration cap.
- `.with_structured_output(Model)` for decisions; `.bind_tools()` for actions.
- Human-in-the-loop = compile with `MemorySaver` + `interrupt_before=[...]`.
- Streaming: drive with `stream_mode=["messages","values"]`; the transport layer
  translates to SSE. Keep the model behind `src/llm.py` so the provider is swappable.
- Stack is on the **1.x line** (langchain 1.x, langgraph 1.x). Verify APIs against
  installed versions, not memory.

## Decision heuristics

- **Tools in-process by default.** Promote a tool to an MCP server only when there is
  a *real* shared boundary: multiple consuming agents, a different owning team, or
  credential isolation. Not for a single-owner, single-consumer tool.
- Refactoring a working in-process tool into MCP later is cheap; premature
  distribution is expensive. Default to the function; escalate on evidence.

## Environment gotchas

- **Python 3.14:** define `TypedDict` graph-state schemas at **module level**, never
  inside a function — `get_type_hints` fails to resolve forward refs (e.g.
  `Annotated`, `add_messages`) in local scope.
- The model name is `OPENAI_MODEL` in `.env`. If the provided API key is scoped to a
  specific model, change it there — do not hard-code it.
- **Never commit `.env` or `.venv/`** (both gitignored). Never put secrets in code.

## Definition of done (do not skip)

1. Before coding a new task: read the prompt and answer four questions — *what does
   it decide? what are the branches? what tools does it need? when does it stop?*
   (See `PLAYBOOK.md`.) Those answers are the design.
2. After structural changes: run `python smoke_test.py` (wiring, no API key).
3. After streaming/transport changes: run `python test_streaming.py`.
4. Add a test when you add logic. **Never claim something works without running it.**
5. Verify each branch with a realistic input before reporting done.
6. Commit in small, logical units with clear messages.

## Commands

```bash
# venv interpreter (Windows)
.venv/Scripts/python.exe

python smoke_test.py        # structural checks, no API key
python test_streaming.py    # SSE translation + LangGraph contract, no API key
python main.py "..."        # CLI run (needs OPENAI_API_KEY in .env)
uvicorn server:app --reload # chat UI at http://localhost:8000
```

## Repo map

- `src/state.py` — typed `AgentState` (single source of truth)
- `src/schemas.py` — Pydantic decision shapes (structured output)
- `src/config.py` — env-driven settings (frozen dataclass)
- `src/llm.py` — model factory (swappable provider)
- `src/tools.py` — `@tool` functions (typed; docstring = spec)
- `src/nodes.py` — node functions + the deterministic router
- `src/graph.py` — graph assembly, edges, loop guard
- `server.py` — FastAPI + SSE transport (decoupled from the agent)
- `web/index.html` — dependency-free chat UI
- `main.py` — CLI entry; `smoke_test.py`, `test_streaming.py` — offline tests
- `PLAYBOOK.md` — step-by-step approach for a new task under time pressure
