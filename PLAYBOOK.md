# Game-Day Playbook

## Minute 0–5: set up the key, prove the pipe works
1. Paste their key into `.env` (`OPENAI_API_KEY=...`).
2. If they named a model, set `OPENAI_MODEL=...` in `.env`.
3. `python main.py "hello"` — confirms auth + model + parsing before you build anything.
   - Auth error? Wrong key. Model-not-found? Change `OPENAI_MODEL`.

## Minute 5–20: read the prompt, map it to the scaffold (don't code yet)
Answer these four questions on paper — they ARE the design:
- **Decision:** what does the agent classify / decide? → rewrite `schemas.py`.
- **Branches:** what are the distinct outcomes? → the `add_conditional_edges` map.
- **Tools:** what does it need to look up or do? → `@tool` funcs in `tools.py`.
- **Done:** when does it stop? → the loop guard / terminal edges.

The graph topology (classify → route → act ⇄ tools → finalize) usually survives.
You're mostly swapping schemas, prompts, and tools.

## Minute 20–90: build inside the existing structure
- New decision shape → `schemas.py`
- New branch → add a node in `nodes.py` + one entry in the edges map in `graph.py`
- New tool → `@tool` in `tools.py`, add to `AGENT_TOOLS`
- External API tool → `requests.get(..., timeout=10)`, `raise_for_status()`, return trimmed dict
- Run `python smoke_test.py` after structural changes (catches wiring breaks with no tokens)

## Minute 90–110: verify + polish
- `python main.py "<realistic input>"` for each branch (happy path, escalate, edge case)
- `uvicorn server:app --reload` → click through the UI once
- Skim the README "Talking points" — that's your architecture narrative

## Minute 110–120: write 5 lines in the README on WHY
Reviewers grade reasoning. State: why a graph (branch + loop), why structured output
(typed decisions), where determinism lives (routing), how it fails safe.

## If you get stuck / short on time
- A working narrow agent beats a broken ambitious one. Ship the happy path first,
  then add branches.
- Every LLM node can degrade: structured output validates; the client retries transient
  errors; tools return `{"error": ...}` instead of throwing.
- Don't reach for MCP, multi-agent, or fancy memory unless the prompt demands it.

## Common curveballs (all absorbed by the current design)
| They ask for... | You do... |
|---|---|
| RAG / "answer from these docs" | Add a `retrieve` tool that searches the docs; agent calls it. |
| Multi-step / "gather then act" | Already handled — agent⇄tools loop. |
| Human approval before an action | Compile with `MemorySaver` + `interrupt_before=["finalize"]`. |
| Structured JSON output | Already the pattern — add a Pydantic schema, `with_structured_output`. |
| "Call this external API" | `@tool` wrapping `requests` with timeout + error handling. |
| Classification / routing | That's the `classify` node + conditional edge, verbatim. |
