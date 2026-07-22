# AI Agent Engineering Exercise

Design and implement a simple conversational AI agent that can answer questions about Pokémon using the PokéAPI.

**API:** https://pokeapi.co/docs/v2

The focus is not on having a polished UI, but on demonstrating a solid understanding of agentic patterns, prompt design, tool usage, API integration, and thoughtful system design.

---

## High-Level Requirements

### 1. Conversational Agent

Build an agent that a user can chat with using natural language, such as through a simple console or web interface.

The agent should maintain conversational context across multiple turns.

For example, if a user asks "Tell me about Charizard" and then follows up with "What abilities can it have?" the agent should understand that "it" refers to Charizard.

### 2. Domain: Pokémon

The agent should be able to discuss Pokémon and answer factual questions based on live data from the PokéAPI.

The agent should ground its answers in API data rather than relying on hardcoded Pokémon knowledge.

Example queries:

- "What type is Pikachu?"
- "What abilities can Bulbasaur have?"
- "What are Charizard's base stats?"
- "Which Pokémon evolve from Eevee?"
- "What is Squirtle's evolution chain?"
- "Which Pokémon are weak to electric-type moves?"
- "What moves can Lucario learn?"
- "Compare the base stats of Gengar and Alakazam."
- "Which Pokémon have the ability intimidate?"
- "Tell me about Mewtwo, including its type, abilities, and base stats."
- "What is the effect of the move thunderbolt?"
- "Show me all fire-type Pokémon from the API."
- "What games does Snorlax appear in?"
- "What is the difference between a Pokémon's species data and its Pokémon data?"

### 3. Tool Design

The agent should use one or more tools, functions, APIs, or retrievers to fetch information from the PokéAPI.

**PokéAPI base URL:** https://pokeapi.co/api/v2/

The tools should handle:

- API request construction
- Fetching data from relevant PokéAPI endpoints
- Response parsing and data cleaning
- Joining related resources when needed
- Error handling and fallbacks
- Handling ambiguous or misspelled user input when reasonable

The agent should not simply expose raw API responses to the user. It should transform API data into clear, user-friendly answers.

Example tool calls below are illustrative only. Candidates are not required to use these exact tools:

- `get_pokemon(name_or_id: str)`
- `get_pokemon_species(name_or_id: str)`
- `get_evolution_chain(pokemon_name: str)`
- `get_type_matchups(type_name: str)`
- `compare_pokemon(pokemon_names: list[str])`
- `search_pokemon_by_ability(ability_name: str)`
- `get_move_details(move_name: str)`

Some questions may require multiple API calls. For example, answering "What is Squirtle's evolution chain?" may require fetching Squirtle's species data, following the evolution chain URL, and then parsing the returned chain into a readable format.

### 4. LLM Integration

The LLM should interpret user intent, decide when to call a tool, and determine how to use the tool results in its response.

Responses should be clear, concise, and grounded in data returned by the tools.

The agent should be able to explain what it found in natural language rather than just returning JSON.

Hardcoding answers or skipping tool usage will not meet the intent of the exercise.

### 5. Agentic Behavior

The agent should show some autonomous reasoning, such as:

- Making multiple tool calls to gather complete information
- Asking clarifying questions when user input is ambiguous
- Resolving follow-up questions using conversational context
- Choosing the right endpoint or combination of endpoints for the user's question
- Comparing, filtering, or summarizing data from multiple API responses
- Iterating on its own response to improve accuracy or completeness

For example, if a user asks "Which one is stronger, Dragonite or Salamence?" the agent should clarify what "stronger" means or choose a reasonable interpretation, such as comparing base stats, while making that assumption explicit.

### 6. Validation

Provide a brief overview or implementation of how you verified the agent's accuracy and reliability.

This may include:

- Unit tests for API wrapper functions
- Tests for parsing and formatting API responses
- Tests for multi-turn conversational context
- Example transcripts showing expected tool calls and responses
- Handling of failed API requests, missing Pokémon, or malformed user input
- Spot checks comparing agent responses against known PokéAPI responses

The validation does not need to be exhaustive, but it should demonstrate that the candidate thought carefully about correctness, reliability, and failure cases.

---

## Submission Expectations

Please include:

1. Instructions for running the agent locally
2. A short explanation of the architecture
3. A description of the tools or functions the agent can use
4. A brief explanation of how the LLM decides when to use tools
5. A few example conversations
6. A summary of validation or testing performed

The implementation can use any programming language, framework, or LLM provider. The UI can be simple. A command-line chat interface is acceptable.

We care more about the quality of the agent design than the visual polish of the application.
