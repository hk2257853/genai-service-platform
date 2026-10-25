# Current Project State

## 1. Active Tech Stack
- **Frameworks:** FastAPI, LangGraph, FastMCP (Model Context Protocol), LlamaIndex
- **LLM / Embeddings:** Gemini / OpenAI client abstractions, dense embeddings + BM25 hybrid retrieval
- **Data & Runtime:** Python 3.11+, STDIO transport for MCP, Pydantic v2

## 2. Completed Milestones (Days 1–13)
- **Days 1–4 (RAG Fundamentals):** FastAPI service layer, BM25 + dense embedding hybrid search, cross-encoder reranking, query expansion.
- **Days 5–7 (Production RAG):** LlamaIndex retrieval pipeline integrated into FastAPI `/ask` endpoint with structured logging.
- **Days 8–10 (LangGraph Agents):** LangGraph state machine, conditional routing, tool calling with in-process tools (`check_order_status`, `get_product_info`), short-term conversation state.
- **Days 11–12 (MCP Server):** Built standalone FastMCP server at `ai_service_platform/mcp_servers/ecommerce_server.py`.
  - Exposes tools over STDIO: `get_order_details(order_id)`, `get_product_details(product_id_or_name)`.
  - Logging isolated strictly to `stderr` to preserve clean JSON-RPC communication on `stdout`.
- **Day 13 (LangGraph + MCP Integration):**
  - Built `MCPLangGraphAgent` in `ai_service_platform/agents/mcp_agent.py` using `langchain-mcp-adapters` and `MultiServerMCPClient`.
  - Dynamically discovers tools via `tools/list` without hardcoded schemas.
  - Multi-turn conversation state persistence with `MemorySaver` checkpointer.
  - Demonstrated coreference resolution and structured error recovery over real mock data (`scripts/demo_mcp_agent.py` and `ai_service_platform/tests/test_mcp_agent.py`).
- **Day 13.1 (Agent Harness Engineering & Autonomous Operating Systems):**
  - Established the 3 core pillars of a production harness: Context Anchors (`AGENTS.md` & `PROJECT_STATE.md`), Deterministic Rules (`rules/`), and Modular On-Demand Skills (`skills/*/SKILL.md`).
  - Implemented closed-loop verification protocols: task completion requires executed proof (`0` exit code) rather than open-loop text generation.
  - Codified token economics and lazy skill loading patterns in `NOTES.md`.
- **Day 14 (Agent vs. RAG, Anti-Patterns):**
  - Formulated rigorous Agent vs. RAG architectural comparison matrix and the 5-level AI autonomy spectrum.
  - Codified the 6 production anti-patterns where agents should NOT be used (deterministic logic, strict SLAs, cost-sensitive endpoints, unbounded mutations, exact calculations, simple search).
  - Codified the core architectural tradeoffs and decision rules in `NOTES.md`.
- **Day 15 (Evals: RAGAS, Deterministic Assertions, LLM-as-a-Judge):**
  - Built a 3-tier evaluation framework in `ai_service_platform/evals/`.
  - Implemented lightweight RAGAS metrics: Faithfulness (claim extraction & entailment), Answer Relevancy (reverse synthetic query cosine similarity), and Contextual Precision (rank-weighted precision/MAP).
  - Implemented deterministic agent trajectory assertions in `agent_assertions.py`: tool selection, sequence order, step & token budget limits, error recovery, and answer integrity without LLM calls.
  - Implemented Pointwise (rubric 1-5 with reasoning) and Pairwise LLM-as-a-Judge with position-swap bias mitigation.
  - Validated with 6 observable tests in `ai_service_platform/tests/test_evals.py` (all passed, 0 exit code).

## 3. Next Goal: Day 16
- **Objective:** TBD / Next milestone in roadmap (e.g. Production Deployment, Monitoring & Observability with OpenTelemetry/Langfuse, or Caching & Rate Limiting).

## 4. Known Gotchas
- **Windows Console Encoding (cp1252):** Avoid emitting non-ASCII unicode glyphs (like `✓` or `→`) in test output on Windows consoles; use ASCII `[PASS]` and `->`.
- **Heavy Eval Libraries (ragas):** `ragas` requires C++ build tools for `scikit-network` on Python 3.14. Using lightweight, transparent custom evaluators built directly on the platform LLM client avoids dependency bloat and executes in ~40s.
- **MCP stdout/stderr:** All logging in MCP servers MUST go to `stderr`. Any `print()` or stdout logging will corrupt the JSON-RPC stream.
- **MCP server lifecycle:** The FastMCP server must be running (or launched by `MultiServerMCPClient`) before agent tests will pass.
- **LlamaIndex index:** If mock data in `ai_service_platform/data/` changes, the index must be rebuilt.
- **Pydantic v2:** Use `model_validate()` not `parse_obj()`. Use `model_dump()` not `.dict()`.

## 5. Key Entry Points
- Evals Suite: `ai_service_platform/evals/evaluators.py`
- Agent Assertions: `ai_service_platform/evals/agent_assertions.py`
- Eval Tests: `ai_service_platform/tests/test_evals.py`
- MCP Server: `ai_service_platform/mcp_servers/ecommerce_server.py`
- MCP Agent: `ai_service_platform/agents/mcp_agent.py`
- Multi-Turn Demo CLI: `scripts/demo_mcp_agent.py`
- Integration Tests: `ai_service_platform/tests/test_mcp_agent.py`
- Learning Notes: `NOTES.md`


