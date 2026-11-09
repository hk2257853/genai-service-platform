# Current Project State

## 1. Active Tech Stack
- **Frameworks:** FastAPI, LangGraph, FastMCP (Model Context Protocol), LlamaIndex
- **LLM / Embeddings:** Gemini / OpenAI client abstractions, dense embeddings + BM25 hybrid retrieval
- **Data & Runtime:** Python 3.11+, STDIO transport for MCP, Pydantic v2

## 2. Completed Milestones (Days 1–17)
- **Days 1–4 (RAG Fundamentals):** FastAPI service layer, BM25 + dense embedding hybrid search, cross-encoder reranking, query expansion.
- **Days 5–7 (Production RAG):** LlamaIndex retrieval pipeline integrated into FastAPI `/ask` endpoint with structured logging.
- **Days 8–10 (LangGraph Agents):** LangGraph state machine, conditional routing, tool calling with in-process tools (`check_order_status`, `get_product_info`), short-term conversation state.
- **Days 11–12 (MCP Server):** Standalone FastMCP server (`ecommerce_server.py`) exposing orders DB as tools over STDIO with stderr logging isolation.
- **Day 13 (LangGraph + MCP Integration):** Multi-turn agent using `MultiServerMCPClient` with `MemorySaver` checkpointer and error recovery.
- **Day 13.1 (Harness Engineering):** Context anchors (`AGENTS.md`, `PROJECT_STATE.md`), deterministic rules, modular skills, and verification loops.
- **Day 14 (Agent vs. RAG):** Architectural decision framework, 5-level autonomy spectrum, and the 6 production anti-patterns.
- **Day 15 (Evals):** Lightweight RAGAS (faithfulness, answer relevancy, contextual precision), deterministic trajectory assertions, and LLM-as-a-judge.
- **Day 16 (Observability & Tracing):** Hierarchical LangSmith DAG run-trees, real-time streaming TTFT telemetry, and client-side token + USD cost accounting.
- **Day 17 (Production AI Patterns):**
  - Semantic Caching with normalized NumPy vector cosine similarity ($\tau = 0.92$), system prompt namespace isolation, and LRU eviction.
  - Cross-Provider Fallback Routing via LiteLLM Router (Gemini -> GPT-4o-mini -> Claude Haiku) with circuit breaker cooldowns.
  - Token-Per-Minute (TPM) 60-second sliding-window limiter with two-phase reservation/settlement and HTTP 429 Retry-After calculation.
  - Verified with 16 observable unit/integration tests and `demo_day17_production_patterns.py`.

## 3. Next Goal: Day 18
- **Objective:** AI System Design: RAG at scale, Vector DB choices (pgvector vs Pinecone vs Weaviate), LLM gateway (routing + cost control), Failure scenarios (timeouts + rate limits + model down).

## 4. Known Gotchas
- **Windows Console Encoding (cp1252):** Avoid non-ASCII unicode glyphs (like checkmarks or arrows) in stdout; use ASCII `[PASS]` and `->`.
- **Heavy Libraries:** Keep evaluation and pattern modules lightweight with NumPy/LiteLLM rather than heavy C++ native extensions on Python 3.14.
- **Pydantic v2:** Use `model_validate()` and `model_dump()`.
- **MCP stdout/stderr:** In MCP servers, isolate all logs strictly to `stderr` to prevent JSON-RPC corruption on `stdout`.

## 5. Key Entry Points
- Semantic Cache: `ai_service_platform/core/semantic_cache.py`
- TPM Limiter: `ai_service_platform/core/tpm_limiter.py`
- Fallback Router: `ai_service_platform/core/fallback_router.py`
- LLM Gateway API: `ai_service_platform/api/gateway.py`
- Production Patterns Tests: `ai_service_platform/tests/test_production_patterns.py`
- Day 17 Demo: `demo_day17_production_patterns.py`
- Observability Demo: `demo_day16_observability.py`
- Learning Notes: `NOTES.md`



