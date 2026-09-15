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

## 3. Next Goal: Day 14
- **Objective:** Revise full flow, Agent vs RAG (when to use which), When NOT to use agents.

## 4. Known Gotchas
- **MCP stdout/stderr:** All logging in MCP servers MUST go to `stderr`. Any `print()` or stdout logging will corrupt the JSON-RPC stream.
- **MCP server lifecycle:** The FastMCP server must be running (or launched by `MultiServerMCPClient`) before agent tests will pass.
- **LlamaIndex index:** If mock data in `ai_service_platform/data/` changes, the index must be rebuilt.
- **Pydantic v2:** Use `model_validate()` not `parse_obj()`. Use `model_dump()` not `.dict()`.

## 5. Key Entry Points
- MCP Server: `ai_service_platform/mcp_servers/ecommerce_server.py`
- MCP Agent: `ai_service_platform/agents/mcp_agent.py`
- Multi-Turn Demo CLI: `scripts/demo_mcp_agent.py`
- Integration Tests: `ai_service_platform/tests/test_mcp_agent.py`
- Learning Notes: `NOTES.md`

