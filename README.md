# GenAI Service Platform

A production-grade Python AI microservice platform built with **FastAPI**, **LiteLLM**, **LlamaIndex**, and **LangGraph**. Demonstrates a 5-layer clean architecture with unified schema inheritance, hybrid RAG retrieval, metadata filtering, and cyclic autonomous state machine agents.

---

## Architecture Overview

```text
ai_service_platform/
├── main.py                         # App factory, router mounts, & exception handlers
├── config.py                       # Centralized application settings & env loader
│
├── core/                           # Shared Cross-Cutting Infrastructure
│   ├── client.py                   # Async LiteLLM wrapper + tenacity retries
│   ├── utils.py                    # Usage metrics extraction & shared helpers
│   ├── exceptions.py               # Custom domain exception hierarchy
│   └── db.py                       # PostgreSQL + pgvector connection pool
│
├── schemas/                        # Pydantic Schemas (inheriting BaseLLMUsage)
│   ├── common.py                   # BaseLLMUsage (prompt/completion tokens, cost)
│   ├── generation.py               # Chat, Structured Sentiment, Streaming
│   ├── rag.py                      # Ingest, Production Ask, Search models
│   ├── llamaindex_rag.py           # LlamaIndex Ingest & Metadata-filtered Ask
│   ├── agent.py                    # ToolCallRecord, AgentRunRequest/Response
│   └── workflow.py                 # Multi-step Workflow & HITL Approval schemas
│
├── services/                       # Business Domain Orchestration Layer
│   ├── generation.py               # LLMService (text generation, sentiment, stream)
│   ├── rag.py                      # RAGService (Hybrid dense+BM25, RRF, Cross-Encoder)
│   ├── llamaindex_rag.py           # LlamaIndexRAGService (SentenceSplitter, filters)
│   ├── agent.py                    # AgentService (LangGraph orchestration & cost metrics)
│   └── workflow.py                 # WorkflowService (HITL pause/resume & state introspection)
│
├── agents/                         # Autonomous Agent State Machines
│   ├── langgraph_agent.py          # StateGraph, tools registry, error recovery, loop guards
│   └── workflow_graph.py           # MultiStepWorkflowAgent, conditional routing, checkpointer, HITL
│
├── rag/                            # Retrieval-Augmented Generation Tooling
│   ├── chunking.py                 # Fixed-size & semantic chunking
│   ├── embeddings.py               # Vector embeddings via LiteLLM
│   ├── bm25.py                     # BM25 sparse keyword index & RRF fusion
│   ├── reranker.py                 # Cross-Encoder neural reranking
│   ├── query_expansion.py          # LLM multi-query expansion
│   └── llamaindex_pipeline.py      # LlamaIndex VectorStoreIndex + async thread bridge
│
└── tests/                          # Automated Integration Test Suite
    ├── test_api.py                 # Core generation & structured output tests
    ├── test_rag.py                 # Chunking comparison tests
    ├── test_production_rag.py      # Hybrid RAG + Cross-Encoder reranking tests
    ├── test_llamaindex_rag.py      # LlamaIndex metadata filtering tests
    ├── test_langgraph_agent.py     # LangGraph multi-turn agent & tool tests
    └── test_langgraph_workflow.py  # Multi-step flow, conditional routing & HITL tests
```

---

## Quick Start

### 1. Install Dependencies
```powershell
py -m pip install -r requirements.txt
py -m pip install -e .
```

### 2. Configure Environment
Create `.env` in the project root:
```env
MODEL_ID=gemini/gemini-3.1-flash-lite
GEMINI_API_KEY=your_gemini_api_key_here

# PostgreSQL Config
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=ai_project
DB_USER=postgres
DB_PASSWORD=your_postgres_password_here
```

### 3. Run Application
```powershell
py -m uvicorn ai_service_platform.main:app --reload
```
* **Swagger Interactive Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Available API Endpoints (`/api/v1`)

| Domain | Method | Endpoint | Description |
|---|---|---|---|
| **Health** | `GET` | `/api/v1/health` | Service health status check |
| **Generation** | `POST` | `/api/v1/chat` | Text generation with token usage & cost tracking |
| **Generation** | `POST` | `/api/v1/sentiment` | Structured Pydantic sentiment analysis |
| **Generation** | `POST` | `/api/v1/stream` | Token streaming via Server-Sent Events (SSE) |
| **RAG** | `POST` | `/api/v1/rag/ingest` | Chunk, embed, and store documents in pgvector |
| **RAG** | `POST` | `/api/v1/rag/ask` | Production RAG (Hybrid BM25 + RRF + Cross-Encoder) |
| **LlamaIndex** | `POST` | `/api/v1/llamaindex/ingest` | LlamaIndex ingestion with key-value metadata |
| **LlamaIndex** | `POST` | `/api/v1/llamaindex/ask` | Grounded Q&A with exact-match metadata filtering |
| **Agents** | `POST` | `/api/v1/agent/run` | Autonomous LangGraph state machine with tool calling |
| **Workflows** | `POST` | `/api/v1/workflow/start` | Multi-step stateful flow with policy guardrails & HITL pause |
| **Workflows** | `POST` | `/api/v1/workflow/resume` | Resume interrupted thread with supervisor approval decision |
| **Workflows** | `GET` | `/api/v1/workflow/state/{id}` | Inspect StateSnapshot of active or interrupted thread |
| **Workflows** | `GET` | `/api/v1/workflow/topology` | Export static compiled graph nodes & edges as native Mermaid syntax |

---

## Automated Testing & HTML Visual Reports

Run the automated test suite and generate an interactive standalone HTML report:

```powershell
# Run tests and generate HTML report
py -m pytest ai_service_platform/tests --html=test_report.html --self-contained-html --capture=sys
```

* Open `test_report.html` in your browser to view:
  * Pass/fail status and execution durations across all tests
  * Expandable captured **input prompts, model responses, tool traces, and USD costs**
