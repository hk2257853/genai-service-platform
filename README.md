# GenAI Service Platform

A production-grade Python AI microservice platform built with FastAPI, LiteLLM, and PostgreSQL. Demonstrates layered clean architecture, async token streaming, structured Pydantic schema validation, resilient tenacity retries, and document RAG pipelines.

## Project Structure

```text
ai_service_platform/
├── main.py                         # App factory, router mounts, & exception handlers
├── config.py                       # Centralized application settings & env loader
│
├── core/                           # Shared Cross-Cutting Infrastructure
│   ├── client.py                   # Async LiteLLM wrapper + tenacity retries
│   ├── exceptions.py               # Custom domain exception hierarchy
│   └── db.py                       # Database connection pool & vector/array operations
│
├── schemas/                        # Pydantic Data Schemas
│   ├── generation.py               # Chat, Sentiment, Stream models
│   └── rag.py                      # Ingest, Ask, SourceMetadata models
│
├── services/                       # Business Domain Services
│   ├── generation.py               # LLMService (text gen, sentiment, streaming)
│   └── rag.py                      # RAGService (document ingestion & grounded Q&A)
│
├── api/                            # Feature-Based API Controllers / Routes
│   ├── health.py                   # GET /api/v1/health
│   ├── generation.py               # POST /api/v1/chat, /sentiment, /stream
│   └── rag.py                      # POST /api/v1/ingest, /ask
│
├── rag/                            # RAG Pipelines & Processing Tooling
│   ├── chunking.py                 # Fixed-size & semantic chunking strategies
│   └── embeddings.py               # Batch & single vector embeddings
│
└── tests/                          # Automated Integration Test Suite
    ├── fixtures/                   # Isolated test data & sample docs
    ├── test_api.py                 # Endpoint integration tests
    └── test_rag.py                 # Chunking comparison tests
```

---

## Quick Start

### 1. Install Dependencies
```powershell
py -m pip install -r requirements.txt
py -m pip install -e .
```

### 2. Configure Environment
Copy `.env.template` to `.env` in the project root:
```env
MODEL_ID=gemini/gemini-flash-latest
GEMINI_API_KEY=your_gemini_api_key_here

# Database Config (PostgreSQL)
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=ai_project
DB_USER=postgres
DB_PASSWORD=your_postgres_password_here
```

---

## Running the Application

Start the FastAPI server using Uvicorn:
```powershell
py -m uvicorn ai_service_platform.main:app --reload
```
Interactive Swagger API documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## Available API Endpoints (`/api/v1`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Health check endpoint |
| `POST` | `/api/v1/chat` | Generate text completion with token usage & cost tracking |
| `POST` | `/api/v1/sentiment` | Extract structured sentiment analysis using Pydantic schema |
| `POST` | `/api/v1/stream` | Stream token-by-token completion via Server-Sent Events (SSE) |
| `POST` | `/api/v1/ingest` | Chunk, embed, and store document in vector/array database |
| `POST` | `/api/v1/ask` | Retrieve grounded document context and execute Q&A |

---

## Running Automated Tests

Run the integration and RAG test suites:
```powershell
# Run API endpoint integration tests
py -m ai_service_platform.tests.test_api

# Run comparative chunking analysis (fixed-size vs semantic)
py -m ai_service_platform.tests.test_rag
```

---

## Zero-Code Provider Swapping

Swapping models or providers is zero-code. To switch from Gemini to OpenAI:
```env
MODEL_ID=openai/gpt-4o
OPENAI_API_KEY=sk-...
```
LiteLLM handles request formatting, token counting, and provider routing transparently under the hood.
