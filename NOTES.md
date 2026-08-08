# Day 1 — LLM Integration Fundamentals

### 1. Streaming (How it works under the hood)
*   **Concept:** Uses **Server-Sent Events (SSE)** over a standard HTTP connection (using `Transfer-Encoding: chunked`), not WebSockets.
*   **Behavior:** The server keeps the TCP socket open and pushes data chunk-by-chunk (using standard formatted lines starting with `data: `) as the model generates tokens in real-time.

### 2. Structured Output (How it guarantees the schema)
*   **Concept:** Uses **constrained decoding** at the provider gateway. The model is mathematically blocked from choosing tokens that would violate the defined JSON schema.
*   **Pydantic Role:** The API enforces the structure, while local Pydantic validation (`model_validate_json`) converts the raw JSON string into a type-safe Python object for IDE auto-complete and fail-safe recovery (e.g. if the connection is cut off).

### 3. Function Calling (The execution loop)
*   **Concept:** An agent loop where the client provides function schemas, the model returns a structured request to call a function (`tool_calls`), the client executes it locally, and the client sends the result back to the model.
*   **Why Registry:** A dictionary dispatch registry (`TOOL_MAP`) is preferred over nested `if-else` blocks to decouple registration from execution logic.

### 4. Cost & Token Tracking
*   **Pre-flight:** Count tokens locally/offline using a tokenizer (e.g., `tiktoken` matching the model's vocabulary) to avoid context limit overflow.
*   **Post-call:** Extract exact token counts from the response metadata (`response.usage`). Cost is calculated **locally** by multiplying these counts against the model's pricing rates.

### 5. Resiliency (Gotchas & Production Setup)
*   **Transient Failures:** APIs throw transient errors (`503 Service Unavailable`, `429 Rate Limit`). Always use native SDK parameters like `num_retries` or library-level decorators (like `tenacity`) for automatic backoff.
*   **Model Routing:** Swapping models via LiteLLM is easy, but watch prefixes (e.g., use `gemini/` prefix to force routing through AI Studio instead of Vertex AI).

# Day 2 — FastAPI Service Setup & Clean Architecture

TDLR: It's basically the Day 1 features hosted as a Fast API server (Ai Platform).

### 1. Layered clean architecture roles
*   **Client Layer (`client.py`):** Acts as the boundary driver for upstream API integrations. Wraps LiteLLM `acompletion`, encapsulates retries, and translates vendor exceptions into custom domain exceptions to keep downstream business logic vendor-agnostic.
*   **Service Layer (`service.py`):** Holds pure business domain orchestration (e.g. structured data mapping/validation, token consumption, dynamic costing). Is completely decoupled from HTTP concerns.
*   **Route/Controller Layer (`routes.py`):** Handlers that receive inputs, run Pydantic validations, pass requests to the Service, and form HTTP responses. Decoupled from LLM client library details.

### 2. Async exceptions and global mapping
*   **Centralized error handling:** Avoid repeating try-except blocks inside FastAPI endpoint routes. Instead, register global FastAPI exception handlers (`app.exception_handler`) to automatically capture custom domain-level exceptions and map them to standard HTTP status codes (`401` for AuthError, `429` for RateLimitError, `503` for ConnectionError, etc.).

### 3. Async generator gotcha
*   **Concept:** Async functions containing `yield` statements return an **async generator object** immediately upon invocation rather than returning a coroutine.
*   **Gotcha:** Trying to do `await client.acompletion_stream(...)` raises `TypeError: 'async_generator' object can't be awaited`. Instead, call it synchronously to get the generator, then consume it with `async for`.

### 4. Tenacity integration for async retries
*   **Rule:** Use tenacity's `AsyncRetrying` wrapper dynamically inside client code, filtering retries exclusively to transient network/rate limit issues, while letting fatal configuration errors (like Auth failures) fail fast.

# Day 3 — RAG Basics: Chunking, Embeddings, pgvector, and Document Q&A

### 1. Chunking Strategies: Fixed-Size vs. Semantic
*   **Fixed-Size Chunking:** Splits text by character count with overlap.
    *   *Design:* Implemented word-boundary backtracking (reversing up to 20% of chunk size to find a space) to prevent clipping words.
    *   *Tradeoff:* Fast, deterministic, and requires no API calls. However, it completely ignores semantic boundaries, which can separate context across chunks.
*   **Semantic Chunking:** Groups sentences based on topic coherence.
    *   *Design:* Breaks document into sentences, generates embeddings, calculates the cosine distance ($1 - \text{cosine\_similarity}$) between adjacent sentences, and triggers a new chunk at distance spikes (threshold ~`0.25`).
    *   *Tradeoff:* Preserves complete thoughts and semantic unity. However, it is slower and incurs higher cost due to embedding calls for every single sentence.
    *   *Optimizations:* All sentence embeddings are fetched in a single batched API call to minimize network roundtrips.

### 2. pgvector & asyncpg Setup
*   **Concept:** Extends PostgreSQL to store and query vector embeddings directly.
*   **Driver Integration:** Registered pgvector with the `asyncpg` connection pool (`register_vector`) to automatically handle encoding/decoding of native Python float lists to Postgres `vector` types.
*   **Index Selection:** Created an **HNSW (Hierarchical Navigable Small World)** index using `vector_cosine_ops` to enable fast approximate nearest neighbors (ANN) search matching the `<=>` cosine distance operator.
*   **Gotcha:** Dynamic schema creation (e.g. vector dimension) cannot be parameterized dynamically via SQL placeholder arguments. The column type `vector({dimension})` must be formatted securely or resolved statically beforehand.

### 3. RAG Pipeline Orchestration
*   **Flow:** Retrieve context first, then synthesize. The query is embedded, the closest matching text chunks are pulled from pgvector, and a system prompt constructs the grounding context for the LLM.
*   **Hallucination Control:** Prompt instruction strictly forces the LLM to state "I do not know" if the answer is not present in the retrieved context.
*   **Cost Efficiency:** Both chunks and queries are embedded asynchronously, using batching where possible. Cost is tracked using `litellm.completion_cost` to maintain budget awareness.

# Day 4 — Production RAG: Hybrid Retrieval, Reranking, Query Expansion

Everything below is 6 concepts + how they're wired together.

## 1. Embeddings / Dense Retrieval
Convert text → vector. Similar meaning → vectors close together (cosine sim / ANN lookup).
- **Good:** understands meaning, catches paraphrases ("laptop broken" → "hardware failure")
- **Bad:** exact identifiers — `PROD-101`, `ERR_4001` — get lost in the embedding space

## 2. BM25 / Sparse Retrieval
Classic TF-IDF keyword search — does the document literally contain the query terms.
- **Good:** exact IDs, error codes, code tokens, rare exact words. Cheap — no model inference.
- **Bad:** no notion of meaning — synonyms/paraphrases miss entirely.

## 3. Hybrid Retrieval
Run dense + BM25 in parallel, combine results. Dense catches meaning, BM25 catches exact keywords — together they cover both failure modes.

**Core principle to say out loud in an interview:** always filter deterministically (cheap algorithms/APIs) before spending LLM tokens/reasoning on the data. Same logic applies outside RAG too — e.g. filter raw HTML/job listings with scripts before ever handing them to an LLM.

## 4. Query Expansion
LLM generates N paraphrases/sub-queries of the user's question before retrieval runs (e.g. "reset password" → "forgot password", "change password", "recover account"). Search with all of them.
- **Purpose:** recall — don't miss a good document just because it used different wording.
- **Cost:** N× retrieval calls + N× embedding calls.

## 5. RRF — Reciprocal Rank Fusion
You now have two ranked lists (dense's ranking, BM25's ranking) with scores that aren't comparable to each other. RRF merges them by **rank**, not raw score.

```
RRF_score(d) = Σ 1 / (k + rank_m(d))   over each retrieval method m
```
- k=60 is a smoothing constant — stops rank-1 from dominating everything.
- A document ranked well by *both* methods gets a higher combined score than one only one method liked.
- One-liner: **merge first** — rewards documents multiple retrieval methods agree on.

## 6. Cross-Encoder Reranking (The Careful Final Judge)
Why two stages?
- **Bi-Encoder (Embeddings / Dense Search - Stage 1):**
  - Query and Document are embedded **independently** into separate vectors (`Vec(Q)` and `Vec(D)`).
  - Similarity is calculated via Cosine Distance.
  - *Why Stage 1:* $O(1)$ lightning fast ANN vector index search across 10,000+ docs, BUT query words and document words **never look at each other directly** (no cross-token self-attention).
- **Cross-Encoder (`ms-marco-MiniLM-L-6-v2` - Stage 2):**
  - Feeds `[Query] + [Document Chunk]` **together** into a single Transformer model.
  - Every word in the query directly attends to every word in the document (full cross-token self-attention).
  - Outputs a high-precision relevance score (e.g., `+6.84` for high relevance vs `-11.32` for irrelevant).
  - *Why Stage 2:* Too computationally heavy to run forward passes on 10,000 docs ($O(N)$ inference cost), but ideal for carefully scoring a small **shortlist of 10--20 candidate chunks**.

**Two-Stage Architecture Funnel:**
```text
Corpus (10,000 docs) 
        ↓  [Stage 1: Fast Hybrid Search (Dense + BM25)]
Candidates (~15-20 shortlist)
        ↓  [Stage 2: Cross-Encoder Reranking (Full Cross-Token Attention)]
Top-K Chunks (~3-5 best) → Sent to LLM Prompt
```
One-liner: **Bi-encoder pulls the fast shortlist; Cross-encoder is the careful final judge that sorts the shortlist.**

## Recall vs Precision
- **Recall** — did we find all the relevant documents? → improved by Query Expansion + Hybrid Retrieval
- **Precision** — are the top results actually the best ones? → improved by Cross-Encoder Reranking

## Final Pipeline
```
User Question
    ↓
Query Expansion (N variants)
    ↓
Dense Search  +  BM25 Search   (hybrid, run on each variant)
    ↓
RRF Merge
    ↓
Cross-Encoder Reranking
    ↓
Top ~5 Chunks
    ↓
LLM → Answer
```

## Hiring analogy (for recall in an interview)
1. Dense + BM25 = HR quickly shortlists 50 resumes.
2. RRF = merges HR's shortlist with the hiring manager's shortlist into one list.
3. Cross-Encoder = the interviewer carefully reviews those 50 and picks the best 5.
4. LLM = answers using only those best 5.

**Find candidates → merge candidates → carefully rank candidates → answer.**
# Day 5 — Framework-First RAG with LlamaIndex: Under-The-Hood Analysis

### 1. Step-by-Step LlamaIndex Pipeline Mechanics
*   **Step 1 — Document:** Raw file comes in, wrapped with metadata (`category`, `version`, etc.).
*   **Step 2 — SentenceSplitter:** Breaks the Document into Nodes (chunks), cutting on sentence boundaries, with overlap so context isn't lost at the edges $\rightarrow$ this is what makes retrieval possible at all (you can't embed a whole doc usefully).
*   **Step 3 — StorageContext:** Decides where these Nodes and their embeddings will actually live (Postgres via `PGVectorStore`) — decoupled so storage is swappable without touching business logic.
*   **Step 4 — VectorStoreIndex:** Takes the Nodes, generates embeddings for each, hands them to the `StorageContext` to persist $\rightarrow$ this is the "build" step, orchestrates steps 2 and 3 together.
*   **Step 5 — PGVectorStore:** The actual place the vectors sit in Postgres — knows how to write/read Node embeddings as rows.
*   **Step 6 — Retriever $\rightarrow$ Query Engine:** At query time, Retriever does similarity search to pull top-$k$ matching Nodes; Query Engine takes those Nodes and calls the LLM to synthesize an actual answer — retrieval and generation are separate concerns.
*   **Step 7 — MetadataFilters:** Applied *before* step 6's similarity search, so you're only searching within the relevant subset of Nodes (e.g. only `category: security` docs), not the whole index.
*   **Step 8 — Settings (Global Config):** The thread running through all of this — tells steps 2, 4, and 6 which embedding model and LLM to actually use.

**Summary Pipeline:**
$$\text{Document} \longrightarrow \text{Chunked into Nodes} \longrightarrow \text{Embedded \& Stored} \longrightarrow \text{Filtered (if needed)} \longrightarrow \text{Retrieved by Similarity} \longrightarrow \text{Synthesized into Answer}$$

---

### 2. Multi-Threading Pattern in Async Framework Adapters (`ThreadPoolExecutor`)
*   **The Problem:** Third-party synchronous framework callbacks (like LlamaIndex's sync `_get_text_embedding`) are sometimes invoked inside async event loops (e.g., FastAPI / uvicorn ASGI thread). Calling `asyncio.run()` directly inside an already running event loop triggers `RuntimeError: asyncio.run() cannot be called from a running event loop`.
*   **The Solution (Worker Thread Isolation):** We offloaded synchronous execution of the async embedding coroutine to a dedicated background thread (`ThreadPoolExecutor(max_workers=1)`):
    ```python
    def sync_thread_run(coro_fn, *args):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(coro_fn(*args)))
            return future.result()
    ```
*   **Why Thread Isolation over `nest_asyncio`?** Monkey-patching libraries like `nest_asyncio` modify loop internals globally, which corrupts async transports (`httpx.ASGITransport` / `sniffio`). Running the coroutine in a separate worker thread creates an isolated event loop in that thread without corrupting the main ASGI event loop.
*   **When to use this pattern:**
    1. **Synchronous-to-Async Bridging:** When overriding legacy or 3rd-party synchronous interface methods that must invoke async I/O functions.
    2. **Offloading CPU-Bound Work:** When running intensive tasks (tokenization, encryption, image transformation) so they don't block FastAPI's single-threaded async event loop.

---

### 3. Sync vs. Async vs. Threading Mental Model (Universal Backend Rules)

#### A. Core Mental Model
*   **Async (`asyncio`):** A **single waiter** juggling 100 tables. While one table's food is in the oven (network / DB / LLM wait), the waiter takes orders from 10 other tables on **1 single thread**.
*   **Sync:** The waiter stands still in front of the oven and freezes until the food is done before serving anyone else (blocks the entire server).
*   **Threads (`ThreadPoolExecutor`):** Hiring an **extra helper** in a separate thread lane to do a specific job without blocking the main waiter.

#### B. The 3 Scenarios You Will Encounter

*   **Scenario 1 — In `async` code, need to run slow/blocking sync work:**
    *   *Examples:* Password hashing (bcrypt), slow disk file I/O, heavy CPU math.
    *   *Danger:* Running heavy sync code directly in `async def` freezes the server for every other user!
    *   *Solution:* `result = await asyncio.to_thread(slow_sync_function, arg1)`
*   **Scenario 2 — In `sync` code, need to call an `async def` function:**
    *   *Standard Script / CLI:* `result = asyncio.run(my_async_function())`
    *   *Inside a Web Server (FastAPI / ASGI loop):* Calling `asyncio.run()` crashes with `RuntimeError: asyncio.run() cannot be called from a running event loop`.
    *   *Solution (Thread Bridge):* Spawn an isolated worker thread that runs `asyncio.run()` in its own event loop and returns the result.
*   **Scenario 3 — Workload Selection Matrix:**

| Workload Type | Real-World Examples | Best Tool | Why |
|---|---|---|---|
| **I/O Bound** | LLM API calls, HTTP requests, Postgres queries | **Async (`asyncio`)** | Handles 10,000+ parallel requests on 1 thread with minimal RAM. |
| **I/O Sync-to-Async Bridge** | Calling async APIs from rigid third-party sync hooks | **Worker Threads (`ThreadPoolExecutor`)** | Creates an isolated space to run async coroutines safely without crashing. |
| **CPU Heavy** | Image resizing, massive array computation, ML model training | **Processes (`ProcessPoolExecutor`)** | Bypasses Python's GIL to utilize multiple physical CPU cores. |

#### C. Golden Rules for Production
> 1. **Async** is for waiting on networks and databases.  
> 2. **Threads** are for bridging sync/async boundaries and non-blocking I/O tasks.  
> 3. **Never call `asyncio.run()` directly inside an active event loop** — always isolate it in a worker thread.

