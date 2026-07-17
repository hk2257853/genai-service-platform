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
