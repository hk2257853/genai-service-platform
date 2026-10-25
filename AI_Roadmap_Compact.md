## AI Agent Instructions (Antigravity / Coding Agent)

- **FRAMEWORK-FIRST, ALWAYS.** If a maintained library/framework solves the problem (provider abstraction, retries, caching, agent orchestration, RAG indexing, rate limiting, etc.), use it. Never hand-roll a custom version, even for "learning by building." No exceptions — applies uniformly across every day, including agent loops (use LangGraph directly, don't hand-roll a raw ReAct loop first).
- **UNDERSTANDING STILL MATTERS, BUT LIVES IN NOTES, NOT CODE.** Before or while using a framework, briefly analyze and note what it's doing under the hood. This goes into NOTES.md as concise, interview-recall-style notes (core concepts, key tradeoffs, "why X over Y," gotchas). It does NOT translate into writing a parallel hand-rolled implementation.
- Clean code, generally:
  - No duplication — shared logic goes in one place (config, base class, util), not copy-pasted across files
  - No magic numbers or hardcoded values — named constants, env vars, or config
  - No hardcoded values that defeat an abstraction already in place (e.g. skipping a provider layer to call an SDK directly)
  - Single responsibility per function/class — no god functions doing setup + logic + I/O together
  - Consistent naming and structure across files/modules
  - Fail loudly, not silently — proper error handling, no swallowed exceptions
  - Comments explain *why*, not *what* — code should be self-explanatory for the *what*
- After each day's work, append concise personal notes (interview-recall style: core concepts, key tradeoffs, "why X over Y" decisions, gotchas hit) to a running NOTES.md — built incrementally per day, not written at the end.

# Backend SDE-2 + AI Integration Roadmap (Final)

## Week 1 — LLM + RAG Core
Day 1 — LLM APIs: OpenAI/Gemini/Anthropic, Structured outputs, Function calling, Streaming, Token + cost awareness
Day 2 — FastAPI service setup, Clean architecture (routes → service → client), Async + retries + error handling
Day 3 — RAG basics, Chunking (fixed vs semantic), Embeddings + pgvector, Simple doc Q&A
Day 4 — Production RAG, Hybrid retrieval (dense + BM25), Reranking (cross-encoder), Query expansion
Day 5 — LlamaIndex RAG pipeline (framework-first: use LlamaIndex for indexing, retrieval, metadata filtering instead of hand-rolling), Note in NOTES.md what LlamaIndex does under the hood concisely
Day 6 — Integrate LlamaIndex RAG into FastAPI, Clean API: /ask, Logging + basic metrics
Day 7 — Revise + simplify, Why hybrid > dense only, Chunking tradeoffs, Latency vs quality

*Week 1 takeaway:* Days 2–4 (By hand) taught the internals (BM25, vector math, reranking) for interview understanding; Days 5–6 (LlamaIndex) taught how production code is written using industry-standard libraries.

## Week 2 — Agents
Day 8 — Agent basics using LangGraph directly (framework-first: no hand-rolled ReAct loop), State machine mindset, Note in NOTES.md what a ReAct loop is and what LangGraph's state machine does under the hood
Day 9 — LangGraph multi-step flows, Conditional routing, Human-in-the-loop patterns
Day 10 — Add tools: check_order_status + get_product_info, Memory (short-term convo state)
Day 11 — MCP basics, Tools vs resources vs prompts, Why it matters (standardized tool access)
Day 12 — Build simple MCP server (FastMCP), Expose orders DB as tools
Day 13 — Connect LangGraph agent → MCP server, Multi-turn agent with real data
Day 13.1 — Agent Harness Engineering & Autonomous Operating Systems (Pillars, closed-loop verification, token economics)
Day 14 — Agent vs RAG (when to use which), When NOT to use agents

## New Fundamental Concepts (extracted from Weeks 3–4)
*These are genuinely new AI-specific concepts — not just applications of Week 1+2 skills. Learn these before building the production project so you're interview-ready on all core topics.*

Day 15 — Evals: RAGAS (faithfulness + answer relevancy + contextual precision), Simple assertions for agents, LLM-as-judge (use model to score output quality + compare prompt/retrieval changes)
TODO: Confirm day 16 and 17 plan with claude
Day 16 — Observability + Tracing: *(Skip generic HTTP request-ID & JSON logging — already built in Java backend and this is a learning project as I know the concept, I can easily implement it in future when required)* → Only implement AI-specific observability: LangSmith or hierarchical agent DAG run-tree tracing, Time-to-First-Token (TTFT) on streaming responses, and per-call Token usage + USD cost monitoring
Day 17 — Production AI Patterns: *(Skip generic Redis exact caching & IP-based RPM rate limiting — already built in Java backend)* → Only implement AI-specific patterns: Semantic caching (vector cosine similarity for intent reuse), Fallback cross-provider model routing (LiteLLM router: Gemini → Claude/GPT-4o mini), and Token-Per-Minute (TPM) / circuit breaker protections for LLM APIs
Day 18 — AI System Design: RAG at scale, Vector DB choices (pgvector vs Pinecone vs Weaviate), LLM gateway (routing + cost control), Failure scenarios (timeouts + rate limits + model down)

*After this point, all core concepts are covered. Everything below is applying Week 1–2 + above concepts + existing backend skills to build a production-grade portfolio project.*

## Application & Portfolio (build phase)
Day 19 — Feature 1: Production RAG over orders DB, Hybrid retrieval + reranking
Day 20 — Feature 2: Agent + MCP tools, Real workflows (not toy chat)
Day 21 — Async pipeline (use Kafka knowledge), Event → AI processing → DB write
Day 22 — End-to-end polish, Service should feel production-grade
Day 23 — Java add-on, LangChain4j + Spring Boot, Semantic product search (pgvector)
Day 24 — Load testing (Locust), Understand LLM latency bottlenecks
Day 25 — Polish project, Architecture diagram, Clean README, Tradeoffs doc
Day 26–30 — Mock interviews, Project end-to-end explanation, Scaling decisions, Failure handling

## Final Output
One production-style FastAPI AI service, RAG with hybrid retrieval + reranking, One LangGraph agent using MCP tools, One MCP server connected to real DB, Evals: RAGAS + assertions + LLM-as-judge, Observability: tracing + cost + latency, Java Spring Boot + LangChain4j demo, Strong system design clarity

## Reality Check
Build while learning — theory alone won't get you hired, One clean system > knowing 10 tools shallowly, Depth > breadth — this is where most people fail
