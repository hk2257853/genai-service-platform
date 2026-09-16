---
name: git-commit
description: Guidelines and approval workflow for crafting high-quality Git commit messages matching this repository's established conventions. Use when preparing, reviewing, presenting commit messages for user approval, or generating git commit messages.
---

# Git Commit Message Conventions & Approval Workflow

This skill captures the established commit message conventions and approval requirements used across this repository. Follow these patterns when staging, preparing, reviewing, and committing work.

---

## Mandatory Human Approval Workflow (Zero Autonomous Commits)
- **NEVER execute `git commit` autonomously.**
- Even when explicitly asked to "commit the code" or wrap up:
  1. Inspect status and diff (`git status`, `git diff`).
  2. Draft the proposed commit message (type, scope, subject, bullet points) following the conventions below.
  3. **Present the write-up / proposed commit message to the user in chat.**
  4. Explicitly ask the user for confirmation (e.g. *"Here is the proposed commit message. May I proceed with the commit?"*).
  5. **Wait for user confirmation.** Only if the user explicitly approves (says "yes", "proceed", etc.) do you execute the `git commit` command.

---

## 1. Commit Message Structure

### Single-Line (Default — use for most commits)
```text
<type>(<scope>): <high-information subject line, can be long>
```

### Multi-Line (Only for large refactors or multi-component changes)
```text
<type>(<scope>): <high-information subject line>

- <Action verb> <concrete change in module/component A>
- <Action verb> <concrete change in module/component B>
- <Action verb> <impact on tests, benchmarks, or documentation>
```

**When to use multi-line:** Only when a commit touches 3+ files across different concerns (e.g. restructuring packages, reorganizing layers, or landing a multi-day milestone). Single-feature or single-component commits should be single-line.

---

## 2. Commit Types & Scopes

### Standard Types:
- `feat`: New capability, agent component, tool, or pipeline.
- `fix`: Bug fix, error handling patch, or pipe corruption fix.
- `refactor`: Structural reorganization without changing external behavior.
- `docs`: Documentation updates, roadmap updates, or `NOTES.md` additions.
- `test`: Test suite additions, fixtures, or verification scripts.

### Established Scopes in this Repo:
- `(agent - day X)` / `(agents)`: Agent logic, state machines, checkpointers.
- `(mcp)` / `(mcp - day X)`: FastMCP servers, STDIO transport, tool schemas.
- `(rag)` / `(rag - day X)`: Hybrid retrieval, BM25, LlamaIndex, rerankers.
- `(workflow - day X)`: Multi-step routing, human-in-the-loop flows.
- `(notes)`: Interview-recall takeaways and architecture notes in `NOTES.md`.

**Day numbering:** Include `- day X` in the scope when the commit corresponds to a specific learning milestone day. Omit it for general improvements or maintenance.

---

## 3. Style & Tone Rules

1. **High Information Density:**
   - **Bad:** `feat: update agent`
   - **Good:** `feat(agent - day 10): implement domain tools (order status, product info) and short-term conversational memory`
2. **Subject Lines Can Be Long:** Do NOT truncate to 72 characters. Prioritize completeness over brevity — a 150-character subject that's specific is better than a 50-character subject that's vague.
3. **Explicit Component Naming:** Mention specific libraries, protocols, or modules involved (e.g. `LangGraph`, `FastMCP`, `MemorySaver`, `BM25`, `JSON-RPC`).
4. **Imperative Action Verbs in Body:**
   - Use `- Add ...`, `- Implement ...`, `- Build ...`, `- Integrate ...`, `- Document ...`, `- Remove ...`.
5. **Link Code to Purpose:** When the subject line has room, state *what* was built and *why* (e.g., `...allowing us to retain context across turns and resolve follow-up queries seamlessly`).

---

## 4. Examples from Repository History

### Single-Line Feature (Focused Component)
```git
feat(agent - day 10): implement domain tools (order status, product info) and short-term conversational memory, allowing us to retain context across turns and resolve follow-up queries seamlessly
```

### Single-Line Feature (Cross-Cutting)
```git
feat(rag): build production doc Q&A system with hybrid BM25 retrieval, RRF, query expansion & cross-encoder reranking
```

### Single-Line Docs
```git
docs(notes): add Day 11 (MCP fundamentals) and Day 12 (FastMCP database tools) notes
```

### Multi-Line Refactor (Restructuring — many files, many concerns)
```git
refactor: restructure ai_service_platform into layered backend architecture

- Reorganize package into 5 distinct layers: core, schemas, services, api, and rag
- Separate Pydantic payload models into schemas/ and domain services into services/
- Split monolithic router into domain controllers (health, generation, rag) under api/
- Move test mocks and documents into tests/fixtures/ with location-agnostic path resolution
- Update README.md, pyproject.toml, and .env.template to reflect production structure
- Remove obsolete flat files, legacy examples directory, and temporary comments
```

### Multi-Line Feature (Multi-Component Setup)
```git
feat(day03): implement RAG basics with chunking, pgvector, and doc Q&A

- Add fixed-size and semantic chunking strategies in chunking.py
- Set up pgvector DB schema and HNSW indexes with asyncpg in db.py
- Implement AsyncLLMClient RAG service and API routes in service.py and routes.py
- Document implementation concepts and tradeoffs in NOTES.md
```
