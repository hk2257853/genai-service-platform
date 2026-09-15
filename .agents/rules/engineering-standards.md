# Engineering Standards & Invariants

## 1. Framework-First, Always
- If an established library solves the problem (LangGraph, FastMCP, LlamaIndex, Pydantic, LiteLLM), **always use it**.
- Never hand-roll custom implementations (e.g. do not hand-roll a raw ReAct loop when LangGraph is available).
- When integrating a framework, document your understanding in `NOTES.md` (core concepts, tradeoffs, gotchas). Do not write parallel hand-rolled code.

## 2. Clean Code Rules
- **No Duplication:** Shared logic belongs in shared modules (`config`, `base`, `utils`).
- **Explicit Configuration:** Use typed settings or env vars; no hardcoded credentials or magic numbers.
- **Single Responsibility:** Keep functions/classes focused. No god functions mixing I/O, logic, and setup.
- **Fail Loudly:** Structured error responses; never swallow exceptions silently.
- **Docstrings as Contracts:** In MCP/tool-calling contexts, docstrings are the LLM's prompt interface. Keep them accurate and constraint-focused.

## 3. Project Structure
- New dependencies go in `requirements.txt` AND `pyproject.toml`.
- New source modules go under `ai_service_platform/<domain>/`.
- Tests go in `ai_service_platform/tests/test_<module>.py`.
- Verification/demo scripts go in `scripts/`.
