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

## 4. Testing & Observability Standards
- **Mandatory Input & Output Print Statements:** Every test written must include explicit print statements:
  - Print the input parameters / prompt / payload before executing the call (e.g. `print(f"\n[TEST INPUT] ...")`).
  - Print the output result / response / state after execution (e.g. `print(f"[TEST OUTPUT] ...")`).
- **Test vs. Server Logging:** While server-side code (such as FastMCP over STDIO) must never use `print()` on stdout, test suites and demo scripts explicitly require print statements to give human developers instant visibility into data flow.
- **Post-Development Verification:** Every newly written or modified test must be executed immediately post-development to verify it runs properly and passes cleanly.
