---
name: tdd-workflow
description: Provides a structured test-driven development and verification workflow for Python backend and AI agent components. Use when implementing new features, connecting agents to tools/MCP servers, writing verification scripts, or debugging regressions.
---

# Test-Driven Verification Workflow (TDD)

This workflow adapts classical TDD for agentic software development, prioritizing fast runtime feedback and zero-hallucination verification.

## The Agentic Verification Cycle

```
Define Requirement → Write/Target Verification Script → Implement/Wire Feature → Execute Script → Inspect & Fix → Verified Done
```

### 1. Identify the Verification Contract
- Before writing extensive business logic, establish:
  - What is the expected input?
  - What is the expected output or state change?
  - What failure modes should be tested (e.g. invalid IDs, missing parameters)?

### 2. Standalone Verification Scripts
- In complex multi-component stacks (e.g. FastAPI + LangGraph + FastMCP), prefer lightweight standalone verification scripts in `scripts/verify_*.py` or `tests/test_*.py`.
- Ensure tests verify real behavior rather than mocking out the core logic being tested.

### 3. Execution & Direct Inspection
- Execute tests directly using terminal run commands:
  ```bash
  python -m pytest tests/test_feature.py -v
  # OR direct script execution:
  python scripts/verify_mcp_agent.py
  ```
- Always inspect both `stdout` and `stderr`:
  - Verify return types (e.g., proper JSON strings, dictionary schemas).
  - Ensure no unhandled warnings or corrupted pipes.

### 4. Self-Correction Loop
- If an assertion fails or an exception is thrown:
  1. Isolate whether the error is in the test or the implementation.
  2. Apply targeted fixes.
  3. Re-run the verification script until all checks pass cleanly.

---

## This Project's Test Commands
- **Full test suite:** `python -m pytest ai_service_platform/tests/ -v`
- **MCP agent integration:** `python -m pytest ai_service_platform/tests/test_mcp_agent.py -v`
- **Interactive MCP demo:** `python scripts/demo_mcp_agent.py`

## Anti-Patterns (Learned from Days 1–13)
- Don't mock the MCP client in integration tests — test against the real STDIO transport.
- Don't use `print()` for debugging in MCP server code — it corrupts JSON-RPC. Use `logging` to stderr.
- Always verify tool call responses have the expected schema before asserting on content.
