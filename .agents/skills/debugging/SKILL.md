---
name: debugging
description: Systematic root-cause debugging playbook for Python services, async pipelines, MCP servers, and LangGraph agent execution failures. Use when encountering exceptions, subprocess communication issues, tool-calling bugs, or test failures.
---

# Systematic Debugging Playbook

When debugging agent, service, or server failures, follow a hypothesis-driven process rather than guessing or making speculative code edits.

## 1. Triage & Error Characterization
- **Read the full traceback:** Identify the exact line, module, and exception type.
- **Categorize the failure mode:**
  - **Process / IPC Error:** Is STDIO being polluted? Check if unexpected `print()` statements are writing to stdout instead of stderr in FastMCP.
  - **Serialization Error:** Did a DB or service return non-serializable types (`datetime`, `Decimal`, `UUID`) into JSON-RPC?
  - **State / Type Mismatch:** Are LangGraph message types (`AIMessage`, `ToolMessage`, `HumanMessage`) or tool call arguments malformed?
  - **Async / Event Loop Issue:** Is an async coroutine being called synchronously or un-awaited?

## 2. Minimal Reproduction
- Do not debug through a 5-step UI or full agent loop if you can isolate the failing node.
- Write a 5-line reproduction snippet to run the failing function in isolation:
  ```bash
  python -c "import asyncio; from ai_service_platform.mcp_servers.ecommerce_server import get_order_details; print(get_order_details('ORD-1001'))"
  ```

## 3. Targeted Fix & Regression Check
- Fix the root cause at the source.
- Re-run the full verification script or test suite to ensure the fix didn't break adjacent functionality.
- Document any non-obvious root causes or gotchas in `NOTES.md`.

---

## Common Failures in This Project

| Symptom | Likely Cause | Fix |
|---|---|---|
| `JSONDecodeError` in MCP client | `print()` polluting stdout | Replace with `logging.error()` to stderr |
| `ToolException: tool not found` | MCP server not started or tool name mismatch | Check `tools/list` response from server |
| `asyncio.run() cannot be called from a running event loop` | Nested event loops | Use `await` directly or `nest_asyncio.apply()` |
| `ValidationError` from Pydantic | v1 vs v2 API mismatch | Use `model_validate()` instead of `parse_obj()` |
