---
name: tdd-workflow
description: Provides a structured test-driven development and verification workflow for Python backend and AI agent components. Use when implementing new features, connecting agents to tools/MCP servers, writing verification scripts, or debugging regressions.
---

# Test-Driven Verification Workflow (TDD)

This workflow adapts classical TDD for agentic software development, prioritizing fast runtime feedback and zero-hallucination verification.

## The Agentic Verification Cycle

```
Define Requirement → Write Observable Test (Input/Output Prints) → Implement Feature → Execute Test Post-Development → Inspect Logs & Fix → Paste Test Logs in Walkthrough → Verified Done
```

### 1. Identify the Verification Contract
- Before writing extensive business logic, establish:
  - What is the expected input?
  - What is the expected output or state change?
  - What failure modes should be tested (e.g. invalid IDs, missing parameters)?

### 2. Observable Test Design (Mandatory Print Statements)
- Any test that is written (pytest or standalone verification script) MUST have clear, descriptive print statements:
  - **Input Print:** State the input going in before the function or agent call.
  - **Output Print:** State the output received after execution.
- Example structure:
  ```python
  # 1. State the input going in
  print("\n" + "=" * 60)
  print(f"[TEST INPUT] Query / Payload: {prompt}")
  print("=" * 60)

  # 2. Execute target function or agent
  result = await agent.ainvoke(prompt=prompt, thread_id="test-thread")

  # 3. State the output received
  print("\n[TEST OUTPUT] Received Result:")
  print(f"  Response Content: {result['state']['messages'][-1].content}")
  print(f"  Tool Calls Made : {result['state']['tool_calls']}")
  print(f"  Token Usage     : {result['state']['accumulated_tokens']}")
  ```

### 3. Post-Development Execution & Direct Inspection
- Immediately post-development, run the test that was written and verify if it's working properly.
- Execute tests directly in terminal. **Always use `-s` with pytest** so print statements are visible on stdout:
  ```bash
  python -m pytest ai_service_platform/tests/test_feature.py -v -s
  # OR direct script execution:
  python scripts/verify_mcp_agent.py
  ```
- Inspect both `stdout` and `stderr`:
  - Analyze the printed inputs and outputs to ensure real behavior aligns with expectations.
  - Ensure process exits cleanly with exit code `0`.

### 4. Self-Correction Loop
- If an assertion fails or an exception is thrown:
  1. Isolate whether the error is in the test or the implementation.
  2. Apply targeted fixes.
  3. Re-run the verification script until all checks pass cleanly.

### 5. Walkthrough Documentation with Test Results & Logs
- When providing the final walkthrough (e.g. in `walkthrough.md` or the final response), paste the complete test execution results along with all the logs that were printed at the very bottom.
- Reading these logs provides clear verification of what inputs were evaluated and what outputs were produced.

---

## This Project's Test Commands
- **Full test suite:** `python -m pytest ai_service_platform/tests/ -v -s`
- **MCP agent integration:** `python -m pytest ai_service_platform/tests/test_mcp_agent.py -v -s`

## Anti-Patterns (Learned from Days 1–13)
- Don't mock the MCP client in integration tests — test against the real STDIO transport.
- Don't use `print()` for debugging inside MCP server code — it corrupts JSON-RPC (use `logging` to stderr). **However, in tests and verification scripts, DO use clear `print()` statements for inputs and outputs.**
- Always verify tool call responses have the expected schema before asserting on content.
