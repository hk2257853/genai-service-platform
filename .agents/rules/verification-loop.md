# Verification & State Handoff Rules

## 1. The Verification Loop Invariant
- **Never claim a task or milestone is complete without executing a verification run.**
- Follow the deterministic workflow:
  ```
  Plan → Implement → Build/Import Check → Execute Tests/Verification → Inspect Diff → Update Handoff
  ```

## 2. The "Definition of Done" Checklist
Before declaring any day, feature, or milestone complete, verify and satisfy every item below:
1. **Imports & Syntax Check:** No missing dependencies, unresolved imports, or syntax errors.
2. **Deterministic Test Execution:** Run the target verification script or `pytest`. Process must exit with code `0`.
3. **Channel & Protocol Integrity:**
   - In MCP servers: Verify that logging is strictly directed to `stderr` and no stray `print()` calls pollute the JSON-RPC `stdout` stream.
   - In LangGraph agents: Verify state transitions, tool call handling, and memory checkpointers return expected payloads.
4. **Git Diff Inspection:** Review `git status` and `git diff` to ensure no accidental debug debris, temporary scratch files, or unwanted modifications.
5. **Interview Notes Logged:** Append concise interview-recall notes to `NOTES.md` (core concepts, tradeoffs, "why X over Y", gotchas).
6. **State Handoff Updated:** Update `.agents/PROJECT_STATE.md` with newly verified components and the exact starting goal for the next day/session.

## 3. Session / Day-End Handoff Invariant
- At the end of every day:
  1. Confirm all 6 items in the "Definition of Done" checklist are met.
  2. Ensure `.agents/PROJECT_STATE.md` reflects the fresh warm-start context for the next session.
