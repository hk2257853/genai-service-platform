# Verification & State Handoff Rules

## 1. The Planning, Verification & Delegation Invariants

### A. Start-of-Day Planning & Mandatory Human Approval
- When beginning a new day or task (e.g., *"Let's begin with Day X: [task]"*):
  1. **Brainstorm:** Brainstorm thoroughly how to execute the task cleanly, analyzing architecture, libraries, and edge cases.
  2. **Implementation Plan:** Produce a clear, structured implementation plan (`implementation_plan.md`).
  3. **Mandatory Review & Approval Gate:** Do not begin implementation autonomously. Present the plan and wait for the user to review and explicitly approve it before writing code.

### B. Sub-Agent Delegation (Strict Independence Only)
- During analysis or implementation, launch sub-agents to execute tasks in parallel **only if the parts are completely independent** (disjoint files, independent logic, no shared state or sequencing dependencies).
- **Prohibition on Overlapping Sub-Agents:** If tasks overlap, edit the same files, or depend on one another, do NOT use sub-agents. Execute sequentially to prevent concurrency conflicts and state corruption.

### C. The Verification Loop Invariant
- **Never claim a task or milestone is complete without executing a verification run.**
- Follow the deterministic workflow:
  ```
  Brainstorm → Plan → User Approval → Implement (Independent Sub-Agents if applicable) → Build/Import Check → Execute Tests/Verification & Inspect Logs → Document Walkthrough with Test Logs → Inspect Diff → Update Handoff
  ```

## 2. The "Definition of Done" Checklist
Before declaring any day, feature, or milestone complete, verify and satisfy every item below:
1. **Imports & Syntax Check:** No missing dependencies, unresolved imports, or syntax errors.
2. **Observable Test Writing (Input & Output Prints):**
   - Any test written (pytest or standalone verification script) MUST include clear, informative print statements:
     - A print statement before execution stating the input going in (e.g., `print(f"\n[TEST INPUT] Prompt/Payload: {input}")`).
     - A print statement after execution stating the output received (e.g., `print(f"[TEST OUTPUT] Result/Response: {output}")`).
3. **Post-Development Test Execution & Analysis:**
   - Immediately post-development, execute the test(s) that were written to analyze and verify they are working properly.
   - Run `pytest` with the `-s` flag (`python -m pytest <test_file> -v -s`) so print statements and logs are displayed in stdout.
   - The test run must exit with code `0` with all assertions passing.
4. **Channel & Protocol Integrity:**
   - In MCP servers: Verify that server-side logging is strictly directed to `stderr` and no stray `print()` calls pollute the JSON-RPC `stdout` stream. (Note: tests calling MCP clients may freely use `print()` for observability).
   - In LangGraph agents: Verify state transitions, tool call handling, and memory checkpointers return expected payloads.
5. **Walkthrough with Test Results & Logs:**
   - When providing the final walkthrough (e.g. in `walkthrough.md` or the final response), paste the complete test execution results along with all stdout/stderr logs that were printed at the bottom.
   - Reading these raw logs gives full visibility into the actual inputs, intermediate steps, and output responses.
6. **Git Diff Inspection & Commit Approval:**
   - Review `git status` and `git diff` to ensure no accidental debug debris, temporary scratch files, or unwanted modifications.
   - **Zero Autonomous Commits:** Never run `git commit` without prior approval. Prepare the commit message write-up, present it in chat, and ask for confirmation. Only execute `git commit` after the user explicitly says yes.
7. **Interview Notes Logged:** Append concise interview-recall notes to `NOTES.md` (core concepts, tradeoffs, "why X over Y", gotchas).
8. **State Handoff Updated:** Update `.agents/PROJECT_STATE.md` with newly verified components and the exact starting goal for the next day/session.

## 3. Session / Day-End Handoff Invariant
- At the end of every day:
  1. Confirm all 8 items in the "Definition of Done" checklist are met.
  2. Ensure `.agents/PROJECT_STATE.md` reflects the fresh warm-start context for the next session.
