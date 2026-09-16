# Workspace Agent Directives

## First Action (Every New Session)
- **Read `.agents/PROJECT_STATE.md`** before doing anything else.

## Session Rules
1. **Lean Context:** Use `.agents/PROJECT_STATE.md` as the single source of truth. Do not re-read large codebases or past conversation files.
2. **Start-of-Day Workflow (Brainstorm → Plan → Approve):** When starting a day/task (e.g., *"Let's begin with Day X: [task]"*):
   - **Brainstorm:** Brainstorm thoroughly how to execute the task cleanly, evaluating architectural choices, edge cases, and library patterns.
   - **Implementation Plan:** Formulate a structured implementation plan (`implementation_plan.md`).
   - **Mandatory Approval Gate:** Never begin implementation on your own. Wait for explicit user review and approval in chat before writing any code.
3. **Sub-Agent Delegation (Strict Independence Only):**
   - During analysis or implementation, launch sub-agents to parallelize work and speed up execution **only if the tasks are completely independent** (disjoint files, no shared state or sequential dependencies).
   - If tasks are overlapping, interdependent, or touch the same files, **do not use sub-agents**; execute them sequentially to avoid collisions.
4. **Observable Tests:** Every test written must include clear print statements stating the input going in and the output received after execution.
5. **Post-Development Verification:** Never declare a feature or milestone complete without immediately executing the written test(s) (e.g. `pytest -s`) to inspect stdout/stderr and verify clean exit code `0`. Follow the checklist in `.agents/rules/verification-loop.md`.
6. **Walkthrough Test Results & Logs:** When providing the walkthrough, always paste the full test results along with all printed logs at the bottom.
7. **No Autonomous Commits:** Never execute `git commit` without explicit approval. When asked to commit, prepare the proposed commit message write-up, present it in chat, and ask for confirmation. Only execute `git commit` if the user replies with approval ("yes", "proceed", etc.).
8. **End-of-Session Handoff:** Before ending, update `NOTES.md` with learnings following the standards in `.agents/rules/engineering-standards.md` (high technical depth, zero format echoing/redundancy, natural textbook voice), and update `.agents/PROJECT_STATE.md` with the next session's starting goal.

