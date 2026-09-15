# Workspace Agent Directives

## First Action (Every New Session)
- **Read `.agents/PROJECT_STATE.md`** before doing anything else.

## Session Rules
1. **Lean Context:** Use `.agents/PROJECT_STATE.md` as the single source of truth. Do not re-read large codebases or past conversation files.
2. **Verify Before Claiming Done:** Never declare a feature or milestone complete without running a test command and inspecting stdout/stderr. Follow the checklist in `.agents/rules/verification-loop.md`.
3. **End-of-Session Handoff:** Before ending, update `NOTES.md` with learnings and `.agents/PROJECT_STATE.md` with the next session's starting goal.
