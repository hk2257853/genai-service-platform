# Workspace Agent Directives

## First Action (Every New Session)
- **Read `.agents/PROJECT_STATE.md`** before doing anything else.

## Session Rules
1. **Lean Context:** Use `.agents/PROJECT_STATE.md` as the single source of truth. Do not re-read large codebases or past conversation files.
2. **Observable Tests:** Every test written must include clear print statements stating the input going in and the output received after execution.
3. **Post-Development Verification:** Never declare a feature or milestone complete without immediately executing the written test(s) (e.g. `pytest -s`) to inspect stdout/stderr and verify clean exit code `0`. Follow the checklist in `.agents/rules/verification-loop.md`.
4. **Walkthrough Test Results & Logs:** When providing the walkthrough, always paste the full test results along with all printed logs at the bottom.
5. **No Autonomous Commits:** Never execute `git commit` without explicit approval. When asked to commit, prepare the proposed commit message write-up, present it in chat, and ask for confirmation. Only execute `git commit` if the user replies with approval ("yes", "proceed", etc.).
6. **End-of-Session Handoff:** Before ending, update `NOTES.md` with learnings and `.agents/PROJECT_STATE.md` with the next session's starting goal.
