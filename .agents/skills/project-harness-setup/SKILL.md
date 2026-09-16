---
name: project-harness-setup
description: Blueprint and bootstrapping guide for initializing the lean agent harness (PROJECT_STATE.md, always-on rules, modular skills, and verification loop) in a new repository or learning project. For setup and scaffolding purposes only; not needed during normal day-to-day development in an already initialized project.
---

> [!NOTE]
> **BOOTSTRAP REFERENCE ONLY:** This skill is a self-contained setup playbook for initializing *new* repositories or tech stacks with this agentic harness. The agent does **not** need to consult or load this skill during normal day-to-day development of an already initialized project.

# Project Harness Setup & Bootstrapping Guide

Use this guide whenever you spin up a new repository (in Python, Java/Spring, TypeScript/Node, Go, or any other stack) to immediately eliminate "Agent Amnesia" and token waste.

---

## 1. Directory Structure Blueprint

In your new repository root, create the following minimal layout:

```text
<project-root>/
├── .agents/
│   ├── AGENTS.md                      # Root directives (auto-loaded by agent engines)
│   ├── PROJECT_STATE.md               # Dynamic memory handoff (~30–50 lines)
│   ├── rules/
│   │   ├── engineering-standards.md   # Core stack invariants & coding rules
│   │   └── verification-loop.md       # "Definition of Done" & test requirements
│   └── skills/
│       ├── tdd-workflow/
│       │   └── SKILL.md               # Lazy-loaded testing playbook
│       └── debugging/
│           └── SKILL.md               # Root-cause debugging playbook
└── NOTES.md                           # Running learnings, trade-offs, and gotchas
```

---

## 2. Step-by-Step Setup Instructions

### Step 1: Create `.agents/PROJECT_STATE.md`
Create `.agents/PROJECT_STATE.md` inside the `.agents/` directory. Keep it under 50 lines.

```markdown
# Current Project State

## 1. Active Tech Stack
- Frameworks / Runtimes: [e.g. FastAPI, LangGraph, Spring Boot]
- Database / Storage: [e.g. Postgres, SQLite, Redis]
- Tools & Protocols: [e.g. FastMCP, REST, Kafka]

## 2. Completed Milestones
- Phase 1: [Brief bullet on what is built and tested]
- Phase 2: [Brief bullet on what is built and tested]

## 3. Active Goal: [Next Immediate Step]
- Objective: [One clear goal for the current/next session]
- Key Tasks:
  1. [Task 1]
  2. [Task 2]

## 4. Key Entry Points
- Main Service: `path/to/entrypoint`
- Test Suites: `path/to/tests`
- Documentation: `NOTES.md`
```

---

### Step 2: Create `.agents/AGENTS.md`
Create `.agents/AGENTS.md` to establish workspace-level invariants:

```markdown
# Workspace Agent Directives

## Core Invariants
1. **Framework-First, Always:** Never hand-roll custom versions where mature, industry-standard libraries exist.
2. **Lean Context & Memory Handoff:** Always consult `.agents/PROJECT_STATE.md` for current architecture and active milestones instead of speculatively re-reading past files.
3. **Start-of-Day Workflow (Brainstorm → Plan → Approve):** When given a new day/task, brainstorm the approach and generate `implementation_plan.md`. Never implement autonomously; wait for user approval in chat.
4. **Sub-Agent Delegation (Strict Independence Only):** Launch sub-agents to parallelize work only for completely independent parts. If tasks overlap or touch shared files, execute sequentially.
5. **Observable Tests:** Every test written must include clear print statements stating the input going in and the output received after execution.
6. **Post-Development Verification:** Never claim a feature or milestone is complete without immediately running the test(s) (e.g. `pytest -s`) to inspect stdout/stderr and verify clean exit code `0`.
7. **Walkthrough Test Results & Logs:** Always paste full test results along with all printed logs at the bottom of the walkthrough.
8. **No Autonomous Commits:** Never execute `git commit` without explicit approval. Present the proposed commit message write-up in chat and wait for user confirmation.
9. **Incremental Notes:** Record takeaways, trade-offs, and gotchas in `NOTES.md` at the end of each session.
10. **First Action:** At the start of every session, read `.agents/PROJECT_STATE.md` before doing anything else.
```

---

### Step 3: Create `.agents/rules/`
Create two Markdown rules inside `.agents/rules/`:

1. **`engineering-standards.md`**: Specify stack-specific clean code invariants (e.g. type safety, error boundaries, no swallowed exceptions, configuration via environment variables).
2. **`verification-loop.md`**: Enforce the **Definition of Done (DoD)** checklist:
   ```markdown
   # Definition of Done Checklist
   Before declaring any feature complete:
   1. [ ] Imports & Syntax: No unresolved imports or build errors.
   2. [ ] Observable Tests: Clear print statements showing input going in and output received.
   3. [ ] Deterministic Test Execution: Verification command (`pytest -v -s`) exits with code 0.
   4. [ ] Channel Integrity: No rogue stdout logging corrupting protocol streams.
   5. [ ] Walkthrough Test Results: Full test output and printed logs pasted at bottom of walkthrough.
   6. [ ] Git Diff Clean & Commit Approval: No debug debris; commit message write-up presented in chat for user confirmation.
   7. [ ] Notes Logged: Concepts and trade-offs added to `NOTES.md`.
   8. [ ] State Handoff: `.agents/PROJECT_STATE.md` updated for the next session.
   ```

---

### Step 4: Create Starter Skills (`.agents/skills/`)
Skills are on-demand playbooks. Each skill lives in `.agents/skills/<name>/SKILL.md` with YAML frontmatter:

```markdown
---
name: <skill-name>
description: <Explicit trigger condition explaining WHEN the agent should read this skill>
---

# <Skill Title>
[Step-by-step procedural playbook for TDD, debugging, deployment, etc.]
```

*Tip:* Copy `tdd-workflow/` and `debugging/` from this repo as universal starting skills.

---

## 3. The Daily Operating Rhythm (How to Work)

1. **Start of Session (New Chat):**
   - Prompt:
     > *"Let's begin with Day X: [one-line task description]."*
   - **Step 1 (Brainstorm & Plan):** The agent brainstorms how to execute the task cleanly and formulates a structured `implementation_plan.md`.
   - **Step 2 (Mandatory Review Gate):** The agent stops and waits for explicit user review and approval in chat. It must **never** execute the implementation autonomously.

2. **During Implementation:**
   - Once the plan is approved, proceed into implementation.
   - **Sub-Agent Delegation (Strict Independence Only):** If parts of analysis or implementation are completely independent (disjoint files, no shared state), launch sub-agents to execute in parallel and accelerate delivery. If tasks overlap or touch shared files, execute sequentially in the main thread to prevent collisions.
   - Write observable tests with input and output print statements and execute verification commands in terminal (`pytest -v -s`).
   - Satisfy all items in the "Definition of Done".

3. **End of Session (Wrap-Up):**
   - Prompt:
     > *"We are concluding this milestone. Please update NOTES.md with today's trade-offs and update .agents/PROJECT_STATE.md for the next session."*
   - Present the proposed Git commit message write-up in chat and request user confirmation.
   - Once the user explicitly approves ("yes" / "proceed"), commit all code and the updated `.agents/PROJECT_STATE.md` together in Git.
