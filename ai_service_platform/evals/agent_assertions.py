"""
Deterministic Agent Trajectory Assertions

These are pure Python assertion helpers — no LLM calls.
They validate agent behavior by inspecting the execution trajectory
(tool calls, step counts, token budgets, error recovery).

Why agent evals differ from RAG evals:
  - RAG evals measure the quality of the *text output* (groundedness, relevancy).
  - Agent evals measure the quality of the *execution path* (which tools, what order,
    how many steps, did it recover from errors?).

Expected trajectory format (list of dicts):
  [
      {"tool_name": "get_order_details", "args": {"order_id": "ORD-1001"}, "status": "success", "result": "..."},
      {"tool_name": "get_product_details", "args": {"product_id": "PROD-101"}, "status": "success", "result": "..."},
  ]
"""

from typing import Any


def assert_tool_called(
    trajectory: list[dict[str, Any]],
    tool_name: str,
    expected_args: dict[str, Any] | None = None,
) -> None:
    """
    Asserts that a specific tool was invoked in the trajectory.
    Optionally verifies that the tool was called with specific arguments.

    Raises AssertionError with a descriptive message on failure.
    """
    tool_calls = [t for t in trajectory if t.get("tool_name") == tool_name]
    assert tool_calls, (
        f"Expected tool '{tool_name}' to be called, but it was not found in trajectory. "
        f"Tools called: {[t.get('tool_name') for t in trajectory]}"
    )

    if expected_args:
        matched = False
        for call in tool_calls:
            actual_args = call.get("args", {})
            if all(actual_args.get(k) == v for k, v in expected_args.items()):
                matched = True
                break
        assert matched, (
            f"Tool '{tool_name}' was called, but not with expected args {expected_args}. "
            f"Actual calls: {[{'args': t.get('args')} for t in tool_calls]}"
        )


def assert_no_hallucinated_tools(
    trajectory: list[dict[str, Any]],
    allowed_tools: list[str],
) -> None:
    """
    Asserts that every tool invoked in the trajectory exists in the allowed tool catalog.

    Catches the "tool hallucination" failure mode where an LLM invents nonexistent
    tool names (e.g., calling "search_internet" when only "get_order_details" exists).
    """
    for step in trajectory:
        tool = step.get("tool_name")
        if tool is not None:
            assert tool in allowed_tools, (
                f"Hallucinated tool detected: '{tool}' is not in the allowed catalog {allowed_tools}"
            )


def assert_within_budget(
    trajectory: list[dict[str, Any]],
    max_steps: int,
    max_tokens: int | None = None,
    max_cost: float | None = None,
) -> None:
    """
    Asserts that the agent completed within the allowed step/token/cost budget.

    Protects against:
      - Infinite loops (agent keeps calling tools without converging)
      - Token blowup (context grows O(N^2) across turns in long agent loops)
      - Cost overruns (each LLM call costs money)
    """
    step_count = len(trajectory)
    assert step_count <= max_steps, (
        f"Agent exceeded step budget: {step_count} steps (max allowed: {max_steps})"
    )

    if max_tokens is not None:
        total_tokens = sum(step.get("tokens", 0) for step in trajectory)
        assert total_tokens <= max_tokens, (
            f"Agent exceeded token budget: {total_tokens} tokens (max allowed: {max_tokens})"
        )

    if max_cost is not None:
        total_cost = sum(step.get("cost", 0.0) for step in trajectory)
        assert total_cost <= max_cost, (
            f"Agent exceeded cost budget: ${total_cost:.6f} (max allowed: ${max_cost:.6f})"
        )


def assert_graceful_error_recovery(
    trajectory: list[dict[str, Any]],
    final_answer: str,
) -> None:
    """
    When a tool returns an error (e.g., "order not found"), the agent should:
      1. NOT crash or raise an unhandled exception.
      2. NOT hallucinate data that wasn't returned by the tool.
      3. Inform the user helpfully about the error.

    This assertion checks that if any tool in the trajectory returned an error,
    the final answer does not contain hallucinated success data.
    """
    error_steps = [t for t in trajectory if t.get("status") == "error"]

    if not error_steps:
        return  # No errors to recover from — assertion trivially passes

    # The final answer should NOT claim success if all tools failed
    all_failed = all(t.get("status") == "error" for t in trajectory)
    if all_failed:
        success_indicators = ["here are the details", "the order is", "the status is"]
        answer_lower = final_answer.lower()
        for indicator in success_indicators:
            assert indicator not in answer_lower, (
                f"All tools returned errors, but the final answer contains success language: '{indicator}'. "
                f"This suggests the agent hallucinated a successful result instead of reporting the error."
            )


def assert_final_answer_contains(
    final_answer: str,
    required_terms: list[str] | None = None,
    forbidden_terms: list[str] | None = None,
) -> None:
    """
    Validates the final answer content:
      - required_terms: At least one must appear (case-insensitive).
      - forbidden_terms: None may appear (catches leaked system prompts, credentials, etc.).
    """
    answer_lower = final_answer.lower()

    if required_terms:
        found = any(term.lower() in answer_lower for term in required_terms)
        assert found, (
            f"Final answer is missing expected content. "
            f"Expected at least one of {required_terms} in the answer."
        )

    if forbidden_terms:
        for term in forbidden_terms:
            assert term.lower() not in answer_lower, (
                f"Final answer contains forbidden content: '{term}'. "
                f"This may indicate leaked system prompts or credentials."
            )
