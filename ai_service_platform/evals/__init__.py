"""Evaluation module for AI outputs: RAGAS metrics, Agent assertions, LLM-as-Judge."""

from ai_service_platform.evals.evaluators import (
    evaluate_faithfulness,
    evaluate_answer_relevancy,
    evaluate_contextual_precision,
    run_ragas_evaluation,
    pointwise_judge,
    pairwise_judge,
)
from ai_service_platform.evals.agent_assertions import (
    assert_tool_called,
    assert_no_hallucinated_tools,
    assert_within_budget,
    assert_graceful_error_recovery,
    assert_final_answer_contains,
)
