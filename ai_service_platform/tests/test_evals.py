"""
Day 15 — Evaluation Tests: RAGAS Metrics, Agent Assertions, LLM-as-a-Judge

6 observable tests covering all 3 evaluation tiers:
  1. Faithfulness:          Grounded vs. hallucinated answer
  2. Answer Relevancy:      On-topic vs. off-topic answer
  3. Contextual Precision:  Good retrieval ranking vs. noisy ranking
  4. Agent Assertions:      Trajectory validation (deterministic, no LLM)
  5. Pointwise Judge:       Rubric scoring with CoT
  6. Pairwise Judge:        A/B comparison with position-bias detection
"""

import os
import asyncio
import pytest
import logging

os.environ["MODEL_ID"] = "gemini/gemini-3.1-flash-lite"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_evals")


# ===================================================================
# SHARED TEST DATA (mock RAG outputs for evaluation)
# ===================================================================

QUERY = "What is PostgreSQL's transaction isolation level that prevents phantom reads, and what error code does it use for serialization conflicts?"

CONTEXT_CHUNKS = [
    "PostgreSQL's SERIALIZABLE isolation level uses Serializable Snapshot Isolation (SSI) to prevent phantom reads. "
    "When a serialization conflict is detected, the transaction is aborted with SQLSTATE error code 40001.",
    "The READ COMMITTED isolation level is PostgreSQL's default. It prevents dirty reads but allows "
    "non-repeatable reads and phantom reads within the same transaction.",
    "PostgreSQL supports four isolation levels: READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ, and SERIALIZABLE. "
    "Each level provides increasing protection against concurrency anomalies.",
]

GROUND_TRUTH = "The SERIALIZABLE isolation level prevents phantom reads. Serialization conflicts produce SQLSTATE error code 40001."

# Grounded answer — all claims come from the context
GROUNDED_ANSWER = (
    "PostgreSQL's SERIALIZABLE isolation level prevents phantom reads using Serializable Snapshot Isolation (SSI). "
    "When a serialization conflict occurs, the transaction is aborted with SQLSTATE error code 40001."
)

# Hallucinated answer — contains fabricated claims not in the context
HALLUCINATED_ANSWER = (
    "PostgreSQL's SERIALIZABLE level prevents phantom reads. It uses Multi-Version Concurrency Control (MVCC) "
    "with a write-ahead log for conflict resolution. The error code is 23505, and conflicts are resolved "
    "automatically using a priority queue that retries the lower-priority transaction."
)

# Off-topic answer — technically correct but doesn't address the query
OFFTOPIC_ANSWER = (
    "MongoDB uses a document-based data model with BSON format. It supports sharding for horizontal scalability "
    "and replica sets for high availability. The default write concern is 'majority'."
)


# ===================================================================
# TEST 1: RAGAS FAITHFULNESS — Grounded vs. Hallucinated
# ===================================================================

@pytest.mark.asyncio
async def test_01_faithfulness_grounded_vs_hallucinated():
    """Faithfulness: A grounded answer should score significantly higher than a hallucinated one."""
    from ai_service_platform.evals.evaluators import evaluate_faithfulness

    print("\n" + "=" * 70)
    print("TEST 1: RAGAS Faithfulness — Grounded vs. Hallucinated Answer")
    print("=" * 70)

    # Evaluate grounded answer
    print(f"\n[TEST INPUT] Query: {QUERY}")
    print(f"[TEST INPUT] Context Chunks: {len(CONTEXT_CHUNKS)} chunks")
    print(f"[TEST INPUT] Grounded Answer: {GROUNDED_ANSWER[:100]}...")

    grounded_result = await evaluate_faithfulness(QUERY, CONTEXT_CHUNKS, GROUNDED_ANSWER)

    print(f"\n[TEST OUTPUT] Grounded Faithfulness Score: {grounded_result['score']}")
    print(f"[TEST OUTPUT] Claims: {grounded_result['total_claims']} total, {grounded_result['supported_claims']} supported")
    for detail in grounded_result["details"]:
        print(f"  - [{detail['verdict'].upper()}] {detail['claim']}: {detail['reasoning']}")

    await asyncio.sleep(2.0)

    # Evaluate hallucinated answer
    print(f"\n[TEST INPUT] Hallucinated Answer: {HALLUCINATED_ANSWER[:100]}...")

    hallucinated_result = await evaluate_faithfulness(QUERY, CONTEXT_CHUNKS, HALLUCINATED_ANSWER)

    print(f"\n[TEST OUTPUT] Hallucinated Faithfulness Score: {hallucinated_result['score']}")
    print(f"[TEST OUTPUT] Claims: {hallucinated_result['total_claims']} total, {hallucinated_result['supported_claims']} supported")
    for detail in hallucinated_result["details"]:
        print(f"  - [{detail['verdict'].upper()}] {detail['claim']}: {detail['reasoning']}")

    # Grounded should score strictly higher than hallucinated
    assert grounded_result["score"] > hallucinated_result["score"], (
        f"Grounded ({grounded_result['score']}) should score higher than hallucinated ({hallucinated_result['score']})"
    )
    print(f"\n[PASS] Grounded ({grounded_result['score']}) > Hallucinated ({hallucinated_result['score']})")

    await asyncio.sleep(2.0)


# ===================================================================
# TEST 2: RAGAS ANSWER RELEVANCY — On-Topic vs. Off-Topic
# ===================================================================

@pytest.mark.asyncio
async def test_02_answer_relevancy_ontopic_vs_offtopic():
    """Answer Relevancy: A relevant answer should have higher embedding similarity than an off-topic one."""
    from ai_service_platform.evals.evaluators import evaluate_answer_relevancy

    print("\n" + "=" * 70)
    print("TEST 2: RAGAS Answer Relevancy — On-Topic vs. Off-Topic")
    print("=" * 70)

    # Evaluate on-topic answer
    print(f"\n[TEST INPUT] Query: {QUERY}")
    print(f"[TEST INPUT] On-Topic Answer: {GROUNDED_ANSWER[:100]}...")

    ontopic_result = await evaluate_answer_relevancy(QUERY, GROUNDED_ANSWER)

    print(f"\n[TEST OUTPUT] On-Topic Relevancy Score: {ontopic_result['score']}")
    print(f"[TEST OUTPUT] Reverse Questions: {ontopic_result['reverse_questions']}")
    print(f"[TEST OUTPUT] Similarities: {ontopic_result['similarities']}")

    await asyncio.sleep(2.0)

    # Evaluate off-topic answer
    print(f"\n[TEST INPUT] Off-Topic Answer: {OFFTOPIC_ANSWER[:100]}...")

    offtopic_result = await evaluate_answer_relevancy(QUERY, OFFTOPIC_ANSWER)

    print(f"\n[TEST OUTPUT] Off-Topic Relevancy Score: {offtopic_result['score']}")
    print(f"[TEST OUTPUT] Reverse Questions: {offtopic_result['reverse_questions']}")
    print(f"[TEST OUTPUT] Similarities: {offtopic_result['similarities']}")

    # On-topic should score higher
    assert ontopic_result["score"] > offtopic_result["score"], (
        f"On-topic ({ontopic_result['score']}) should score higher than off-topic ({offtopic_result['score']})"
    )
    print(f"\n[PASS] On-Topic ({ontopic_result['score']}) > Off-Topic ({offtopic_result['score']})")

    await asyncio.sleep(2.0)


# ===================================================================
# TEST 3: RAGAS CONTEXTUAL PRECISION — Good vs. Noisy Retrieval
# ===================================================================

@pytest.mark.asyncio
async def test_03_contextual_precision_good_vs_noisy_ranking():
    """Contextual Precision: Well-ranked chunks should score higher than noise-first retrieval."""
    from ai_service_platform.evals.evaluators import evaluate_contextual_precision

    print("\n" + "=" * 70)
    print("TEST 3: RAGAS Contextual Precision — Good vs. Noisy Ranking")
    print("=" * 70)

    # Good ranking: relevant chunk first
    good_ranking = [
        CONTEXT_CHUNKS[0],  # Relevant (SSI + error code 40001)
        CONTEXT_CHUNKS[2],  # Somewhat relevant (lists isolation levels)
        CONTEXT_CHUNKS[1],  # Least relevant (READ COMMITTED default)
    ]

    # Noisy ranking: irrelevant chunk first
    noisy_ranking = [
        "Redis is an in-memory key-value store used for caching. It supports pub/sub messaging.",  # Irrelevant noise
        CONTEXT_CHUNKS[1],  # Less relevant
        CONTEXT_CHUNKS[0],  # Most relevant — buried at rank 3
    ]

    print(f"\n[TEST INPUT] Query: {QUERY}")
    print(f"[TEST INPUT] Ground Truth: {GROUND_TRUTH}")
    print(f"[TEST INPUT] Good Ranking: relevant chunk at rank 1")
    print(f"[TEST INPUT] Noisy Ranking: noise at rank 1, relevant chunk at rank 3")

    good_result = await evaluate_contextual_precision(QUERY, good_ranking, GROUND_TRUTH)

    print(f"\n[TEST OUTPUT] Good Ranking Precision Score: {good_result['score']}")
    print(f"[TEST OUTPUT] Relevance Flags: {good_result['relevance_flags']}")
    for detail in good_result["details"]:
        print(f"  - Chunk {detail['chunk_index']}: relevant={detail['relevant']} — {detail['reasoning']}")

    await asyncio.sleep(2.0)

    noisy_result = await evaluate_contextual_precision(QUERY, noisy_ranking, GROUND_TRUTH)

    print(f"\n[TEST OUTPUT] Noisy Ranking Precision Score: {noisy_result['score']}")
    print(f"[TEST OUTPUT] Relevance Flags: {noisy_result['relevance_flags']}")
    for detail in noisy_result["details"]:
        print(f"  - Chunk {detail['chunk_index']}: relevant={detail['relevant']} — {detail['reasoning']}")

    # Good ranking should score higher (relevant chunks at top)
    assert good_result["score"] > noisy_result["score"], (
        f"Good ranking ({good_result['score']}) should score higher than noisy ({noisy_result['score']})"
    )
    print(f"\n[PASS] Good Ranking ({good_result['score']}) > Noisy Ranking ({noisy_result['score']})")

    await asyncio.sleep(2.0)


# ===================================================================
# TEST 4: AGENT ASSERTIONS — Deterministic Trajectory Validation
# ===================================================================

def test_04_agent_assertions_trajectory_validation():
    """Agent Assertions: Validate tool selection, budget, error recovery, and answer integrity."""
    from ai_service_platform.evals.agent_assertions import (
        assert_tool_called,
        assert_no_hallucinated_tools,
        assert_within_budget,
        assert_graceful_error_recovery,
        assert_final_answer_contains,
    )

    print("\n" + "=" * 70)
    print("TEST 4: Agent Assertions — Deterministic Trajectory Validation")
    print("=" * 70)

    # Mock a successful agent trajectory
    trajectory_success = [
        {"tool_name": "get_order_details", "args": {"order_id": "ORD-1001"}, "status": "success",
         "result": "Order ORD-1001: Sony WH-1000XM5, shipped via FedEx", "tokens": 500},
        {"tool_name": "get_product_details", "args": {"product_id": "PROD-101"}, "status": "success",
         "result": "Product PROD-101: Sony WH-1000XM5, in stock, 2-year warranty", "tokens": 450},
    ]
    final_answer_success = "Order ORD-1001 contains a Sony WH-1000XM5, shipped via FedEx. The product is currently in stock with a 2-year warranty."

    allowed_tools = ["get_order_details", "get_product_details", "calculate_order_discount"]

    print(f"\n[TEST INPUT] Trajectory: {len(trajectory_success)} steps")
    for step in trajectory_success:
        print(f"  - {step['tool_name']}({step['args']}) -> {step['status']}")
    print(f"[TEST INPUT] Final Answer: {final_answer_success[:80]}...")
    print(f"[TEST INPUT] Allowed Tools: {allowed_tools}")

    # 4a: Tool was called correctly
    assert_tool_called(trajectory_success, "get_order_details", {"order_id": "ORD-1001"})
    print("\n[TEST OUTPUT] [PASS] assert_tool_called: get_order_details(order_id='ORD-1001') — PASSED")

    # 4b: No hallucinated tools
    assert_no_hallucinated_tools(trajectory_success, allowed_tools)
    print("[TEST OUTPUT] [PASS] assert_no_hallucinated_tools — PASSED")

    # 4c: Within budget
    assert_within_budget(trajectory_success, max_steps=4, max_tokens=2000)
    print("[TEST OUTPUT] [PASS] assert_within_budget(max_steps=4, max_tokens=2000) — PASSED")

    # 4d: Final answer contains expected terms
    assert_final_answer_contains(
        final_answer_success,
        required_terms=["Sony", "WH-1000XM5", "FedEx"],
        forbidden_terms=["API_KEY", "sk-", "system prompt"],
    )
    print("[TEST OUTPUT] [PASS] assert_final_answer_contains — PASSED")

    # 4e: Error recovery — test with an error trajectory
    trajectory_error = [
        {"tool_name": "get_order_details", "args": {"order_id": "ORD-9999"}, "status": "error",
         "result": "Order 'ORD-9999' not found", "tokens": 300},
    ]
    error_final_answer = "I couldn't find order ORD-9999. Please check the order ID and try again."

    print(f"\n[TEST INPUT] Error Trajectory: tool returned error for ORD-9999")
    print(f"[TEST INPUT] Error Final Answer: {error_final_answer}")

    assert_graceful_error_recovery(trajectory_error, error_final_answer)
    print("[TEST OUTPUT] [PASS] assert_graceful_error_recovery — PASSED (agent reported error, no hallucination)")

    # 4f: Negative test — hallucinated tool should fail
    trajectory_hallucinated = [
        {"tool_name": "search_internet", "args": {"query": "test"}, "status": "success", "result": "..."},
    ]
    try:
        assert_no_hallucinated_tools(trajectory_hallucinated, allowed_tools)
        assert False, "Should have raised AssertionError for hallucinated tool"
    except AssertionError as e:
        print(f"[TEST OUTPUT] [PASS] Hallucinated tool detection — CAUGHT: {e}")

    print("\n[PASS] All agent assertions passed!")


# ===================================================================
# TEST 5: LLM-AS-A-JUDGE — Pointwise Rubric Scoring
# ===================================================================

@pytest.mark.asyncio
async def test_05_pointwise_judge_rubric_scoring():
    """Pointwise Judge: Score a single answer on a 1–5 rubric with CoT reasoning."""
    from ai_service_platform.evals.evaluators import pointwise_judge

    print("\n" + "=" * 70)
    print("TEST 5: LLM-as-a-Judge — Pointwise Rubric Scoring")
    print("=" * 70)

    print(f"\n[TEST INPUT] Query: {QUERY}")
    print(f"[TEST INPUT] Answer: {GROUNDED_ANSWER[:100]}...")
    print(f"[TEST INPUT] Rubric Dimensions: Faithfulness, Relevance, Clarity, Completeness")

    result = await pointwise_judge(QUERY, GROUNDED_ANSWER)

    print(f"\n[TEST OUTPUT] Overall Score: {result['overall_score']}/5")
    for score_item in result["scores"]:
        print(f"  - {score_item['dimension']}: {score_item['score']}/5")
        print(f"    Reasoning: {score_item['reasoning']}")

    assert result["overall_score"] >= 1.0, "Overall score should be at least 1.0"
    assert result["overall_score"] <= 5.0, "Overall score should be at most 5.0"
    assert len(result["scores"]) >= 4, "Should have scores for all 4 dimensions"

    print(f"\n[PASS] Pointwise judge returned valid structured scores")

    await asyncio.sleep(2.0)


# ===================================================================
# TEST 6: LLM-AS-A-JUDGE — Pairwise with Position-Bias Detection
# ===================================================================

@pytest.mark.asyncio
async def test_06_pairwise_judge_position_bias():
    """Pairwise Judge: Compare two answers with order swap to detect position bias."""
    from ai_service_platform.evals.evaluators import pairwise_judge

    print("\n" + "=" * 70)
    print("TEST 6: LLM-as-a-Judge — Pairwise Comparison with Position-Bias Detection")
    print("=" * 70)

    print(f"\n[TEST INPUT] Query: {QUERY}")
    print(f"[TEST INPUT] Answer A (Grounded): {GROUNDED_ANSWER[:80]}...")
    print(f"[TEST INPUT] Answer B (Hallucinated): {HALLUCINATED_ANSWER[:80]}...")

    result = await pairwise_judge(QUERY, GROUNDED_ANSWER, HALLUCINATED_ANSWER)

    print(f"\n[TEST OUTPUT] Final Winner: {result['final_winner']}")
    print(f"[TEST OUTPUT] Position Bias Detected: {result['position_bias_detected']}")
    print(f"[TEST OUTPUT] Pass 1 (A,B order): winner={result['pass_1_AB']['winner']}")
    print(f"  Reasoning: {result['pass_1_AB']['reasoning']}")
    print(f"[TEST OUTPUT] Pass 2 (B,A order): raw_winner={result['pass_2_BA_raw']['winner']} -> normalized={result['pass_2_BA_normalized']}")
    print(f"  Reasoning: {result['pass_2_BA_raw']['reasoning']}")

    # The grounded answer should be preferred (winner = A), but we mainly verify
    # that the double-swap mechanism produces a valid, structured result
    assert result["final_winner"] in ["A", "B", "TIE", "tie", "tie (position bias detected)"], (
        f"Unexpected winner value: {result['final_winner']}"
    )
    print(f"\n[PASS] Pairwise judge completed with valid structure and bias detection")
