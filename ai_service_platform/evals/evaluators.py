"""
Evaluation Engine: RAGAS Metrics + LLM-as-a-Judge

RAGAS (Retrieval-Augmented Generation Assessment):
  - Faithfulness:          Are the claims in the answer grounded in the retrieved context?
  - Answer Relevancy:      Does the generated answer actually address the user's query?
  - Contextual Precision:  Are the relevant chunks ranked at the top of the retrieval results?

LLM-as-a-Judge:
  - Pointwise:  Score a single answer on a rubric (1–5) with CoT reasoning.
  - Pairwise:   Compare two answers head-to-head with position-bias mitigation (double-swap).
"""

import json
import logging
from typing import Any

import litellm
import numpy as np

from pydantic import BaseModel, Field

from ai_service_platform import config
from ai_service_platform.core.client import AsyncLLMClient
from ai_service_platform.rag.embeddings import get_embeddings_batch

logger = logging.getLogger("evals")


# ---------------------------------------------------------------------------
# Pydantic Schemas for Structured LLM Output
# ---------------------------------------------------------------------------

class ClaimExtractionResult(BaseModel):
    """LLM output: list of atomic factual claims extracted from an answer."""
    claims: list[str] = Field(description="List of atomic factual claims from the answer")


class ClaimVerificationResult(BaseModel):
    """LLM output: verdict for a single claim against context."""
    claim: str = Field(description="The claim being verified")
    verdict: str = Field(description="'supported' or 'unsupported'")
    reasoning: str = Field(description="Brief explanation of why")


class ClaimVerificationBatch(BaseModel):
    """LLM output: batch verification of all claims."""
    verdicts: list[ClaimVerificationResult]


class ReverseQuestions(BaseModel):
    """LLM output: synthetic questions that the answer would address."""
    questions: list[str] = Field(description="3 questions this answer would respond to")


class ChunkRelevanceVerdict(BaseModel):
    """LLM output: whether a single retrieved chunk is relevant to answering the query."""
    chunk_index: int = Field(description="0-based index of the chunk")
    relevant: bool = Field(description="True if this chunk contains information useful for answering the query")
    reasoning: str = Field(description="Brief explanation")


class ChunkRelevanceBatch(BaseModel):
    """LLM output: relevance verdicts for all retrieved chunks."""
    verdicts: list[ChunkRelevanceVerdict]


class PointwiseScore(BaseModel):
    """LLM output: rubric-based score for a single dimension."""
    dimension: str = Field(description="The evaluation dimension name")
    score: int = Field(description="Score from 1 to 5")
    reasoning: str = Field(description="Chain-of-thought reasoning for this score")


class PointwiseJudgement(BaseModel):
    """LLM output: complete pointwise evaluation across all dimensions."""
    scores: list[PointwiseScore]
    overall_score: float = Field(description="Weighted average score across all dimensions")


class PairwiseVerdict(BaseModel):
    """LLM output: which answer is better in a head-to-head comparison."""
    winner: str = Field(description="'A', 'B', or 'tie'")
    reasoning: str = Field(description="Chain-of-thought reasoning for the preference")


# ---------------------------------------------------------------------------
# Shared LLM Client
# ---------------------------------------------------------------------------

def _get_client() -> AsyncLLMClient:
    """Returns a shared AsyncLLMClient using the project's configured MODEL_ID."""
    return AsyncLLMClient(default_model=config.MODEL_ID, max_retries=3)


# ===================================================================
# RAGAS METRIC 1: FAITHFULNESS
# ===================================================================

async def evaluate_faithfulness(
    query: str,
    context_chunks: list[str],
    answer: str,
) -> dict[str, Any]:
    """
    Evaluates faithfulness (groundedness) of a generated answer against retrieved context.

    Algorithm:
      1. Decompose the answer into atomic factual claims using the LLM.
      2. For each claim, ask the LLM: "Is this claim supported by the context?"
      3. Score = |supported claims| / |total claims|.

    A score of 1.0 means fully grounded; 0.0 means fully hallucinated.
    """
    client = _get_client()
    joined_context = "\n---\n".join(context_chunks)

    # Step 1: Extract atomic claims from the answer
    extract_response = await client.acompletion(
        action_name="faithfulness_extract_claims",
        messages=[
            {"role": "system", "content": "Extract all atomic factual claims from the given answer. Each claim should be a single, verifiable statement of fact."},
            {"role": "user", "content": f"Answer to decompose:\n{answer}"},
        ],
        response_format=ClaimExtractionResult,
        temperature=config.STRUCTURED_TEMPERATURE,
    )
    claims_result = ClaimExtractionResult.model_validate_json(
        extract_response.choices[0].message.content
    )
    claims = claims_result.claims

    if not claims:
        return {"score": 1.0, "total_claims": 0, "supported_claims": 0, "details": []}

    # Step 2: Verify each claim against the context
    verify_response = await client.acompletion(
        action_name="faithfulness_verify_claims",
        messages=[
            {"role": "system", "content": (
                "You are a fact-checker. For each claim below, determine if it is 'supported' or 'unsupported' "
                "by the provided context. A claim is 'supported' ONLY if the context explicitly contains the information. "
                "If the context does not mention it, the claim is 'unsupported'."
            )},
            {"role": "user", "content": (
                f"Context:\n{joined_context}\n\n"
                f"Claims to verify:\n" + "\n".join(f"- {c}" for c in claims)
            )},
        ],
        response_format=ClaimVerificationBatch,
        temperature=config.STRUCTURED_TEMPERATURE,
    )
    verdicts = ClaimVerificationBatch.model_validate_json(
        verify_response.choices[0].message.content
    )

    supported = sum(1 for v in verdicts.verdicts if v.verdict.lower() == "supported")
    score = supported / len(claims)

    return {
        "score": round(score, 4),
        "total_claims": len(claims),
        "supported_claims": supported,
        "details": [v.model_dump() for v in verdicts.verdicts],
    }


# ===================================================================
# RAGAS METRIC 2: ANSWER RELEVANCY
# ===================================================================

async def evaluate_answer_relevancy(
    query: str,
    answer: str,
    n_questions: int = 3,
) -> dict[str, Any]:
    """
    Evaluates whether the generated answer is relevant to the original query.

    Algorithm:
      1. Ask the LLM to generate N synthetic "reverse questions" that the answer would address.
      2. Embed the original query and each reverse question.
      3. Compute mean cosine similarity between original query embedding and reverse question embeddings.

    High similarity = the answer addresses the query's intent.
    Low similarity  = the answer is off-topic or tangential.
    """
    client = _get_client()

    # Step 1: Generate reverse questions
    reverse_response = await client.acompletion(
        action_name="relevancy_reverse_questions",
        messages=[
            {"role": "system", "content": (
                f"Given the following answer, generate exactly {n_questions} different questions that this answer "
                "would be a good response to. The questions should capture the core intent of the answer."
            )},
            {"role": "user", "content": f"Answer:\n{answer}"},
        ],
        response_format=ReverseQuestions,
        temperature=0.4,
    )
    reverse_result = ReverseQuestions.model_validate_json(
        reverse_response.choices[0].message.content
    )
    reverse_questions = reverse_result.questions[:n_questions]

    # Step 2: Embed original query + reverse questions
    all_texts = [query] + reverse_questions
    embeddings = await get_embeddings_batch(all_texts)

    query_emb = np.array(embeddings[0])
    reverse_embs = [np.array(e) for e in embeddings[1:]]

    # Step 3: Compute cosine similarities
    similarities = []
    for rev_emb in reverse_embs:
        cos_sim = float(np.dot(query_emb, rev_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(rev_emb)))
        similarities.append(cos_sim)

    mean_score = float(np.mean(similarities))

    return {
        "score": round(mean_score, 4),
        "reverse_questions": reverse_questions,
        "similarities": [round(s, 4) for s in similarities],
    }


# ===================================================================
# RAGAS METRIC 3: CONTEXTUAL PRECISION
# ===================================================================

async def evaluate_contextual_precision(
    query: str,
    context_chunks: list[str],
    ground_truth: str,
) -> dict[str, Any]:
    """
    Evaluates whether the retriever ranked relevant chunks above irrelevant ones.

    Algorithm:
      1. For each retrieved chunk, ask the LLM: "Is this chunk relevant to answering the query,
         given that the correct answer contains this ground truth?"
      2. Compute Contextual Precision@K = sum(Precision@k * v_k) / total_relevant,
         where v_k = 1 if chunk at rank k is relevant.

    A score of 1.0 means all relevant chunks were ranked at the top.
    A low score means noise is ranked above signal.
    """
    client = _get_client()

    numbered_chunks = "\n".join(
        f"[Chunk {i}]: {chunk}" for i, chunk in enumerate(context_chunks)
    )

    # Step 1: Get relevance verdicts for each chunk
    relevance_response = await client.acompletion(
        action_name="contextual_precision_relevance",
        messages=[
            {"role": "system", "content": (
                "You are a retrieval quality evaluator. For each retrieved chunk, determine if it contains "
                "information that is useful for answering the user's query, given the ground-truth answer. "
                "Return a verdict for every chunk."
            )},
            {"role": "user", "content": (
                f"User Query: {query}\n\n"
                f"Ground-Truth Answer: {ground_truth}\n\n"
                f"Retrieved Chunks:\n{numbered_chunks}"
            )},
        ],
        response_format=ChunkRelevanceBatch,
        temperature=config.STRUCTURED_TEMPERATURE,
    )
    verdicts = ChunkRelevanceBatch.model_validate_json(
        relevance_response.choices[0].message.content
    )

    # Step 2: Compute Contextual Precision@K
    # Sort verdicts by chunk_index to respect retrieval rank order
    sorted_verdicts = sorted(verdicts.verdicts, key=lambda v: v.chunk_index)
    relevance_flags = [v.relevant for v in sorted_verdicts]

    total_relevant = sum(relevance_flags)
    if total_relevant == 0:
        return {"score": 0.0, "relevance_flags": relevance_flags, "details": [v.model_dump() for v in sorted_verdicts]}

    # Precision@k for each position, weighted by whether that position is relevant
    weighted_precision_sum = 0.0
    relevant_so_far = 0
    for k, is_relevant in enumerate(relevance_flags):
        if is_relevant:
            relevant_so_far += 1
            precision_at_k = relevant_so_far / (k + 1)
            weighted_precision_sum += precision_at_k

    score = weighted_precision_sum / total_relevant

    return {
        "score": round(score, 4),
        "relevance_flags": relevance_flags,
        "total_relevant": total_relevant,
        "details": [v.model_dump() for v in sorted_verdicts],
    }


# ===================================================================
# CONVENIENCE: RUN ALL RAGAS METRICS
# ===================================================================

async def run_ragas_evaluation(
    query: str,
    context_chunks: list[str],
    answer: str,
    ground_truth: str | None = None,
) -> dict[str, Any]:
    """
    Runs all available RAGAS metrics and returns a consolidated report.

    - Faithfulness and Answer Relevancy always run.
    - Contextual Precision only runs if ground_truth is provided.
    """
    faithfulness = await evaluate_faithfulness(query, context_chunks, answer)
    relevancy = await evaluate_answer_relevancy(query, answer)

    report = {
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
    }

    if ground_truth:
        precision = await evaluate_contextual_precision(query, context_chunks, ground_truth)
        report["contextual_precision"] = precision

    return report


# ===================================================================
# LLM-AS-A-JUDGE: POINTWISE SCORING
# ===================================================================

async def pointwise_judge(
    query: str,
    answer: str,
    rubric_dimensions: list[str] | None = None,
) -> dict[str, Any]:
    """
    Scores a single answer on a 1–5 rubric across multiple dimensions using CoT reasoning.

    Default dimensions: Faithfulness, Relevance, Clarity, Completeness.
    The LLM must provide chain-of-thought reasoning before assigning each score.
    """
    if rubric_dimensions is None:
        rubric_dimensions = ["Faithfulness", "Relevance", "Clarity", "Completeness"]

    client = _get_client()

    dimensions_str = ", ".join(rubric_dimensions)
    response = await client.acompletion(
        action_name="pointwise_judge",
        messages=[
            {"role": "system", "content": (
                "You are an expert evaluator. Score the given answer on each of these dimensions: "
                f"{dimensions_str}. "
                "For each dimension, provide:\n"
                "1. Chain-of-thought reasoning explaining your assessment\n"
                "2. A score from 1 (very poor) to 5 (excellent)\n\n"
                "Then provide an overall_score as the arithmetic mean of all dimension scores."
            )},
            {"role": "user", "content": f"Question: {query}\n\nAnswer to evaluate:\n{answer}"},
        ],
        response_format=PointwiseJudgement,
        temperature=config.STRUCTURED_TEMPERATURE,
    )

    judgement = PointwiseJudgement.model_validate_json(
        response.choices[0].message.content
    )

    return {
        "scores": [s.model_dump() for s in judgement.scores],
        "overall_score": round(judgement.overall_score, 2),
    }


# ===================================================================
# LLM-AS-A-JUDGE: PAIRWISE COMPARISON WITH POSITION-BIAS MITIGATION
# ===================================================================

async def pairwise_judge(
    query: str,
    answer_a: str,
    answer_b: str,
) -> dict[str, Any]:
    """
    Compares two answers head-to-head with position-bias mitigation.

    Algorithm:
      1. Run comparison in order (A first, B second) → get verdict_1.
      2. Run comparison in swapped order (B first, A second) → get verdict_2.
      3. If both consistently prefer the same candidate → declare winner.
         If the model flips preference when order is swapped → flag position bias.

    This double-swap technique is the standard approach to detect and mitigate
    the known positional bias in LLM evaluators (tendency to prefer the first option).
    """
    client = _get_client()

    system_prompt = (
        "You are an impartial evaluator. Compare the two answers below for the given question. "
        "Decide which answer is better overall (considering accuracy, relevance, clarity, and completeness). "
        "Reply with winner='A' if Answer A is better, winner='B' if Answer B is better, or winner='tie' if equal. "
        "Provide reasoning before your verdict."
    )

    # Pass 1: A first, B second
    response_ab = await client.acompletion(
        action_name="pairwise_judge_AB",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Question: {query}\n\n"
                f"Answer A:\n{answer_a}\n\n"
                f"Answer B:\n{answer_b}"
            )},
        ],
        response_format=PairwiseVerdict,
        temperature=config.STRUCTURED_TEMPERATURE,
    )
    verdict_ab = PairwiseVerdict.model_validate_json(
        response_ab.choices[0].message.content
    )

    # Pass 2: B first, A second (swapped labels)
    response_ba = await client.acompletion(
        action_name="pairwise_judge_BA",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Question: {query}\n\n"
                f"Answer A:\n{answer_b}\n\n"
                f"Answer B:\n{answer_a}"
            )},
        ],
        response_format=PairwiseVerdict,
        temperature=config.STRUCTURED_TEMPERATURE,
    )
    verdict_ba = PairwiseVerdict.model_validate_json(
        response_ba.choices[0].message.content
    )

    # Normalize the swapped verdict: if BA says "A" wins, that's actually our original B
    swapped_winner_map = {"A": "B", "B": "A", "tie": "tie"}
    normalized_ba_winner = swapped_winner_map.get(verdict_ba.winner.upper(), "tie")

    # Determine final result
    pass1 = verdict_ab.winner.upper()
    pass2 = normalized_ba_winner.upper()

    if pass1 == pass2:
        final_winner = pass1
        position_bias_detected = False
    elif pass1 == "TIE" or pass2 == "TIE":
        final_winner = "tie"
        position_bias_detected = False
    else:
        # Model flipped preference when order changed → position bias
        final_winner = "tie (position bias detected)"
        position_bias_detected = True

    return {
        "final_winner": final_winner,
        "position_bias_detected": position_bias_detected,
        "pass_1_AB": {"winner": verdict_ab.winner, "reasoning": verdict_ab.reasoning},
        "pass_2_BA_raw": {"winner": verdict_ba.winner, "reasoning": verdict_ba.reasoning},
        "pass_2_BA_normalized": normalized_ba_winner,
    }
