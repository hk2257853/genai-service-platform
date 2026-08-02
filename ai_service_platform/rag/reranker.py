import logging
from typing import List, Dict, Any

logger = logging.getLogger("cross_encoder_reranker")

_reranker_model = None


def get_reranker():
    """Lazy loader for SentenceTransformer CrossEncoder model."""
    global _reranker_model
    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            logger.info("Loading Cross-Encoder reranker model: %s...", model_name)
            _reranker_model = CrossEncoder(model_name)
            logger.info("Cross-Encoder reranker model loaded successfully.")
        except Exception as e:
            logger.warning("Failed to load CrossEncoder model: %s. Reranker will run in fallback mode.", e)
            _reranker_model = "FALLBACK"
    return _reranker_model


class CrossEncoderReranker:
    """Reranks candidate retrieved chunks using a neural Cross-Encoder model."""

    def __init__(self):
        pass

    def rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Scores (query, chunk_text) pairs with the Cross-Encoder model 
        and returns the candidates sorted by relevance score descending.
        """
        if not candidates:
            return []

        model = get_reranker()

        if model == "FALLBACK" or model is None:
            logger.info("Reranker running in fallback mode (preserving candidate order).")
            # Preserve candidate order, assigning dummy scores
            results = []
            for idx, candidate in enumerate(candidates):
                item = candidate.copy()
                item["rerank_score"] = float(1.0 / (idx + 1))
                results.append(item)
            return results[:top_k]

        # Prepare (query, text) pairs for prediction
        pairs = [[query, candidate["text_content"]] for candidate in candidates]
        
        try:
            scores = model.predict(pairs)
            
            reranked = []
            for candidate, score in zip(candidates, scores):
                item = candidate.copy()
                item["rerank_score"] = float(score)
                reranked.append(item)

            # Sort descending by Cross-Encoder relevance score
            reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
            logger.info(
                "Cross-Encoder reranked %d candidates. Top score: %.4f, Bottom score: %.4f",
                len(candidates),
                reranked[0]["rerank_score"] if reranked else 0.0,
                reranked[-1]["rerank_score"] if reranked else 0.0
            )
            return reranked[:top_k]

        except Exception as e:
            logger.error("Error during Cross-Encoder prediction: %s", e)
            return candidates[:top_k]
