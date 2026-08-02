import re
import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

logger = logging.getLogger("bm25_index")


def tokenize_text(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric words for BM25 keyword matching."""
    return re.findall(r'\w+', text.lower())


class BM25Index:
    """In-memory BM25 Okapi index for keyword/code retrieval."""

    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Indexes or appends document chunks into the BM25 model."""
        if not chunks:
            return

        self.chunks.extend(chunks)
        tokenized_corpus = [tokenize_text(chunk["text_content"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index updated. Total indexed documents: %d", len(self.chunks))

    def clear(self):
        """Clears the index."""
        self.chunks = []
        self.bm25 = None

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Queries the BM25 index and returns top matching chunks with scores."""
        if not self.bm25 or not self.chunks:
            logger.warning("BM25 search called on an empty index.")
            return []

        tokenized_query = tokenize_text(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        
        # Combine chunks with scores and sort descending
        scored_results = []
        for chunk, score in zip(self.chunks, scores):
            if score > 0:  # Only return chunks with non-zero keyword match
                scored_results.append({
                    "document_name": chunk["document_name"],
                    "text_content": chunk["text_content"],
                    "score": float(score)
                })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:limit]


def reciprocal_rank_fusion(
    dense_rankings: List[List[Dict[str, Any]]],
    sparse_rankings: List[List[Dict[str, Any]]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Merges multiple ranked retrieval lists (dense & sparse across query variations)
    using Reciprocal Rank Fusion (RRF). RRF Score = sum(1 / (k + rank)).
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    all_rankings = dense_rankings + sparse_rankings

    for ranking in all_rankings:
        for rank, item in enumerate(ranking, start=1):
            key = f"{item['document_name']}:::{item['text_content']}"
            chunk_map[key] = item
            
            # Reciprocal rank addition
            rrf_score = 1.0 / (k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + rrf_score

    # Sort chunks by final aggregated RRF score descending
    fused_results = []
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    for key in sorted_keys:
        item = chunk_map[key].copy()
        item["rrf_score"] = rrf_scores[key]
        fused_results.append(item)

    return fused_results
