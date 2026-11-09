"""Semantic Cache for LLM Intent Reuse.

Computes cosine similarity over dense vector embeddings of incoming prompts.
If the similarity of a new prompt to an existing cached entry exceeds the
configured threshold (default: 0.92) within the same namespace, the cached
generation response is returned instantly without invoking the upstream LLM.
"""

from __future__ import annotations

import time
import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
import numpy as np

import litellm
from ai_service_platform import config

logger = logging.getLogger("semantic_cache")


@dataclass
class CacheEntry:
    """Represents a cached prompt-response pair with normalized vector embedding."""
    prompt: str
    response: Any
    metadata: dict[str, Any]
    embedding: np.ndarray
    created_at: float
    last_accessed: float


@dataclass
class CacheHit:
    """Encapsulates a successful semantic cache match."""
    response: Any
    similarity: float
    cached_prompt: str
    metadata: dict[str, Any]
    latency_saved_ms: float = 0.0


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalizes a 1D vector so cosine similarity equals the dot product."""
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm


class SemanticCache:
    """Production in-memory vector cache with LRU eviction and namespace isolation."""

    def __init__(
        self,
        threshold: float = config.SEMANTIC_CACHE_THRESHOLD,
        ttl_seconds: int = config.SEMANTIC_CACHE_TTL_SECONDS,
        max_entries: int = config.SEMANTIC_CACHE_MAX_ENTRIES,
        enabled: bool = config.SEMANTIC_CACHE_ENABLED,
        embedder: Callable[[str], Awaitable[np.ndarray | list[float]]] | None = None,
    ):
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.enabled = enabled
        self._custom_embedder = embedder

        # Namespace -> OrderedDict of {entry_id: CacheEntry} (maintains LRU order)
        self._cache: dict[str, OrderedDict[str, CacheEntry]] = defaultdict(OrderedDict)

        # Operational telemetry counters
        self._stats = {
            "hits": 0,
            "misses": 0,
            "saved_tokens": 0,
            "saved_cost_usd": 0.0,
        }

    async def _embed(self, text: str) -> np.ndarray:
        """Generates an L2-normalized dense vector for the given text."""
        if self._custom_embedder is not None:
            raw = await self._custom_embedder(text)
            arr = np.asarray(raw, dtype=np.float32)
            return _l2_normalize(arr)

        # Default: Use LiteLLM embedding API matching current model provider
        model = "gemini/gemini-embedding-001" if "gemini" in config.MODEL_ID.lower() else "openai/text-embedding-3-small"
        response = await litellm.aembedding(model=model, input=[text])
        raw_vec = response["data"][0]["embedding"]
        arr = np.asarray(raw_vec, dtype=np.float32)
        return _l2_normalize(arr)

    def prune_expired(self, namespace: str | None = None) -> int:
        """Removes entries whose TTL has elapsed. Returns the count of purged entries."""
        now = time.time()
        purged = 0
        namespaces = [namespace] if namespace else list(self._cache.keys())

        for ns in namespaces:
            bucket = self._cache.get(ns)
            if not bucket:
                continue
            expired_keys = [
                k for k, entry in bucket.items()
                if (now - entry.created_at) > self.ttl_seconds
            ]
            for k in expired_keys:
                del bucket[k]
                purged += 1

        if purged > 0:
            logger.debug("[SemanticCache] Pruned %d expired entries", purged)
        return purged

    async def lookup(
        self,
        prompt: str,
        namespace: str = "default",
        min_threshold: float | None = None,
        embedding: np.ndarray | None = None,
    ) -> CacheHit | None:
        """Looks up a semantically similar prompt within the given namespace.

        Returns CacheHit if max cosine similarity >= threshold, else None.
        """
        if not self.enabled:
            return None

        bucket = self._cache.get(namespace)
        if not bucket:
            self._stats["misses"] += 1
            return None

        # Clean expired items
        self.prune_expired(namespace)
        if not bucket:
            self._stats["misses"] += 1
            return None

        threshold = min_threshold if min_threshold is not None else self.threshold

        # Embed incoming query if not pre-computed
        if embedding is None:
            query_vec = await self._embed(prompt)
        else:
            query_vec = _l2_normalize(np.asarray(embedding, dtype=np.float32))

        # Vectorized dot product against all entries in this namespace
        entry_keys = list(bucket.keys())
        matrix = np.vstack([bucket[k].embedding for k in entry_keys])
        similarities = np.dot(matrix, query_vec)

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        best_key = entry_keys[best_idx]

        if best_sim >= threshold:
            entry = bucket[best_key]
            # Update LRU ordering
            bucket.move_to_end(best_key)
            entry.last_accessed = time.time()

            # Record telemetry
            self._stats["hits"] += 1
            tokens_saved = int(entry.metadata.get("total_tokens", 0))
            cost_saved = float(entry.metadata.get("cost", 0.0))
            self._stats["saved_tokens"] += tokens_saved
            self._stats["saved_cost_usd"] += cost_saved

            logger.info(
                "[SemanticCache] HIT (sim=%.4f >= %.2f) in ns '%s' | Prompt: '%s' ~= '%s'",
                best_sim,
                threshold,
                namespace,
                prompt[:40],
                entry.prompt[:40],
            )
            return CacheHit(
                response=entry.response,
                similarity=best_sim,
                cached_prompt=entry.prompt,
                metadata=entry.metadata,
                latency_saved_ms=float(entry.metadata.get("latency_ms", 0.0)),
            )

        self._stats["misses"] += 1
        logger.debug(
            "[SemanticCache] MISS (best_sim=%.4f < %.2f) in ns '%s' for '%s'",
            best_sim,
            threshold,
            namespace,
            prompt[:40],
        )
        return None

    async def store(
        self,
        prompt: str,
        response: Any,
        namespace: str = "default",
        metadata: dict[str, Any] | None = None,
        embedding: np.ndarray | None = None,
    ) -> None:
        """Stores a new prompt-response pair with normalized vector embedding."""
        if not self.enabled:
            return

        bucket = self._cache[namespace]

        # Enforce LRU capacity limit per namespace
        if len(bucket) >= self.max_entries:
            oldest_key, _ = bucket.popitem(last=False)
            logger.debug("[SemanticCache] Evicted LRU entry '%s' from ns '%s'", oldest_key, namespace)

        if embedding is None:
            vec = await self._embed(prompt)
        else:
            vec = _l2_normalize(np.asarray(embedding, dtype=np.float32))

        now = time.time()
        entry_id = f"{namespace}:{now}:{hash(prompt)}"
        bucket[entry_id] = CacheEntry(
            prompt=prompt,
            response=response,
            metadata=metadata or {},
            embedding=vec,
            created_at=now,
            last_accessed=now,
        )
        logger.debug("[SemanticCache] Stored entry in ns '%s' (Total in ns: %d)", namespace, len(bucket))

    def get_stats(self) -> dict[str, Any]:
        """Returns cumulative operational stats and health metrics."""
        hits = self._stats["hits"]
        misses = self._stats["misses"]
        total = hits + misses
        hit_ratio = round(hits / total, 4) if total > 0 else 0.0

        total_entries = sum(len(b) for b in self._cache.values())

        return {
            "hits": hits,
            "misses": misses,
            "total_lookups": total,
            "hit_ratio": hit_ratio,
            "saved_tokens": self._stats["saved_tokens"],
            "saved_cost_usd": round(self._stats["saved_cost_usd"], 6),
            "total_cached_entries": total_entries,
            "threshold": self.threshold,
            "ttl_seconds": self.ttl_seconds,
            "max_entries": self.max_entries,
            "enabled": self.enabled,
        }

    def clear(self) -> None:
        """Clears all cached entries and resets statistics."""
        self._cache.clear()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "saved_tokens": 0,
            "saved_cost_usd": 0.0,
        }
        logger.info("[SemanticCache] Cache cleared and stats reset")


# Global singleton instance for gateway execution
semantic_cache = SemanticCache()

