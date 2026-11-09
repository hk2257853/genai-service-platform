"""Token-Per-Minute (TPM) Sliding-Window Rate Limiter.

Protects upstream LLM provider quotas against token exhaustion using a two-phase
reservation and settlement protocol across a sliding 60-second time window.
"""

from __future__ import annotations

import time
import uuid
import logging
from dataclasses import dataclass
from typing import Any

from ai_service_platform import config
from ai_service_platform.core.exceptions import LLMTPMRateLimitError

logger = logging.getLogger("tpm_limiter")


@dataclass
class TokenReservation:
    """Represents an active token reservation in the sliding ledger."""
    id: str
    timestamp: float
    reserved_tokens: int
    actual_tokens: int | None = None


class TPMRateLimiter:
    """Sliding-window token rate limiter with pre-call reservation and post-call settlement."""

    def __init__(self, max_tpm: int = config.TPM_LIMIT, window_seconds: float = 60.0):
        self.max_tpm = max_tpm
        self.window_seconds = window_seconds
        self._ledger: list[TokenReservation] = []

    def _prune(self, now: float | None = None) -> None:
        """Removes ledger entries older than window_seconds."""
        cutoff = (now or time.time()) - self.window_seconds
        self._ledger = [entry for entry in self._ledger if entry.timestamp > cutoff]

    @property
    def current_tokens_in_window(self) -> int:
        """Calculates total tokens committed in the current sliding window."""
        self._prune()
        total = 0
        for entry in self._ledger:
            total += entry.actual_tokens if entry.actual_tokens is not None else entry.reserved_tokens
        return total

    def estimate_tokens(self, prompt_or_messages: str | list[dict[str, Any]], expected_output_tokens: int = 300) -> int:
        """Estimates prompt token count using standard character heuristics plus buffer."""
        if isinstance(prompt_or_messages, str):
            text_len = len(prompt_or_messages)
        elif isinstance(prompt_or_messages, list):
            text_len = sum(len(str(m.get("content", ""))) for m in prompt_or_messages if isinstance(m, dict))
        else:
            text_len = len(str(prompt_or_messages))

        # Standard rule of thumb: ~4 characters per token
        prompt_tokens = max(1, text_len // 4)
        return prompt_tokens + expected_output_tokens

    def acquire(self, estimated_tokens: int) -> str:
        """Attempts to reserve tokens in the sliding window.

        Raises LLMTPMRateLimitError if reservation would exceed max_tpm.
        Returns reservation_id on success.
        """
        now = time.time()
        self._prune(now)

        current_usage = sum(
            entry.actual_tokens if entry.actual_tokens is not None else entry.reserved_tokens
            for entry in self._ledger
        )

        projected_usage = current_usage + estimated_tokens
        if projected_usage > self.max_tpm:
            # Calculate how many seconds until oldest entries expire to free required capacity
            needed_reduction = projected_usage - self.max_tpm
            freed = 0
            retry_after = self.window_seconds

            for entry in self._ledger:
                entry_tokens = entry.actual_tokens if entry.actual_tokens is not None else entry.reserved_tokens
                freed += entry_tokens
                if freed >= needed_reduction:
                    retry_after = max(1.0, (entry.timestamp + self.window_seconds) - now)
                    break

            msg = (
                f"Token-Per-Minute limit exceeded: Current usage {current_usage} + "
                f"estimated {estimated_tokens} = {projected_usage} tokens > limit of {self.max_tpm}. "
                f"Retry after {retry_after:.1f}s."
            )
            logger.warning("[TPMLimiter] REJECTED: %s", msg)
            raise LLMTPMRateLimitError(message=msg, retry_after_seconds=round(retry_after, 2))

        reservation_id = uuid.uuid4().hex
        self._ledger.append(
            TokenReservation(
                id=reservation_id,
                timestamp=now,
                reserved_tokens=estimated_tokens,
                actual_tokens=None,
            )
        )
        logger.debug(
            "[TPMLimiter] ACQUIRED reservation %s (%d tokens) | Window Total: %d / %d TPM",
            reservation_id[:8],
            estimated_tokens,
            current_usage + estimated_tokens,
            self.max_tpm,
        )
        return reservation_id

    def settle(self, reservation_id: str, actual_tokens: int) -> None:
        """Settles an active reservation with actual prompt + completion tokens."""
        for entry in self._ledger:
            if entry.id == reservation_id:
                diff = actual_tokens - entry.reserved_tokens
                entry.actual_tokens = actual_tokens
                logger.debug(
                    "[TPMLimiter] SETTLED reservation %s (reserved=%d -> actual=%d, diff=%+d)",
                    reservation_id[:8],
                    entry.reserved_tokens,
                    actual_tokens,
                    diff,
                )
                return

        # If reservation expired from window before settlement, record it fresh
        self._ledger.append(
            TokenReservation(
                id=reservation_id,
                timestamp=time.time(),
                reserved_tokens=actual_tokens,
                actual_tokens=actual_tokens,
            )
        )

    def release(self, reservation_id: str) -> None:
        """Releases a reservation without consuming tokens (e.g. on semantic cache hit or error)."""
        self._ledger = [entry for entry in self._ledger if entry.id != reservation_id]
        logger.debug("[TPMLimiter] RELEASED reservation %s", reservation_id[:8])

    def get_status(self) -> dict[str, Any]:
        """Returns current TPM window consumption and capacity metrics."""
        self._prune()
        current = self.current_tokens_in_window
        return {
            "current_window_tokens": current,
            "max_tpm": self.max_tpm,
            "remaining_tokens": max(0, self.max_tpm - current),
            "utilization_pct": round((current / self.max_tpm) * 100, 2) if self.max_tpm > 0 else 0.0,
            "active_window_seconds": self.window_seconds,
            "window_entries_count": len(self._ledger),
        }

    def reset(self) -> None:
        """Clears the ledger (useful for clean slate testing)."""
        self._ledger.clear()
        logger.info("[TPMLimiter] Ledger reset")


# Global singleton instance
tpm_limiter = TPMRateLimiter()

