"""Cross-Provider Fallback Model Router with LiteLLM.

Provides multi-vendor resilience across LLM providers (e.g. Gemini -> OpenAI -> Anthropic)
with automated failover cascades, circuit breaker cooldowns, and operational routing telemetry.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, AsyncIterator
import litellm
from litellm import Router

from ai_service_platform import config
from ai_service_platform.core.exceptions import (
    AllProvidersUnavailableError,
    LLMRateLimitError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMAuthenticationError,
    LLMAPIError,
    LLMException,
)

logger = logging.getLogger("fallback_router")


class FallbackRouter:
    """Production LiteLLM router orchestrating cross-vendor fallback cascades and circuit cooldowns."""

    def __init__(
        self,
        primary_model: str = config.MODEL_ID,
        fallback_models: list[str] | None = None,
        allowed_fails: int = config.ROUTER_ALLOWED_FAILS,
        cooldown_time: int = config.ROUTER_COOLDOWN_TIME,
    ):
        self.primary_model = primary_model
        self.fallback_models = fallback_models if fallback_models is not None else config.FALLBACK_MODELS
        self.allowed_fails = allowed_fails
        self.cooldown_time = cooldown_time

        # Track manual test overrides and simulated cooldowns
        self._manual_cooldowns: dict[str, float] = {}  # model_name -> expiration_timestamp
        self._fail_counts: dict[str, int] = {}

        self.router = self._build_router()

    def _build_router(self) -> Router:
        """Constructs and configures the underlying LiteLLM Router instance."""
        model_list: list[dict[str, Any]] = [
            {
                "model_name": "primary_gateway",
                "litellm_params": {
                    "model": self.primary_model,
                },
            }
        ]

        fallback_names: list[str] = []
        for i, fb_model in enumerate(self.fallback_models):
            fb_group = f"fallback_gateway_{i + 1}"
            fallback_names.append(fb_group)
            model_list.append(
                {
                    "model_name": fb_group,
                    "litellm_params": {
                        "model": fb_model,
                    },
                }
            )

        fallbacks = [{"primary_gateway": fallback_names}] if fallback_names else []

        logger.info(
            "[FallbackRouter] Initialized router: Primary='%s', Fallbacks=%s, allowed_fails=%d, cooldown=%ds",
            self.primary_model,
            self.fallback_models,
            self.allowed_fails,
            self.cooldown_time,
        )

        return Router(
            model_list=model_list,
            fallbacks=fallbacks,
            allowed_fails=self.allowed_fails,
            cooldown_time=self.cooldown_time,
            retry_after=1,
            num_retries=1,
        )

    def trip_model(self, model_identifier: str, cooldown_seconds: int | None = None) -> None:
        """Manually trips a model deployment into cooldown (simulating circuit breaker OPEN state)."""
        duration = cooldown_seconds if cooldown_seconds is not None else self.cooldown_time
        expires_at = time.time() + duration
        self._manual_cooldowns[model_identifier] = expires_at
        logger.warning(
            "[FallbackRouter] CIRCUIT TRIPPED for '%s' for %ds (expires at %.1f)",
            model_identifier,
            duration,
            expires_at,
        )

    def reset_cooldowns(self) -> None:
        """Clears all manual and internal model cooldowns."""
        self._manual_cooldowns.clear()
        self._fail_counts.clear()
        # Re-build clean router instance
        self.router = self._build_router()
        logger.info("[FallbackRouter] All cooldowns and failure counters reset")

    def is_in_cooldown(self, model_identifier: str) -> bool:
        """Checks if a model is currently in cooldown (circuit breaker OPEN)."""
        now = time.time()
        expiry = self._manual_cooldowns.get(model_identifier)
        if expiry and now < expiry:
            return True
        if expiry and now >= expiry:
            del self._manual_cooldowns[model_identifier]
        return False

    def get_deployment_status(self) -> list[dict[str, Any]]:
        """Returns health, tier, and circuit status for each configured model deployment."""
        status_list: list[dict[str, Any]] = []

        # Primary deployment
        primary_cooldown = self.is_in_cooldown(self.primary_model)
        status_list.append({
            "model": self.primary_model,
            "role": "primary",
            "circuit_state": "OPEN" if primary_cooldown else "CLOSED",
            "cooldown_active": primary_cooldown,
            "fail_count": self._fail_counts.get(self.primary_model, 0),
        })

        # Fallback deployments
        for fb_model in self.fallback_models:
            fb_cooldown = self.is_in_cooldown(fb_model)
            status_list.append({
                "model": fb_model,
                "role": "fallback",
                "circuit_state": "OPEN" if fb_cooldown else "CLOSED",
                "cooldown_active": fb_cooldown,
                "fail_count": self._fail_counts.get(fb_model, 0),
            })

        return status_list

    async def acompletion(self, **kwargs) -> Any:
        """Executes completion with automatic fallback cascade and circuit awareness."""
        # Check if primary model is manually tripped into cooldown
        target_model_group = "primary_gateway"
        if self.is_in_cooldown(self.primary_model):
            # Fast-fail bypass: directly route to available fallback
            available_fallbacks = [
                (i, fb) for i, fb in enumerate(self.fallback_models)
                if not self.is_in_cooldown(fb)
            ]
            if available_fallbacks:
                idx, fb_model = available_fallbacks[0]
                target_model_group = f"fallback_gateway_{idx + 1}"
                logger.info(
                    "[FallbackRouter] Primary '%s' circuit is OPEN. Fast-failing directly to fallback '%s' (%s)",
                    self.primary_model,
                    fb_model,
                    target_model_group,
                )
            else:
                logger.error("[FallbackRouter] All primary and fallback models are in cooldown!")
                raise AllProvidersUnavailableError(
                    message="All LLM providers are currently in cooldown (circuits OPEN)",
                    tried_models=[self.primary_model] + self.fallback_models,
                )

        kwargs["model"] = target_model_group

        try:
            response = await self.router.acompletion(**kwargs)
            return response
        except litellm.exceptions.AuthenticationError as e:
            logger.error("[FallbackRouter] Authentication failed across providers: %s", e)
            raise LLMAuthenticationError(f"Authentication failed across providers: {e}", e)
        except litellm.exceptions.RateLimitError as e:
            self._fail_counts[self.primary_model] = self._fail_counts.get(self.primary_model, 0) + 1
            logger.warning("[FallbackRouter] RateLimitError: %s", e)
            raise LLMRateLimitError(f"Rate limit exceeded on all routed providers: {e}", e)
        except (litellm.exceptions.APIConnectionError, litellm.exceptions.ServiceUnavailableError) as e:
            self._fail_counts[self.primary_model] = self._fail_counts.get(self.primary_model, 0) + 1
            logger.warning("[FallbackRouter] Connection/ServiceUnavailable error: %s", e)
            raise LLMConnectionError(f"All routed providers unavailable: {e}", e)
        except litellm.exceptions.Timeout as e:
            self._fail_counts[self.primary_model] = self._fail_counts.get(self.primary_model, 0) + 1
            logger.warning("[FallbackRouter] Timeout error: %s", e)
            raise LLMTimeoutError(f"Timeout on routed providers: {e}", e)
        except Exception as e:
            err_str = str(e).lower()
            if "no fallback model group found" in err_str or "all fallbacks failed" in err_str:
                raise AllProvidersUnavailableError(
                    message=f"All configured providers in router cascade failed: {e}",
                    tried_models=[self.primary_model] + self.fallback_models,
                    original_exception=e,
                )
            raise LLMException(f"Router exception: {e}", original_exception=e)

    async def acompletion_stream(self, **kwargs) -> AsyncIterator[Any]:
        """Executes streaming completion with automatic fallback cascade."""
        kwargs["model"] = "primary_gateway"
        try:
            response_stream = await self.router.acompletion(**kwargs)
            return response_stream
        except Exception as e:
            raise LLMException(f"Router streaming exception: {e}", original_exception=e)


# Global singleton instance
fallback_router = FallbackRouter()

