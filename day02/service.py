import logging
from typing import AsyncIterator
from litellm import completion_cost
from pydantic import BaseModel, Field

import config
from day02.client import AsyncLLMClient

logger = logging.getLogger("llm_service")


class ProductSentimentAnalysis(BaseModel):
    """Pydantic schema for structured output sentiment analysis."""
    product_name: str = Field(description="The name of the product mentioned.")
    sentiment: str = Field(description="The sentiment of the user. Options: Positive, Negative, Neutral.")
    rating: int = Field(description="An estimated star rating from 1 to 5 based on the text.")
    pros: list[str] = Field(description="Key strengths/features liked by the customer.")
    cons: list[str] = Field(description="Key complaints or pain points mentioned.")
    requires_refund: bool = Field(description="True if the user is demanding a refund, else False.")


class LLMResponse(BaseModel):
    """Standardized response model for general generation requests."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


class SentimentAnalysisResponse(BaseModel):
    """Standardized response model for structured sentiment analysis."""
    analysis: ProductSentimentAnalysis
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


class LLMService:
    """Service layer executing business logic, output validation, and cost logging."""

    def __init__(self, client: AsyncLLMClient | None = None):
        self.client = client or AsyncLLMClient(default_model=config.MODEL_ID)

    async def generate_text(self, prompt: str) -> LLMResponse:
        """Asynchronously generates text and tracks cost and usage."""
        response = await self.client.acompletion(
            action_name="service_generate",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        content = response.choices[0].message.content or ""
        usage = response.usage
        cost = completion_cost(completion_response=response) or 0.0

        return LLMResponse(
            content=content,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost=cost
        )

    async def analyze_sentiment(self, feedback_text: str) -> SentimentAnalysisResponse:
        """Asynchronously analyzes feedback text and validates against schema."""
        response = await self.client.acompletion(
            action_name="service_sentiment",
            messages=[{"role": "user", "content": f"Analyze this customer feedback:\n\n{feedback_text}"}],
            response_format=ProductSentimentAnalysis,
            temperature=config.STRUCTURED_TEMPERATURE
        )

        raw_json_str = response.choices[0].message.content or ""
        
        # Pydantic parsing of returned schema
        try:
            analysis = ProductSentimentAnalysis.model_validate_json(raw_json_str)
        except Exception as e:
            logger.error("Failed to validate JSON response schema: %s. Raw content: %s", e, raw_json_str)
            raise ValueError(f"Model returned invalid schema format: {e}") from e

        usage = response.usage
        cost = completion_cost(completion_response=response) or 0.0

        return SentimentAnalysisResponse(
            analysis=analysis,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost=cost
        )

    async def stream_text(self, prompt: str) -> AsyncIterator[str]:
        """Asynchronously streams generated text, capturing usage metadata in final chunk if present."""
        stream_generator = self.client.acompletion_stream(
            action_name="service_stream",
            messages=[{"role": "user", "content": prompt}],
            stream_options={"include_usage": True}
        )

        async for chunk in stream_generator:
            # Check for token usage metrics
            if hasattr(chunk, "usage") and chunk.usage is not None:
                usage = chunk.usage
                cost = completion_cost(completion_response=chunk) or 0.0
                logger.info(
                    "[Stream Finished] Prompt: %d, Completion: %d, Cost: $%.8f",
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    cost
                )
            elif len(chunk.choices) > 0:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
