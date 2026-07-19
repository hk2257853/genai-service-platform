import logging
from typing import List
import litellm

import config
from day02.client import handle_litellm_exceptions_async

logger = logging.getLogger("embeddings_utils")


def get_embedding_model() -> str:
    """Selects a matching embedding model based on the configured MODEL_ID provider."""
    base_model = config.MODEL_ID.lower()
    if "gemini" in base_model:
        # Google's standard embedding model (3072 dimensions)
        return "gemini/gemini-embedding-001"
    elif "openai" in base_model:
        # OpenAI's standard embedding model (1536 dimensions)
        return "openai/text-embedding-3-small"
    else:
        # Default fallback, standard for general setups
        return "openai/text-embedding-3-small"


async def get_embedding(text: str) -> List[float]:
    """Generates a vector embedding for a single text input."""
    model = get_embedding_model()
    async with handle_litellm_exceptions_async("embeddings_single"):
        response = await litellm.aembedding(model=model, input=[text])
        return response["data"][0]["embedding"]


async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generates vector embeddings for a list of text inputs in a single batch call."""
    if not texts:
        return []
    model = get_embedding_model()
    async with handle_litellm_exceptions_async("embeddings_batch"):
        response = await litellm.aembedding(model=model, input=texts)
        # Reconstruct list order mapping to input
        return [item["embedding"] for item in response["data"]]
