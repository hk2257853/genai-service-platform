import logging
from typing import List
from ai_service_platform.core.client import AsyncLLMClient

logger = logging.getLogger("query_expansion")

EXPANSION_SYSTEM_PROMPT = (
    "You are an AI assistant helping optimize document retrieval queries.\n"
    "Your goal is to generate alternative versions of the user's input query to capture "
    "different phrasing, technical terminology, and synonyms.\n\n"
    "Rules:\n"
    "1. Generate exactly {num_queries} alternative search query formulations.\n"
    "2. Each alternative should be concise and focused on capturing relevant document chunks.\n"
    "3. Return ONLY the alternative queries, one per line, without numbering, bullets, or commentary."
)


async def expand_query(
    query: str, 
    llm_client: AsyncLLMClient, 
    num_queries: int = 3
) -> List[str]:
    """Generates alternative search queries using the LLM to increase recall across diverse document terms."""
    if not query or not query.strip():
        return [query]

    logger.info("Expanding user query into %d variations: '%s'", num_queries, query)

    messages = [
        {"role": "system", "content": EXPANSION_SYSTEM_PROMPT.format(num_queries=num_queries)},
        {"role": "user", "content": f"User Query: {query}"}
    ]

    try:
        response = await llm_client.acompletion(
            action_name="query_expansion",
            messages=messages,
            temperature=0.7
        )

        content = response.choices[0].message.content or ""
        lines = [line.strip() for line in content.splitlines() if line.strip()]

        # Ensure the original query is included first, followed by unique expansions
        expanded = [query]
        for line in lines:
            # Clean off any residual numbering like "1. ", "- " if present
            cleaned = line.lstrip("0123456789.-* ").strip()
            if cleaned and cleaned not in expanded:
                expanded.append(cleaned)

        logger.info("Query expansion generated %d unique search queries.", len(expanded))
        return expanded
    except Exception as e:
        logger.warning("Query expansion failed, falling back to original query: %s", e)
        return [query]
