from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from ai_service_platform.schemas.common import BaseLLMUsage
from ai_service_platform.schemas.rag import SourceMetadata


class LlamaIndexIngestRequest(BaseModel):
    content: str = Field(..., min_length=5, description="Document file path or raw text string.")
    document_name: str = Field(..., min_length=2, description="Identifying name of the document.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Structured key-value metadata tags for pre-retrieval filtering (e.g., {'category': 'database'})."
    )


class LlamaIndexAskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question / query for LlamaIndex RAG pipeline.")
    limit: int = Field(3, ge=1, le=10, description="Max candidate context nodes to retrieve.")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Key-value metadata criteria for exact-match pre-retrieval filtering (e.g., {'category': 'security'})."
    )


class LlamaIndexAskResponse(BaseLLMUsage):
    """Standardized response model for LlamaIndex RAG pipeline queries."""
    answer: str
    sources: List[SourceMetadata]
    filters_applied: Optional[Dict[str, Any]] = None
