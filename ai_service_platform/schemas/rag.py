from typing import List
from pydantic import BaseModel, Field


# Metadata and source models
class SourceMetadata(BaseModel):
    document_name: str
    snippet: str
    distance: float


# API Request schemas
class IngestRequest(BaseModel):
    content: str = Field(..., min_length=5, description="Document raw text content or absolute local file path.")
    document_name: str = Field(..., min_length=2, description="Identifying name of the document.")
    method: str = Field("semantic", description="Chunking strategy: 'semantic' (recommended) or 'fixed'.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The RAG query / question to answer.")
    limit: int = Field(3, ge=1, le=10, description="Max source context documents to retrieve.")


# API Response schemas
class IngestResponse(BaseModel):
    document_name: str
    chunk_count: int
    method_used: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceMetadata]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
