from typing import List, Optional
from pydantic import BaseModel, Field


# Metadata and source models
class SourceMetadata(BaseModel):
    document_name: str
    snippet: str
    distance: float
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


# API Request schemas
class IngestRequest(BaseModel):
    content: str = Field(..., min_length=5, description="Document raw text content or absolute local file path.")
    document_name: str = Field(..., min_length=2, description="Identifying name of the document.")
    method: str = Field("semantic", description="Chunking strategy: 'semantic' (recommended) or 'fixed'.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The RAG query / question to answer.")
    limit: int = Field(3, ge=1, le=10, description="Max source context documents to retrieve.")
    use_query_expansion: bool = Field(False, description="Expand user query into alternative paraphrases using LLM.")
    use_hybrid: bool = Field(True, description="Combine dense vector search with sparse BM25 keyword search via RRF.")
    use_reranker: bool = Field(True, description="Apply Cross-Encoder neural reranking on retrieved candidate chunks.")


# API Response schemas
class IngestResponse(BaseModel):
    document_name: str
    chunk_count: int
    method_used: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceMetadata]
    expanded_queries: Optional[List[str]] = None
    retrieval_method: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


class SearchChunkResult(BaseModel):
    document_name: str
    text_content: str
    snippet: str
    distance: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    expanded_queries: List[str]
    retrieved_count: int
    chunks: List[SearchChunkResult]
