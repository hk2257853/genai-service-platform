import os
import asyncio
import logging
import concurrent.futures
from typing import List, Dict, Any, Optional

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

from ai_service_platform.rag.embeddings import get_embedding, get_embeddings_batch

logger = logging.getLogger("llamaindex_pipeline")


def sync_thread_run(coro_fn, *args):
    """Executes an async function in a dedicated worker thread, preserving the main ASGI asyncio loop."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: asyncio.run(coro_fn(*args)))
        return future.result()


class PlatformLlamaEmbedding(BaseEmbedding):
    """Custom LlamaIndex BaseEmbedding adapter wrapping our LiteLLM embedding functions."""
    
    embed_dim: int = 768

    @classmethod
    def class_name(cls) -> str:
        return "PlatformLlamaEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        return sync_thread_run(get_embedding, query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return sync_thread_run(get_embedding, text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return sync_thread_run(get_embeddings_batch, texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await get_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await get_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await get_embeddings_batch(texts)


class LlamaIndexPipeline:
    """Framework-first RAG pipeline using LlamaIndex for indexing, node chunking, and metadata filtering."""

    def __init__(self):
        self.documents: List[Document] = []
        self.index: Optional[VectorStoreIndex] = None
        self.splitter = SentenceSplitter(chunk_size=500, chunk_overlap=100)
        self.embed_model = PlatformLlamaEmbedding()
        Settings.embed_model = self.embed_model

    def ingest(self, content_or_path: str, document_name: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Parses document text, attaches metadata, splits into nodes, and updates LlamaIndex VectorStoreIndex."""
        text = content_or_path
        if os.path.exists(content_or_path) and os.path.isfile(content_or_path):
            with open(content_or_path, "r", encoding="utf-8") as f:
                text = f.read()

        doc_metadata = metadata.copy() if metadata else {}
        doc_metadata["document_name"] = document_name

        doc = Document(text=text, metadata=doc_metadata)
        self.documents.append(doc)

        nodes = self.splitter.get_nodes_from_documents([doc])
        logger.info("LlamaIndex created %d nodes for document '%s' with metadata: %s", 
                    len(nodes), document_name, doc_metadata)

        if self.index is None:
            self.index = VectorStoreIndex(nodes, embed_model=self.embed_model)
        else:
            self.index.insert_nodes(nodes)

        return len(nodes)

    def retrieve_with_filters(
        self, 
        query: str, 
        limit: int = 3, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves candidate nodes using LlamaIndex VectorIndexRetriever 
        with exact-match MetadataFilters.
        """
        if self.index is None:
            logger.warning("LlamaIndex pipeline search called on uninitialized index.")
            return []

        metadata_filters = None
        if filters:
            exact_filters = [
                ExactMatchFilter(key=k, value=v) 
                for k, v in filters.items()
            ]
            metadata_filters = MetadataFilters(filters=exact_filters)
            logger.info("Applied LlamaIndex MetadataFilters: %s", filters)

        retriever = self.index.as_retriever(
            similarity_top_k=limit,
            filters=metadata_filters
        )

        nodes_with_scores = retriever.retrieve(query)
        logger.info("LlamaIndex retrieved %d nodes for query: '%s'", len(nodes_with_scores), query)

        results = []
        for node_with_score in nodes_with_scores:
            node = node_with_score.node
            results.append({
                "document_name": node.metadata.get("document_name", "Unknown"),
                "text_content": node.get_content(),
                "score": float(node_with_score.score or 0.0),
                "metadata": node.metadata
            })

        return results
