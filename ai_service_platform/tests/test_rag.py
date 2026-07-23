import asyncio
import os
import logging

from ai_service_platform.rag.chunking import fixed_size_chunking, semantic_chunking
from ai_service_platform.rag.embeddings import get_embeddings_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_rag_local")


async def main():
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    doc_path = os.path.join(fixtures_dir, "sample_doc.txt")
    if not os.path.exists(doc_path):
        logger.error("Could not find sample_doc.txt at %s.", doc_path)
        return

    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("=" * 60)
    print("COMPARATIVE CHUNKING ANALYSIS")
    print("=" * 60)
    print(f"Original text length: {len(text)} characters\n")

    # 1. Run Fixed-Size Chunking
    fixed_chunks = fixed_size_chunking(text, chunk_size=500, chunk_overlap=100)
    print(f"--- 1. FIXED-SIZE CHUNKING (Size: 500 chars, Overlap: 100 chars) ---")
    print(f"Total Chunks Generated: {len(fixed_chunks)}")
    for i, chunk in enumerate(fixed_chunks):
        print(f"\nChunk {i+1} (Length: {len(chunk)} chars):")
        print(f"[{chunk[:100]}... {chunk[-80:]}]")
    print("-" * 60)

    # 2. Run Semantic Chunking
    print(f"\n--- 2. SEMANTIC CHUNKING (Cosine Distance Threshold: 0.25) ---")
    semantic_chunks = await semantic_chunking(text, embed_fn=get_embeddings_batch, distance_threshold=0.25)
    print(f"Total Chunks Generated: {len(semantic_chunks)}")
    for i, chunk in enumerate(semantic_chunks):
        print(f"\nChunk {i+1} (Length: {len(chunk)} chars):")
        print(f"[{chunk[:100]}... {chunk[-80:]}]")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
