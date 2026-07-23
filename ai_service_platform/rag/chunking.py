import re
import numpy as np
from typing import Callable, Awaitable, List

def split_sentences(text: str) -> List[str]:
    """Splits a document text into sentences using regex heuristics."""
    sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
    sentences = sentence_end.split(text)
    return [s.strip() for s in sentences if s.strip()]


def fixed_size_chunking(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """Splits text into chunks of chunk_size characters with chunk_overlap, respecting word boundaries."""
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        if end < text_len:
            max_backtrack = int(chunk_size * 0.2)
            backtrack_limit = max(start, end - max_backtrack)
            
            while end > backtrack_limit and text[end] != ' ':
                end -= 1
            
            if end == backtrack_limit:
                end = min(start + chunk_size, text_len)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == text_len:
            break

        start = end - chunk_overlap
        if start >= end:
            start = end
        if start < 0:
            start = 0

    return chunks


def cosine_distance(v1: List[float], v2: List[float]) -> float:
    """Calculates the cosine distance (1 - cosine similarity) between two vectors."""
    a = np.array(v1)
    b = np.array(v2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1.0 - (np.dot(a, b) / (norm_a * norm_b)))


async def semantic_chunking(
    text: str, 
    embed_fn: Callable[[List[str]], Awaitable[List[List[float]]]], 
    distance_threshold: float = 0.25
) -> List[str]:
    """Groups sentences into semantically cohesive chunks by analyzing similarity shifts between neighbors."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    embeddings = await embed_fn(sentences)

    distances = []
    for i in range(len(embeddings) - 1):
        dist = cosine_distance(embeddings[i], embeddings[i+1])
        distances.append(dist)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(len(distances)):
        next_sentence = sentences[i + 1]
        dist = distances[i]

        if dist > distance_threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [next_sentence]
        else:
            current_chunk.append(next_sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
