import re
import numpy as np
from typing import Callable, Awaitable, List

def split_sentences(text: str) -> List[str]:
    """Splits a document text into sentences using regex heuristics."""
    # Split by sentence end punctuation followed by whitespace, keeping abbreviations intact
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
        
        # If we are not at the end of the text, back up to the nearest space to avoid breaking words
        if end < text_len:
            # Backtrack up to 20% of the chunk size to find a space
            max_backtrack = int(chunk_size * 0.2)
            backtrack_limit = max(start, end - max_backtrack)
            
            while end > backtrack_limit and text[end] != ' ':
                end -= 1
            
            # If we couldn't find a space within the limit, reset to standard end
            if end == backtrack_limit:
                end = min(start + chunk_size, text_len)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == text_len:
            break

        # Advance start index
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
        return 1.0  # Max distance if one is a zero vector
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

    # 1. Generate embeddings in a single batch call to optimize API performance
    embeddings = await embed_fn(sentences)

    # 2. Compute cosine distance between adjacent sentences
    distances = []
    for i in range(len(embeddings) - 1):
        dist = cosine_distance(embeddings[i], embeddings[i+1])
        distances.append(dist)

    # 3. Group sentences into chunks based on distance spikes
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(len(distances)):
        next_sentence = sentences[i + 1]
        dist = distances[i]

        if dist > distance_threshold:
            # Semantic boundary triggered -> finalize current chunk and start a new one
            chunks.append(" ".join(current_chunk))
            current_chunk = [next_sentence]
        else:
            current_chunk.append(next_sentence)

    # Add the final remaining chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
