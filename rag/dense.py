"""Phase 2 — dense (semantic) retrieval with a numpy brute-force index."""
from __future__ import annotations

import numpy as np

from rag.config import model
from rag.ingest import Chunk, chunk_documents, load_corpus

# BGE retrieval models expect the *query* (not the passages) to carry this
# instruction prefix. Leaving it off measurably hurts recall for bge-*-en-v1.5.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseIndex:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.chunk_ids: list[str] = []
        self.matrix: np.ndarray | None = None

    def build(self, chunks: list[Chunk]) -> "DenseIndex":
        self.chunks = list(chunks)
        self.chunk_ids = [c.chunk_id for c in self.chunks]
        texts = [c.text for c in self.chunks]
        self.matrix = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        q = model.encode(
            QUERY_PREFIX + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        scores = self.matrix @ q
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top]


def build_dense_index(chunks: list[Chunk] | None = None) -> DenseIndex:
    if chunks is None:
        chunks = chunk_documents(load_corpus())
    return DenseIndex().build(chunks)


if __name__ == "__main__":
    index = build_dense_index()
    queries = [
        "Which nation lifted the trophy at the very first tournament, held in South America?",
        "Who has found the net more times than anyone across World Cup history?",
        "Which loss is remembered as a national catastrophe for the host country?",
        "Which player is the best from Argentina?"
    ]
    for query in queries:
        print(f"\nQ: {query}")
        for chunk_id, score in index.search(query, k=3):
            print(f"  {score:.3f}  {chunk_id}")
