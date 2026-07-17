"""Phase 3 — sparse (lexical) retrieval with BM25."""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from rag.ingest import Chunk, chunk_documents, load_corpus

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word tokens. The SAME function must tokenize
    both the chunks and the query, or terms won't match."""
    return _TOKEN_RE.findall(text.lower())


class SparseIndex:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.chunk_ids: list[str] = []
        self.bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> "SparseIndex":
        self.chunks = list(chunks)
        self.chunk_ids = [c.chunk_id for c in self.chunks]
        tokenized_corpus = [tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top]


def build_sparse_index(chunks: list[Chunk] | None = None) -> SparseIndex:
    if chunks is None:
        chunks = chunk_documents(load_corpus())
    return SparseIndex().build(chunks)


if __name__ == "__main__":
    index = build_sparse_index()
    queries = [
        "catenaccio",                                                     # rare exact term
        "Mineirazo",                                                      # rare exact term
        "Tofiq Bahramov linesman",                                        # proper nouns
        "Which loss is remembered as a national catastrophe for the host country?",  # paraphrase
    ]
    for query in queries:
        print(f"\nQ: {query}")
        for chunk_id, score in index.search(query, k=3):
            print(f"  {score:.3f}  {chunk_id}")
