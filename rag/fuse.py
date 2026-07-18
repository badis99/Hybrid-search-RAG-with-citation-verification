"""Phase 4 — hybrid fusion with Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

from collections import defaultdict

from rag.dense import DenseIndex, build_dense_index
from rag.sparse import SparseIndex, build_sparse_index
from rag.ingest import chunk_documents, load_corpus


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge several ranked lists by RANK, not raw score.

    rrf_score(d) = sum over each list of 1 / (k + rank), rank starting at 1.
    A chunk that appears in more than one list has its contributions summed,
    which both rewards agreement and deduplicates in a single step.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(ranked, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class HybridIndex:
    def __init__(self, dense: DenseIndex, sparse: SparseIndex):
        self.dense = dense
        self.sparse = sparse

    def search(
        self,
        query: str,
        k_each: int = 50,
        pool: int = 100,
        rrf_k: int = 60,
    ) -> list[tuple[str, float]]:
        dense_hits = self.dense.search(query, k_each)
        sparse_hits = self.sparse.search(query, k_each)
        fused = reciprocal_rank_fusion([dense_hits, sparse_hits], k=rrf_k)
        return fused[:pool]


def build_hybrid_index(chunks=None) -> HybridIndex:
    if chunks is None:
        chunks = chunk_documents(load_corpus())
    return HybridIndex(build_dense_index(chunks), build_sparse_index(chunks))


if __name__ == "__main__":
    index = build_hybrid_index()
    queries = [
        "catenaccio",                                                    # sparse-favoring
        "Which side triumphed at the very first tournament on home soil?",  # dense-favoring
        "Who scored a hat-trick in a World Cup final?",                  # mixed
    ]
    for query in queries:
        print(f"\nQ: {query}")
        print("  DENSE :", [c for c, _ in index.dense.search(query, 3)])
        print("  SPARSE:", [c for c, _ in index.sparse.search(query, 3)])
        print("  HYBRID:", [f"{c} {s:.4f}" for c, s in index.search(query, k_each=10, pool=3)])
