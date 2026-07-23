"""Phase 5 — cross-encoder reranking, stage 2 of the two-stage pattern."""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from rag.fuse import HybridIndex, build_hybrid_index
from rag.ingest import Chunk

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[Chunk],
        top_n: int = 5,
    ) -> list[tuple[str, float]]:
        """Score every (query, chunk_text) pair jointly, then reorder.

        Unlike a bi-encoder, the model sees the query and the passage together
        in one forward pass, so it can judge their interaction directly. All
        pairs go into a single predict() call so the library batches them.
        Scores are raw relevance logits — use them to sort, not as probabilities.
        """
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [(c.chunk_id, float(s)) for c, s in ranked[:top_n]]


def search_and_rerank(
    hybrid: HybridIndex,
    reranker: Reranker,
    query: str,
    k_each: int = 10,
    pool: int = 20,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Full retrieval path: hybrid (recall) -> cross-encoder (precision).

    hybrid.search returns chunk_ids only, but the cross-encoder needs the
    chunk TEXT, so ids are resolved back to Chunk objects in between.
    """
    fused = hybrid.search(query, k_each=k_each, pool=pool)
    by_id = {c.chunk_id: c for c in hybrid.dense.chunks}
    candidates = [by_id[chunk_id] for chunk_id, _ in fused]
    return reranker.rerank(query, candidates, top_n=top_n)


if __name__ == "__main__":
    hybrid = build_hybrid_index()
    reranker = Reranker()
    queries = [
        "Who has found the net more times than anyone across World Cup history?",
        "catenaccio",
        "Which side triumphed at the very first tournament on home soil?",
    ]
    for query in queries:
        print(f"\nQ: {query}")
        before = hybrid.search(query, k_each=10, pool=20)
        print("  HYBRID  :")
        for rank, (chunk_id, _) in enumerate(before[:5], start=1):
            print(f"    {rank}. {chunk_id}")
        print("  RERANKED:")
        after = search_and_rerank(hybrid, reranker, query, top_n=5)
        for rank, (chunk_id, score) in enumerate(after, start=1):
            print(f"    {rank}. {chunk_id}  ({score:+.2f})")
