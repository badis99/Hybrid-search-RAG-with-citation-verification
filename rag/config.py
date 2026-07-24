"""Every tunable knob in the pipeline, grouped by the phase that introduced it."""
from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

# Shared bi-encoder: loaded once here so dense indexing and query embedding always
# use the SAME model (mixing them would put the vectors in different spaces).
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


@dataclass
class Config:
    # Phase 1 — chunking
    chunk_size_tokens: int = 300
    chunk_overlap_tokens: int = 45

    # Phase 2/3 — retrieval breadth, per retriever
    k_each: int = 10

    # Phase 4 — fusion
    rrf_k: int = 60
    fusion_pool_size: int = 20

    # Phase 5 — reranking
    rerank_top_n: int = 5

    # Phase 6 — generation
    llm_model: str = "openai/gpt-oss-120b"

    # Phase 7 — verification. "nli" is local, free and deterministic;
    # "llm-judge" is more paraphrase-tolerant but spends API calls.
    verifier: str = "nli"
    verify_threshold: float = 0.5
