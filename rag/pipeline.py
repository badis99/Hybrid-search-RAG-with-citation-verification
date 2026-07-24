"""Phase 8 — one entry point wiring every stage, with a full debug trace."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.config import Config
from rag.fuse import HybridIndex, reciprocal_rank_fusion
from rag.generate import generate
from rag.ingest import Chunk, chunk_documents, load_corpus
from rag.dense import build_dense_index
from rag.rerank import Reranker
from rag.sparse import build_sparse_index
from rag.verify import LLMJudge, NLIVerifier, Verification, verify_answer


@dataclass
class PipelineResult:
    query: str
    answer: str
    abstained: bool
    faithfulness: float
    verified_claims: list[Verification] = field(default_factory=list)
    unsupported_claims: list[Verification] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """Every stage stays a swappable function; this only orchestrates them.

    The debug dict is the point: when an answer is wrong you need to see which
    arrow of the pipeline broke — was the chunk retrieved at all, did it survive
    fusion, did rerank keep it, did the model cite it, did verification agree?
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.chunks: list[Chunk] = []
        self.by_id: dict[str, Chunk] = {}
        self.hybrid: HybridIndex | None = None
        self.reranker: Reranker | None = None
        self.verifier = None

    def build(self) -> "Pipeline":
        """Offline work: chunk once, index once, load models once."""
        cfg = self.config
        self.chunks = chunk_documents(
            load_corpus(), size=cfg.chunk_size_tokens, overlap=cfg.chunk_overlap_tokens
        )
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.hybrid = HybridIndex(
            build_dense_index(self.chunks), build_sparse_index(self.chunks)
        )
        self.reranker = Reranker()
        self.verifier = (
            NLIVerifier(threshold=cfg.verify_threshold)
            if cfg.verifier == "nli"
            else LLMJudge(model=cfg.llm_model)
        )
        return self

    def answer(self, query: str) -> PipelineResult:
        cfg = self.config
        debug: dict[str, Any] = {"config": vars(cfg).copy()}

        # 1/2/3 — retrieve with each retriever independently
        dense_hits = self.hybrid.dense.search(query, cfg.k_each)
        sparse_hits = self.hybrid.sparse.search(query, cfg.k_each)
        debug["dense"] = [(cid, round(s, 3)) for cid, s in dense_hits[:5]]
        debug["sparse"] = [(cid, round(s, 3)) for cid, s in sparse_hits[:5]]

        # 4 — fuse by rank
        fused = reciprocal_rank_fusion([dense_hits, sparse_hits], k=cfg.rrf_k)
        fused = fused[: cfg.fusion_pool_size]
        debug["fused_pool_size"] = len(fused)
        debug["fused"] = [(cid, round(s, 4)) for cid, s in fused[:5]]

        # 5 — rerank the pool, keep the top few
        candidates = [self.by_id[cid] for cid, _ in fused]
        reranked = self.reranker.rerank(query, candidates, top_n=cfg.rerank_top_n)
        debug["reranked"] = [(cid, round(s, 2)) for cid, s in reranked]

        # 6 — generate a grounded answer over exactly those chunks
        context_chunks = [self.by_id[cid] for cid, _ in reranked]
        generated = generate(query, context_chunks, model=cfg.llm_model)
        debug["claims"] = [(c.claim, c.cited_chunk_ids) for c in generated.claims]
        debug["invalid_citations"] = generated.invalid_citations
        debug["abstained"] = generated.abstained

        # 7 — verify each claim against its own cited chunk
        verified = verify_answer(generated, self.by_id, self.verifier)
        debug["verifier"] = self.verifier.method
        debug["verdicts"] = [
            (v.claim[:60], v.verdict, round(v.score, 2))
            for v in verified.verified + verified.unsupported
        ]

        return PipelineResult(
            query=query,
            answer=generated.answer_text,
            abstained=generated.abstained,
            faithfulness=verified.overall_faithfulness,
            verified_claims=verified.verified,
            unsupported_claims=verified.unsupported,
            debug=debug,
        )


def _trace(result: PipelineResult) -> None:
    d = result.debug
    print(f"\n{'=' * 78}\nQ: {result.query}")
    print(f"  1-3 dense   : {[c for c, _ in d['dense'][:3]]}")
    print(f"      sparse  : {[c for c, _ in d['sparse'][:3]]}")
    print(f"  4   fused   : {[c for c, _ in d['fused'][:3]]}  (pool={d['fused_pool_size']})")
    print(f"  5   reranked: {[c for c, _ in d['reranked'][:3]]}")
    print(f"  6   answer  : {result.answer[:96]}")
    print(f"      abstained={d['abstained']}  invalid_citations={d['invalid_citations']}")
    for claim, cites in d["claims"]:
        print(f"        - {claim[:66]}")
        print(f"          cites {cites}")
    print(f"  7   verify ({d['verifier']}): faithfulness={result.faithfulness:.2f}")
    for claim, verdict, score in d["verdicts"]:
        mark = "OK " if verdict == "supported" else "FLAG"
        print(f"        [{mark}] {verdict:16} {score:.2f}  {claim}")


if __name__ == "__main__":
    pipeline = Pipeline().build()
    print(f"built: {len(pipeline.chunks)} chunks indexed")

    queries = [
        "Who is the all-time top scorer in World Cup history?",          # in corpus
        "Which side triumphed at the very first tournament on home soil?",  # paraphrase
        "catenaccio",                                                     # exact term
        "Which loss is remembered as a national catastrophe?",            # paraphrase
        "How many World Cups has Brazil won?",                            # simple fact
        "Which country won the 2026 World Cup?",                          # out of corpus
    ]
    for query in queries:
        _trace(pipeline.answer(query))
