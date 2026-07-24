"""Phase 9 — retrieval metrics, the ablation table, and verifier precision/recall."""
from __future__ import annotations

import math
from collections import defaultdict

from evaluation.eval_set import EVAL_QUERIES
from rag.dense import build_dense_index
from rag.fuse import reciprocal_rank_fusion
from rag.ingest import chunk_documents, load_corpus
from rag.rerank import Reranker
from rag.sparse import build_sparse_index
from rag.verify import LABELLED_CLAIMS, LLMJudge, NLIVerifier


# --------------------------------------------------------------------------
# metrics — `ranked` is a list of chunk_ids, best first; `gold` is a set
# --------------------------------------------------------------------------
def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """Did any gold chunk make the top k? (binary per query, averaged later)"""
    return 1.0 if set(ranked[:k]) & gold else 0.0


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(ranked[:k]) & gold) / k


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    """1/rank of the FIRST gold hit — rewards putting it at the very top."""
    for i, chunk_id in enumerate(ranked, start=1):
        if chunk_id in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """Binary-relevance nDCG: like recall but discounted by position."""
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, chunk_id in enumerate(ranked[:k], start=1)
        if chunk_id in gold
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


# --------------------------------------------------------------------------
# retrieval ablation
# --------------------------------------------------------------------------
def evaluate(rankings: dict[str, list[str]], gold_by_query: dict[str, set[str]]) -> dict:
    n = len(rankings)
    agg = defaultdict(float)
    for query, ranked in rankings.items():
        gold = gold_by_query[query]
        agg["R@1"] += recall_at_k(ranked, gold, 1)
        agg["R@3"] += recall_at_k(ranked, gold, 3)
        agg["R@5"] += recall_at_k(ranked, gold, 5)
        agg["MRR"] += reciprocal_rank(ranked, gold)
        agg["nDCG@5"] += ndcg_at_k(ranked, gold, 5)
        agg["P@3"] += precision_at_k(ranked, gold, 3)
    return {metric: value / n for metric, value in agg.items()}


def run_ablation(k_each: int = 10, rrf_k: int = 60, rerank_top_n: int = 5) -> dict:
    chunks = chunk_documents(load_corpus())
    by_id = {c.chunk_id: c for c in chunks}
    dense = build_dense_index(chunks)
    sparse = build_sparse_index(chunks)
    reranker = Reranker()

    gold_by_query = {q: set(g) for q, g, _ in EVAL_QUERIES}
    configs: dict[str, dict[str, list[str]]] = {
        "dense only": {}, "sparse only": {}, "hybrid (RRF)": {}, "hybrid + rerank": {},
    }

    for query, _gold, _cat in EVAL_QUERIES:
        dense_hits = dense.search(query, k_each)
        sparse_hits = sparse.search(query, k_each)
        fused = reciprocal_rank_fusion([dense_hits, sparse_hits], k=rrf_k)

        configs["dense only"][query] = [cid for cid, _ in dense_hits]
        configs["sparse only"][query] = [cid for cid, _ in sparse_hits]
        configs["hybrid (RRF)"][query] = [cid for cid, _ in fused]

        pool = [by_id[cid] for cid, _ in fused]
        reranked = reranker.rerank(query, pool, top_n=rerank_top_n)
        configs["hybrid + rerank"][query] = [cid for cid, _ in reranked]

    return {name: evaluate(r, gold_by_query) for name, r in configs.items()}


def per_category(k_each: int = 10, rrf_k: int = 60) -> dict:
    """Where each retriever wins — the reason the eval set is categorised."""
    chunks = chunk_documents(load_corpus())
    dense = build_dense_index(chunks)
    sparse = build_sparse_index(chunks)
    out: dict[str, dict[str, float]] = defaultdict(dict)
    by_cat: dict[str, list] = defaultdict(list)
    for query, gold, cat in EVAL_QUERIES:
        by_cat[cat].append((query, set(gold)))

    for cat, items in by_cat.items():
        for name in ("dense only", "sparse only", "hybrid (RRF)"):
            total = 0.0
            for query, gold in items:
                d = dense.search(query, k_each)
                s = sparse.search(query, k_each)
                if name == "dense only":
                    ranked = [cid for cid, _ in d]
                elif name == "sparse only":
                    ranked = [cid for cid, _ in s]
                else:
                    ranked = [cid for cid, _ in reciprocal_rank_fusion([d, s], k=rrf_k)]
                total += reciprocal_rank(ranked, gold)
            out[cat][name] = total / len(items)
    return out


# --------------------------------------------------------------------------
# verifier evaluation
# --------------------------------------------------------------------------
def evaluate_verifier(verifier, by_id) -> dict:
    """Precision/recall treating "supported" as the positive class.

    Precision = of the claims it passed, how many deserved to pass. That is the
    number that matters here: a false positive is an unsupported claim reaching
    the user, which is exactly what this project exists to prevent.
    """
    tp = fp = tn = fn = 0
    for claim, chunk_id, expected in LABELLED_CLAIMS:
        got = verifier.verify(claim, by_id[chunk_id].text).is_supported
        if got and expected:
            tp += 1
        elif got and not expected:
            fp += 1
        elif not got and not expected:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": (tp + tn) / len(LABELLED_CLAIMS),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _table(title: str, rows: dict, columns: list[str]) -> None:
    print(f"\n{title}")
    print("  " + "config".ljust(18) + "".join(c.rjust(9) for c in columns))
    print("  " + "-" * (18 + 9 * len(columns)))
    for name, scores in rows.items():
        print("  " + name.ljust(18) + "".join(f"{scores[c]:9.3f}" for c in columns))


if __name__ == "__main__":
    print(f"eval set: {len(EVAL_QUERIES)} labelled queries")

    ablation = run_ablation()
    _table("RETRIEVAL ABLATION", ablation, ["R@1", "R@3", "R@5", "MRR", "nDCG@5", "P@3"])

    print("\nMRR BY QUERY CATEGORY")
    cats = per_category()
    names = ["dense only", "sparse only", "hybrid (RRF)"]
    print("  " + "category".ljust(14) + "".join(n.rjust(15) for n in names))
    print("  " + "-" * (14 + 15 * len(names)))
    for cat, scores in cats.items():
        print("  " + cat.ljust(14) + "".join(f"{scores[n]:15.3f}" for n in names))

    chunks = chunk_documents(load_corpus())
    by_id = {c.chunk_id: c for c in chunks}
    print(f"\nVERIFIER EVAL ({len(LABELLED_CLAIMS)} labelled claims)")
    verifier_rows = {}
    for verifier in (NLIVerifier(), LLMJudge()):
        verifier_rows[verifier.method] = evaluate_verifier(verifier, by_id)
    _table("", verifier_rows, ["precision", "recall", "f1", "accuracy"])
    for name, s in verifier_rows.items():
        print(f"  {name:10} tp={s['tp']} fp={s['fp']} tn={s['tn']} fn={s['fn']}")
