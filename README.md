# Hybrid-Search RAG with Citation Verification

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-learning%20build-orange)

A from-scratch retrieval-augmented generation (RAG) system that combines **dense
vector search** and **BM25 keyword search**, reranks the results with a
**cross-encoder**, and then runs a **citation-verification pass** that checks
every claim in the generated answer against the source it cites.

That last step is the point of the project. Most RAG demos stop at "retrieve,
then generate" and quietly trust that the model's citations are real. This one
treats a citation as a claim to be verified, not a decoration — the quality
layer that separates a demo from something you'd actually rely on.

> This is a learning build. It is designed to be implemented one phase at a time,
> understanding each layer before moving to the next. The repository ships with a
> working environment, a corpus loader, a sample dataset, and typed stubs for
> every remaining component. See the [Build roadmap](#build-roadmap).

---

## Table of contents

- [Why this project](#why-this-project)
- [Architecture](#architecture)
- [Key concepts](#key-concepts)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Build roadmap](#build-roadmap)
- [Evaluation](#evaluation)
- [Tech stack and model choices](#tech-stack-and-model-choices)
- [Extensions](#extensions)
- [References](#references)
- [License](#license)

---

## Why this project

A RAG system can fail in exactly two ways:

1. **Retrieval failure** — the right context was never fetched. Nothing
   downstream can fix this.
2. **Generation failure** — the right context *was* fetched, but the model
   produced something wrong or unsupported by it.

This project attacks both. Hybrid search plus cross-encoder reranking maximize
the odds of retrieving the right passages (attacking failure 1). The
citation-verification pass catches the model when it states something the
retrieved text does not actually support (attacking failure 2).

## Architecture

Indexing happens once, offline. Everything from **Query** downward runs per
request against the pre-built indexes.

```mermaid
flowchart TD
    D[Documents] --> C[Chunking]
    C --> V[(Vector index<br/>dense embeddings)]
    C --> B[(BM25 index<br/>sparse keywords)]

    Q([Query]) --> V
    Q --> B
    V --> F{{RRF fusion}}
    B --> F
    F --> R[Cross-encoder<br/>rerank]
    R --> G[LLM generation<br/>answer + citations]
    G --> VF[Citation verifier<br/>NLI / LLM-judge]
    VF --> A([Verified answer])
```

The flow, stage by stage:

| Stage | What it does | Why it's there |
| --- | --- | --- |
| Chunking | Splits documents into small, citable passages with stable IDs | Retrieval precision and fine-grained citations |
| Dense retrieval | Embeds query + chunks, finds nearest neighbours | Catches paraphrases and semantic matches |
| Sparse retrieval (BM25) | Keyword scoring over the same chunks | Catches exact terms: names, codes, rare words |
| RRF fusion | Merges the two ranked lists by rank, not raw score | Combines their complementary strengths |
| Cross-encoder rerank | Re-scores the fused pool jointly on (query, passage) | High-precision ordering of the final few |
| LLM generation | Answers from the reranked passages, citing by chunk | Grounded answer with a parseable claim→source map |
| Citation verifier | Checks each claim against its cited chunk (NLI / LLM-judge) | Flags or removes unsupported claims |

## Key concepts

**Hybrid search.** Dense (embedding) retrieval understands meaning but blurs
exact tokens; sparse (BM25) retrieval nails exact tokens but misses paraphrases.
They fail on different queries, so using both and merging beats either alone.

**Reciprocal Rank Fusion (RRF).** Dense scores (cosine, roughly 0–1) and BM25
scores (unbounded) are not comparable, so you cannot simply add them. RRF uses
only the *rank* of each result:

```
rrf_score(d) = Σ  1 / (k + rank_i(d))       # over each retriever i, with k ≈ 60
```

**Cross-encoder reranking.** A bi-encoder embeds query and document separately
(fast, but never sees them together). A cross-encoder feeds `(query, document)`
through the model jointly and scores their relevance directly — much more
accurate, but too slow to run over a whole corpus. So it runs only on the fused
candidate pool. This two-stage pattern (cheap wide recall → expensive precise
rerank) is standard in production RAG.

**Citation verification.** "Does chunk *X* support claim *Y*?" is a Natural
Language Inference (NLI) problem: with premise = cited chunk and hypothesis =
claim, classify entailment / neutral / contradiction. Two interchangeable
implementations are included as a design choice to compare:

- an **NLI cross-encoder** — fast, cheap, deterministic entailment scores;
- an **LLM-as-judge** — flexible on paraphrase, but needs a strict rubric to
  avoid rating loosely-related text as "supported".

Compound sentences are decomposed into atomic claims first, and each claim is
checked against *its cited chunk* — which is what catches a citation that points
to the wrong source.

## Project structure

```
hybrid-rag-citation-verification/
├── README.md
├── requirements.txt
├── pyproject.toml            # makes `rag` / `evaluation` importable (pip install -e .)
├── .gitignore
├── .env.example              # copy to .env for API keys (Phase 6+)
├── data/
│   └── corpus/               # your source documents (.md / .txt / .json)
│       └── *.md              # sample coffee-brewing docs ship here
├── rag/
│   ├── config.py             # every tunable knob, grouped by phase
│   ├── ingest.py             # load_corpus() [done] + chunk_documents() [Phase 1]
│   ├── dense.py              # DenseIndex               [Phase 2]
│   ├── sparse.py             # SparseIndex (BM25)       [Phase 3]
│   ├── fuse.py               # reciprocal_rank_fusion   [Phase 4]
│   ├── rerank.py             # Reranker (cross-encoder) [Phase 5]
│   ├── generate.py           # generate() + Claim model [Phase 6]
│   ├── verify.py             # citation verification    [Phase 7]
│   └── pipeline.py           # Pipeline.answer()        [Phase 8]
├── evaluation/
│   └── metrics.py            # recall@k, MRR, nDCG      [Phase 9]
└── scripts/
    └── check_setup.py        # Phase 0 checkpoint
```

Every stage is a small module with a clean input→output interface, so stages are
independently testable and swappable. `load_corpus()` is fully implemented;
everything downstream is a typed stub that raises `NotImplementedError` with a
docstring describing what to build and the concepts to learn first.

## Getting started

### Prerequisites

- Python 3.11 or newer
- (Phase 6+) an API key for an LLM provider, or a local model via Ollama

### Install

```bash
git clone https://github.com/<you>/hybrid-rag-citation-verification.git
cd hybrid-rag-citation-verification

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Phase 0 needs nothing installed. For Phase 1+:
pip install -r requirements.txt
pip install -e .                   # optional but recommended: makes `import rag` work anywhere
```

### Run the Phase 0 checkpoint

This uses only the standard library, so you can run it before installing
anything:

```bash
python scripts/check_setup.py
```

Expected output: it loads the sample corpus, prints a preview of a few
documents, confirms document IDs are deterministic, and reports the checkpoint
as passed.

### Use your own corpus

Drop `.md`, `.txt`, or `.json` files into `data/corpus/` and re-run the check.
For `.json`, provide either a single `{"text": ..., "id": ..., "source": ...}`
object or a list of them. Start small — 20 to 200 documents on a topic you know
well, so you can tell when an answer is wrong.

## Usage

Once the pipeline is implemented (Phase 8), the intended entry point is:

```python
from rag.pipeline import Pipeline

# Build the indexes once (chunk -> embed + BM25).
pipeline = Pipeline().build()

result = pipeline.answer("What makes cold brew less acidic than hot coffee?")

print(result.answer_text)
for claim in result.verified_claims:
    print(f"  [supported {claim.score:.2f}] {claim.claim}")
for claim in result.unsupported_claims:
    print(f"  [UNSUPPORTED] {claim.claim}")

# result.debug holds every stage's intermediate output for inspection.
```

## Configuration

All knobs live in [`rag/config.py`](rag/config.py). Change a value, re-run,
compare — that is the experiment loop.

| Setting | Default | Phase | Purpose |
| --- | --- | --- | --- |
| `chunk_size_tokens` | 300 | 1 | Target chunk length |
| `chunk_overlap_tokens` | 45 | 1 | Overlap between chunks (~15%) |
| `embedding_model` | `all-MiniLM-L6-v2` | 2 | Bi-encoder for dense retrieval |
| `dense_top_k` | 50 | 2 | Dense candidates retrieved |
| `sparse_top_k` | 50 | 3 | BM25 candidates retrieved |
| `rrf_k` | 60 | 4 | RRF constant |
| `fusion_pool_size` | 100 | 4 | Candidates handed to the reranker |
| `reranker_model` | `ms-marco-MiniLM-L-6-v2` | 5 | Cross-encoder reranker |
| `rerank_top_n` | 5 | 5 | Passages sent to the LLM |
| `llm_model` | (set your own) | 6 | Generation model |
| `nli_model` | `nli-deberta-v3-base` | 7 | NLI model for verification |
| `verify_threshold` | 0.5 | 7 | Min entailment score to accept a claim |

## Build roadmap

Each phase depends on the last. For every phase: learn the concepts, implement
the stub, then run its checkpoint before moving on.

- [x] **Phase 0 — Setup.** Environment, project skeleton, corpus loader.
- [ ] **Phase 1 — Chunking.** Recursive splitting, overlap, stable chunk IDs.
- [ ] **Phase 2 — Dense retrieval.** Embeddings, cosine similarity, ANN.
- [ ] **Phase 3 — Sparse retrieval.** BM25, tokenization, exact-match strength.
- [ ] **Phase 4 — Hybrid fusion.** Reciprocal Rank Fusion.
- [ ] **Phase 5 — Reranking.** Cross-encoder, the two-stage pattern.
- [ ] **Phase 6 — Generation.** Grounded answers with parseable citations.
- [ ] **Phase 7 — Verification.** NLI and LLM-judge; claim decomposition.
- [ ] **Phase 8 — Pipeline.** Wire it together with a debug trace.
- [ ] **Phase 9 — Evaluation.** Retrieval metrics, faithfulness, the ablation table.
- [ ] **Phase 10 — Iterate.** Query rewriting, parent-doc retrieval, context ordering.

## Evaluation

Evaluation has two halves: did retrieval fetch the right chunks, and did
generation use them faithfully.

**Retrieval** is measured against a small labelled set of `query → gold
chunk_id(s)` using Recall@k, MRR, and nDCG@k. The payoff is an ablation table
that proves each component earns its place:

| Config | Recall@10 | MRR | nDCG@10 |
| --- | --- | --- | --- |
| Dense only | | | |
| Sparse only (BM25) | | | |
| Hybrid (RRF) | | | |
| Hybrid + rerank | | | |

Expected story: hybrid ≥ either single method, and reranking lifts precision the
most. If it doesn't, that's a finding worth investigating.

**Generation** is measured with faithfulness (are claims grounded?), answer
relevance, and context precision/recall. [RAGAS](https://github.com/explodinggradients/ragas)
computes these reference-free by decomposing answers into claims and verifying
each against context — essentially an automated, at-scale version of the
verifier built by hand in Phase 7.

## Tech stack and model choices

Chosen to run on a laptop (CPU is fine) and to be simple enough to learn from.
Every choice is swappable — leaderboards move, the concepts don't, so always
test on your own data.

| Component | Starter | Upgrade path |
| --- | --- | --- |
| Embeddings | `all-MiniLM-L6-v2` | `bge-small-en-v1.5` → `bge-m3` |
| Vector index | numpy brute-force | `faiss-cpu`, `chromadb` |
| Sparse | `rank_bm25` | `bm25s` |
| Reranker | `ms-marco-MiniLM-L-6-v2` | `bge-reranker-v2-m3` |
| Generation | any LLM API / Ollama | — |
| NLI verifier | `nli-deberta-v3-base` | fact-verification models, LLM-judge |
| Evaluation | hand-rolled metrics | `ragas`, `deepeval` |

## Extensions

Once the core works, each of these is a self-contained follow-up: query
rewriting / multi-query / HyDE; parent-document or sentence-window retrieval
(retrieve small, feed larger context); context ordering to counter "lost in the
middle"; caching and streaming; and a small web UI.

## References

- Karpukhin et al., *Dense Passage Retrieval* (DPR)
- Robertson & Zaragoza, *The Probabilistic Relevance Framework* (BM25)
- Cormack et al., 2009, *Reciprocal Rank Fusion*
- The [sentence-transformers](https://www.sbert.net/) documentation (cross-encoders)
- Es et al., 2024, *RAGAS*; and the *ALCE* benchmark on citation quality
- Liu et al., *Lost in the Middle*

## License

MIT — see below. The sample documents in `data/corpus/` are original content
written for this repository and are released under the same license.
