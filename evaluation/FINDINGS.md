# Phase 9 — Evaluation findings

Measured on 30 labelled queries (8 paraphrase, 11 exact-term, 11 factual) against
a 21-chunk corpus, plus 20 labelled `(claim, chunk, supported?)` triples.

## Retrieval ablation

| Config | R@1 | R@3 | R@5 | MRR | nDCG@5 | P@3 |
| --- | --- | --- | --- | --- | --- | --- |
| Dense only | 0.567 | 0.667 | 0.800 | 0.668 | 0.672 | 0.233 |
| Sparse only (BM25) | **0.900** | 0.933 | 0.967 | **0.923** | **0.931** | 0.344 |
| Hybrid (RRF) | 0.733 | 0.933 | 0.967 | 0.832 | 0.860 | 0.333 |
| Hybrid + rerank | 0.800 | **0.967** | 0.967 | 0.872 | 0.892 | 0.344 |

Recall@10 is deliberately **not** reported: with 21 chunks, retrieving 10 covers
half the corpus and every config scores ~1.0. It measures nothing at this scale.
R@1 and MRR are the metrics with signal here.

### MRR by query category

| Category | Dense only | Sparse only | Hybrid (RRF) |
| --- | --- | --- | --- |
| Paraphrase (n=8) | 0.781 | 0.795 | **0.802** |
| Exact-term (n=11) | 0.558 | **1.000** | 0.776 |
| Factual (n=11) | 0.697 | **0.939** | 0.909 |

## What the numbers say

**The expected story did not hold.** Hybrid was supposed to beat both single
methods; instead **sparse-only won outright** (MRR 0.923 vs hybrid's 0.832). This
is not a fusion bug — RRF behaves correctly on the paraphrase category, the one it
exists for, where it beats both parents (0.802 vs 0.795/0.781). The problem is
**the eval set**, and the category breakdown shows exactly why.

Eleven of thirty queries are bare rare tokens (`catenaccio`, `Mineirazo`,
`Tofiq Bahramov`, `Silvio Gazzaniga`). BM25 scores a **perfect 1.000 MRR** on
them — they are its ideal input. Most "factual" queries also carry proper nouns
(`Brazil`, `1998`, `Maradona`) that BM25 anchors on, giving it 0.939 there too. So
22 of 30 queries live on BM25's home turf, and only 8 are genuinely
paraphrase-shaped.

That exposes a real property of RRF worth internalising: **it is a democratic
average, so it cannot beat a dominant expert on that expert's home ground.** When
BM25 ranks the gold chunk first and dense ranks it fifth, fusion mixes a perfect
ranking with a mediocre one and lands in between. Hybrid's value is *robustness
across query types*, not winning every type — and a benchmark weighted 22:8
toward one type will never show that.

**Reranking earned its place.** It improved every metric over hybrid (R@1
0.733 → 0.800, MRR 0.832 → 0.872, nDCG@5 0.860 → 0.892), recovering roughly half
of what fusion diluted by re-reading each `(query, chunk)` pair jointly. It still
did not overtake sparse-only, for the same eval-set reason.

**Dense retrieval is the weakest single method here** (MRR 0.668) — unsurprising
given the corpus is dense with proper nouns, dates, and scorelines, which is
precisely what embeddings blur.

## Verifier evaluation

Positive class = "supported"; 20 claims (10 supported, 10 not).

| Verifier | Precision | Recall | F1 | Accuracy | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NLI (deberta-v3) | **1.000** | 0.600 | 0.750 | 0.800 | 6 | 0 | 10 | 4 |
| LLM-judge | **1.000** | 1.000 | 1.000 | 1.000 | 10 | 0 | 10 | 0 |

**Both have perfect precision — zero false positives.** For this project that is
the number that matters: a false positive is an unsupported claim reaching the
user, which is the failure the whole system exists to prevent. Neither verifier
ever waved one through, and both caught every contradiction and mis-citation.

They differ entirely in **recall**. NLI missed 4 of 10 supported claims — it errs
strict, flagging correct answers. Cheap and deterministic, but it would generate
review noise in production.

The judge's flawless 20/20 should be read with suspicion, not celebration: 20
claims is a tiny set, and `gpt-oss-120b` is judging output from the same model
that generated it, so self-agreement inflates the score. The NLI path's value is
that it is genuinely independent.

## What I would do next

1. **Rebalance the eval set** toward paraphrase and multi-hop queries, which are
   under-represented at 8/30. Do this on a **held-out** split — rebalancing until
   hybrid wins on the same numbers used to judge it is tuning on the test set.
2. **Grow both labelled sets.** 30 queries and 20 claims are small enough that a
   couple of items move every metric materially.
3. **Investigate NLI's 4 false negatives.** Sentence-level scoring fixed 2 of 4 in
   a Phase 7 probe but made a third worse (it splits facts that span sentences), so
   it needs measuring at scale before adoption, not a patch.
4. **Get an independent judge.** Using a different model family from the generator
   would remove the self-agreement confound.
