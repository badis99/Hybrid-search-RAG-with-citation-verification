"""Phase 6 — grounded generation with machine-parseable citations."""
from __future__ import annotations

from dataclasses import dataclass

import anthropic
from pydantic import BaseModel

from rag.ingest import Chunk

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """You answer questions strictly from a numbered set of context passages.

Rules:
1. Use ONLY the passages provided. Never use your own knowledge, even if you are
   confident the answer is correct. If the passages do not contain the answer,
   you MUST abstain.
2. To abstain: set answer_text to "I don't know based on the provided context."
   and return an EMPTY claims list. Abstaining is the correct, expected behaviour
   for questions the passages do not cover — it is never a failure.
3. Decompose your answer into ATOMIC claims: one single, independently checkable
   fact per claim. Never bundle two facts into one claim.
   Bad:  "Brazil has won five titles and is the only ever-present team."
   Good: "Brazil has won the World Cup five times."
         "Brazil is the only team to have appeared at every World Cup."
4. Every claim must cite the number(s) of the passage(s) that directly support it,
   in cited_chunks. Only cite a passage that actually states the fact. Never cite
   a passage number that was not provided to you.
5. answer_text is the prose answer for the reader. The claims list must cover the
   same facts, atomised."""


class Claim(BaseModel):
    claim: str
    cited_chunks: list[int]


class RawAnswer(BaseModel):
    """What the model returns — citations are passage NUMBERS, not chunk_ids."""
    answer_text: str
    claims: list[Claim]


@dataclass
class ResolvedClaim:
    claim: str
    cited_chunk_ids: list[str]


@dataclass
class Answer:
    answer_text: str
    claims: list[ResolvedClaim]
    abstained: bool
    invalid_citations: list[int]


def build_context(chunks: list[Chunk]) -> tuple[str, dict[int, str]]:
    """Number the passages [1]..[n] and keep the number -> chunk_id map.

    The model cites small integers rather than chunk_ids: a digit is a token span
    it cannot plausibly corrupt, while "world-cup-1950-maracanazo::0" is long and
    invites hallucination. We resolve the numbers back to real ids ourselves.
    """
    blocks, number_map = [], {}
    for number, chunk in enumerate(chunks, start=1):
        number_map[number] = chunk.chunk_id
        blocks.append(f"[{number}] {chunk.text}")
    return "\n\n".join(blocks), number_map


def resolve_citations(raw: RawAnswer, number_map: dict[int, str]) -> Answer:
    """Map cited passage numbers back to chunk_ids, rejecting any we never sent.

    Structured output constrains the SHAPE of the response, not its VALUES — the
    model can still emit [7] when it was given five passages. That is caught here.
    """
    claims: list[ResolvedClaim] = []
    invalid: list[int] = []
    for claim in raw.claims:
        chunk_ids = []
        for number in claim.cited_chunks:
            if number in number_map:
                chunk_ids.append(number_map[number])
            else:
                invalid.append(number)
        claims.append(ResolvedClaim(claim=claim.claim, cited_chunk_ids=chunk_ids))
    return Answer(
        answer_text=raw.answer_text,
        claims=claims,
        abstained=not raw.claims,
        invalid_citations=invalid,
    )


def generate(query: str, reranked_chunks: list[Chunk], model: str = MODEL) -> Answer:
    """Answer `query` grounded in `reranked_chunks`, with resolved citations.

    Uses structured outputs so the response is constrained to the RawAnswer schema
    — the "unparseable citations" failure mode is eliminated at the API level
    rather than defended against with regex.
    """
    context, number_map = build_context(reranked_chunks)
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context passages:\n\n{context}\n\nQuestion: {query}",
            }
        ],
        output_format=RawAnswer,
    )
    return resolve_citations(response.parsed_output, number_map)


if __name__ == "__main__":
    from rag.fuse import build_hybrid_index
    from rag.rerank import Reranker, search_and_rerank

    hybrid = build_hybrid_index()
    reranker = Reranker()
    by_id = {c.chunk_id: c for c in hybrid.dense.chunks}

    queries = [
        "Who is the all-time top scorer in World Cup history?",  # in corpus
        "Who won the 2026 World Cup?",                           # NOT in corpus
    ]
    for query in queries:
        top = search_and_rerank(hybrid, reranker, query, top_n=5)
        chunks = [by_id[chunk_id] for chunk_id, _ in top]
        answer = generate(query, chunks)
        print(f"\nQ: {query}")
        print(f"  abstained: {answer.abstained}")
        print(f"  answer: {answer.answer_text}")
        for claim in answer.claims:
            print(f"    - {claim.claim}  -> {claim.cited_chunk_ids}")
        if answer.invalid_citations:
            print(f"  INVALID citations: {answer.invalid_citations}")
