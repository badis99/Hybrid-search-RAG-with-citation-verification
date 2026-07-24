"""Phase 7 — citation verification: does the cited chunk actually support the claim?"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from rag.generate import Answer, ResolvedClaim
from rag.ingest import Chunk

load_dotenv()

JUDGE_MODEL = "openai/gpt-oss-120b"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
VERIFY_THRESHOLD = 0.5

# "Being topically related is NOT support" is the load-bearing sentence: a lenient
# judge's usual failure is accepting a passage about the right subject that never
# states the fact. The required verbatim span is the second guard.
JUDGE_SYSTEM = """You check whether a PASSAGE supports a CLAIM. Be strict.

verdict:
- "supported" ONLY if the passage explicitly states the claim. Every component of
  the claim (names, numbers, dates, superlatives) must appear in the passage or
  follow directly from it.
- "partially_supported" if the passage states some of the claim but not all of it.
- "not_supported" if the passage does not state the claim, is merely about the
  same topic, or contradicts it.

evidence_span must be an EXACT VERBATIM quote copied from the passage that states
the claim. Never paraphrase. If nothing in the passage states the claim, return an
empty string with verdict "not_supported".

Being about the same subject is NOT support. A passage on the same topic that does
not state this specific fact is "not_supported"."""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["supported", "partially_supported", "not_supported"],
        },
        "evidence_span": {"type": "string"},
    },
    "required": ["verdict", "evidence_span"],
    "additionalProperties": False,
}


class RawVerdict(BaseModel):
    verdict: str
    evidence_span: str


@dataclass
class Verification:
    claim: str
    verdict: str          # supported | partially_supported | not_supported
    score: float          # entailment probability (NLI) or confidence (judge)
    evidence_span: str    # judge only; NLI has no span to point at
    method: str
    span_found: bool      # did the quoted span actually occur in the passage?

    @property
    def is_supported(self) -> bool:
        return self.verdict == "supported"


@dataclass
class VerifiedAnswer:
    answer_text: str
    verified: list[Verification]
    unsupported: list[Verification]
    overall_faithfulness: float


class LLMJudge:
    """Flexible on paraphrase, but needs the strict rubric above to stay honest."""

    method = "llm-judge"

    def __init__(self, model: str = JUDGE_MODEL):
        self.client = Groq()
        self.model = model

    def _call(self, claim: str, premise: str, attempts: int = 6):
        """Groq's free tier caps tokens-per-minute; the 429 tells us how long to
        wait, so honour it rather than hammering."""
        for attempt in range(attempts):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": f"PASSAGE:\n{premise}\n\nCLAIM:\n{claim}"},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "verdict",
                            "schema": VERDICT_SCHEMA,
                            "strict": True,
                        },
                    },
                )
            except RateLimitError as err:
                if attempt == attempts - 1:
                    raise
                match = re.search(r"try again in ([\d.]+)s", str(err))
                time.sleep(float(match.group(1)) + 1.0 if match else 5.0 * (attempt + 1))
        raise RuntimeError("unreachable")

    def verify(self, claim: str, premise: str) -> Verification:
        response = self._call(claim, premise)
        raw = RawVerdict.model_validate_json(response.choices[0].message.content)
        span = raw.evidence_span.strip()
        # Don't take the quote on trust — a span the judge invented is itself a
        # signal the verdict is unreliable.
        span_found = bool(span) and span.lower() in premise.lower()
        return Verification(
            claim=claim,
            verdict=raw.verdict,
            score=1.0 if raw.verdict == "supported" else 0.0,
            evidence_span=span,
            method=self.method,
            span_found=span_found,
        )


class NLIVerifier:
    """Fast, deterministic, no prompt drift — and a genuinely independent model."""

    method = "nli"

    def __init__(self, model_name: str = NLI_MODEL, threshold: float = VERIFY_THRESHOLD):
        self.model = CrossEncoder(model_name)
        self.threshold = threshold
        # Map labels BY NAME. Index order is not standardised across NLI
        # checkpoints, and guessing wrong silently inverts every verdict.
        self.label_index = {
            name.lower(): idx for idx, name in self.model.config.id2label.items()
        }

    def verify(self, claim: str, premise: str) -> Verification:
        # premise = the cited passage, hypothesis = the claim. Swapping them asks
        # a different question entirely.
        logits = np.asarray(self.model.predict([(premise, claim)]))[0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        entail = float(probs[self.label_index["entailment"]])
        top = self.model.config.id2label[int(probs.argmax())].lower()
        verdict = "supported" if entail >= self.threshold else "not_supported"
        return Verification(
            claim=claim,
            verdict=verdict,
            score=entail,
            evidence_span=f"nli:{top}",
            method=self.method,
            span_found=False,
        )


def verify_answer(
    answer: Answer,
    chunks_by_id: dict[str, Chunk],
    verifier,
) -> VerifiedAnswer:
    """Verify each claim against ITS OWN cited chunk(s).

    Checking "does any retrieved chunk support this" would hide mis-citations —
    a true fact attributed to the wrong source would pass. Cited-chunk-first is
    what makes a mis-citation detectable.
    """
    verified: list[Verification] = []
    unsupported: list[Verification] = []
    for claim in answer.claims:
        premise = "\n\n".join(
            chunks_by_id[cid].text for cid in claim.cited_chunk_ids if cid in chunks_by_id
        )
        if not premise:
            # A claim citing nothing resolvable cannot be supported.
            unsupported.append(
                Verification(claim.claim, "not_supported", 0.0, "", verifier.method, False)
            )
            continue
        result = verifier.verify(claim.claim, premise)
        (verified if result.is_supported else unsupported).append(result)

    total = len(verified) + len(unsupported)
    # An abstention has no claims: vacuously faithful, since it asserted nothing.
    faithfulness = 1.0 if total == 0 else len(verified) / total
    return VerifiedAnswer(answer.answer_text, verified, unsupported, faithfulness)


# (claim, cited chunk_id, should_be_supported) — the Phase 9 verifier eval set,
# started here so NLI and the judge can be compared on identical inputs.
LABELLED_CLAIMS: list[tuple[str, str, bool]] = [
    # --- genuinely supported by the cited chunk ---
    ("Brazil has won the World Cup five times.", "brazil-national-team::0", True),
    ("Miroslav Klose is the all-time leading goalscorer in World Cup history.",
     "world-cup-goalscoring-records::0", True),
    ("Just Fontaine scored thirteen goals at the 1958 World Cup.",
     "world-cup-goalscoring-records::0", True),
    ("The first World Cup was held in Uruguay in 1930.", "world-cup-1930-uruguay::0", True),
    ("Geoff Hurst scored a hat-trick in the 1966 World Cup final.",
     "world-cup-1966-england::0", True),
    ("Italy has won the World Cup four times.", "italy-national-team::0", True),
    ("Pele won the World Cup three times.", "pele::0", True),
    ("Argentina won the 2022 World Cup.", "world-cup-2022-qatar::0", True),
    ("Germany beat Brazil 7-1 in the 2014 semi-final.", "world-cup-2014-germany::0", True),
    ("Spain won the 2023 Women's World Cup.", "womens-world-cup::0", True),
    # --- contradicted by the cited chunk ---
    ("Brazil has won the World Cup three times.", "brazil-national-team::0", False),
    ("Miroslav Klose scored 25 goals in World Cup history.",
     "world-cup-goalscoring-records::0", False),
    ("The first World Cup was held in Uruguay in 1934.", "world-cup-1930-uruguay::0", False),
    ("Pele won the World Cup two times.", "pele::0", False),
    ("France won the 2022 World Cup.", "world-cup-2022-qatar::0", False),
    # --- true facts, but cited to a chunk that does not state them (mis-citations) ---
    ("Miroslav Klose is the all-time leading goalscorer in World Cup history.",
     "pele::0", False),
    ("Italy plays the defensive system known as catenaccio.",
     "brazil-national-team::0", False),
    ("Diego Maradona scored the Hand of God goal.", "world-cup-1930-uruguay::0", False),
    ("The 1950 defeat is remembered as the Maracanazo.", "germany-national-team::0", False),
    ("Lionel Messi won the Golden Ball in 2022.", "world-cup-1966-england::0", False),
]


if __name__ == "__main__":
    from rag.generate import Answer, ResolvedClaim
    from rag.ingest import chunk_documents, load_corpus

    by_id = {c.chunk_id: c for c in chunk_documents(load_corpus())}
    judge, nli = LLMJudge(), NLIVerifier()

    RECORDS = "world-cup-goalscoring-records::0"
    TRUE_CLAIM = "Miroslav Klose is the all-time leading goalscorer in World Cup history."

    tests = [
        ("1. genuine claim (must PASS)",
         Answer("Klose leads.", [ResolvedClaim(TRUE_CLAIM, [RECORDS])], False, []), True),
        ("2. injected hallucination (must FLAG)",
         Answer("Klose scored 25.",
                [ResolvedClaim("Miroslav Klose scored 25 goals in World Cup history.", [RECORDS])],
                False, []), False),
        ("3. mis-citation: true fact, wrong chunk (must FLAG)",
         Answer("Klose leads.", [ResolvedClaim(TRUE_CLAIM, ["pele::0"])], False, []), False),
    ]

    print("=" * 74)
    for label, answer, expect_supported in tests:
        print(f"\n{label}")
        for verifier in (judge, nli):
            out = verify_answer(answer, by_id, verifier)
            v = (out.verified + out.unsupported)[0]
            ok = "PASS" if v.is_supported == expect_supported else "*** WRONG ***"
            print(f"   {verifier.method:10} verdict={v.verdict:20} "
                  f"score={v.score:.2f}  faith={out.overall_faithfulness:.2f}  [{ok}]")
            if v.method == "llm-judge" and v.evidence_span:
                print(f"              span_found={v.span_found}  {v.evidence_span[:70]!r}")

    print("\n" + "=" * 74)
    print(f"4. NLI vs LLM-judge on {len(LABELLED_CLAIMS)} labelled claims\n")
    stats = {"llm-judge": [0, 0], "nli": [0, 0]}   # [correct, total]
    disagreements = []
    for claim, chunk_id, expected in LABELLED_CLAIMS:
        premise = by_id[chunk_id].text
        results = {}
        for verifier in (judge, nli):
            r = verifier.verify(claim, premise)
            results[verifier.method] = r
            stats[verifier.method][1] += 1
            if r.is_supported == expected:
                stats[verifier.method][0] += 1
        if results["llm-judge"].is_supported != results["nli"].is_supported:
            disagreements.append((claim, chunk_id, expected, results))

    for method, (correct, total) in stats.items():
        print(f"   {method:10} {correct}/{total} correct  ({correct / total:.0%})")

    print(f"\n   they disagreed on {len(disagreements)} of {len(LABELLED_CLAIMS)}:")
    for claim, chunk_id, expected, results in disagreements:
        judged, nlied = results["llm-judge"], results["nli"]
        right = "judge" if judged.is_supported == expected else "nli"
        print(f"     - {claim[:58]!r}")
        print(f"       cited {chunk_id} | truth={'supported' if expected else 'unsupported'} "
              f"| judge={judged.verdict} nli={nlied.verdict}({nlied.score:.2f}) -> {right} right")
