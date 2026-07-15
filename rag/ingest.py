import json
from pathlib import Path
from dataclasses import dataclass

import tiktoken

# Token counter for measuring chunk size. cl100k_base is a solid general-purpose
# tokenizer; your Phase 2 embedding model (all-MiniLM-L6-v2) tokenizes differently
# and caps at 256 tokens, so treat these counts as a close proxy, not exact.
_ENCODER = tiktoken.get_encoding("cl100k_base")


def _ntok(text: str) -> int:
    return len(_ENCODER.encode(text))

@dataclass(frozen=True)
class Doc:
    doc_id: str
    text: str
    source: str

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int

def load_corpus(corpus_dir="data/corpus") -> list[Doc]:
    docs = []
    for path in sorted(Path(corpus_dir).iterdir()):
        if path.suffix in (".md", ".txt"):
            text = path.read_text(encoding="utf-8")
            doc_id = path.stem
            source = str(path)
            docs.append(Doc(doc_id,text,source))
        elif path.suffix == ".json":
            datas = (json.load(path) if isinstance(json.load(path),list) else [json.load(path)])
            for data in data:
                docs.append(Doc(data.doc_id,data.text,data.source))

    return docs

# Separators tried largest-first. Recursive splitting only drops to a finer
# separator when a piece is still bigger than `size` tokens.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _hard_split(text: str, start: int, end: int, size: int) -> list[tuple[int, int]]:
    """Last resort for an unbreakable run (no separators left): slice by tokens."""
    ids = _ENCODER.encode(text[start:end])
    spans, pos = [], start
    for i in range(0, len(ids), size):
        piece = _ENCODER.decode(ids[i:i + size])
        spans.append((pos, pos + len(piece)))
        pos += len(piece)
    return spans


def _atomic_spans(text: str, size: int) -> list[tuple[int, int]]:
    """Split text into contiguous (start, end) char spans, each <= `size` tokens
    where possible. Contiguous means text[start:end] reconstructs the original,
    separators and whitespace included."""

    def rec(start: int, end: int, sep_idx: int) -> list[tuple[int, int]]:
        if _ntok(text[start:end]) <= size:
            return [(start, end)]
        if sep_idx >= len(_SEPARATORS):
            return _hard_split(text, start, end, size)
        sep = _SEPARATORS[sep_idx]
        spans, pos = [], start
        for part in text[start:end].split(sep):
            p_start, p_end = pos, pos + len(part)
            if part.strip():
                spans.extend(rec(p_start, p_end, sep_idx + 1))
            pos = p_end + len(sep)          # step over the separator we split on
        return spans

    return rec(0, len(text), 0)


def chunk_documents(docs, size=300, overlap=45) -> list[Chunk]:
    """Split each Doc into overlapping Chunks with deterministic IDs.

    Recursive splitting first breaks a document into atomic units on natural
    boundaries (paragraph -> line -> sentence -> word). Units are then greedily
    packed into chunks up to `size` tokens, and each new chunk re-includes about
    `overlap` tokens from the tail of the previous one, so a fact straddling a
    boundary survives whole in at least one chunk. chunk_id = f"{doc_id}::{i}" is
    stable across runs because doc_id is stable and i is positional.
    """
    chunks: list[Chunk] = []
    for doc in docs:
        spans = _atomic_spans(doc.text, size)
        i, idx, n = 0, 0, len(spans)
        while i < n:
            # Grow the chunk one unit at a time until the next unit would overflow.
            j = i + 1
            while j < n and _ntok(doc.text[spans[i][0]:spans[j][1]]) <= size:
                j += 1
            start, end = spans[i][0], spans[j - 1][1]
            chunks.append(
                Chunk(f"{doc.doc_id}::{idx}", doc.doc_id, doc.text[start:end], start, end)
            )
            idx += 1
            if j >= n:
                break
            # Back up so the next chunk re-includes ~overlap tokens of context.
            back = j - 1
            while back > i and _ntok(doc.text[spans[back][0]:end]) < overlap:
                back -= 1
            i = max(back, i + 1)            # max(...) guarantees forward progress
    return chunks


if __name__ == "__main__":
    # Quick Phase 1 checkpoint: token-length distribution + a few sample chunks.
    chunks = chunk_documents(load_corpus())
    lengths = sorted(_ntok(c.text) for c in chunks)
    print(f"{len(chunks)} chunks from the corpus")
    print(f"tokens/chunk  min={lengths[0]}  "
          f"median={lengths[len(lengths) // 2]}  max={lengths[-1]}")
    for c in chunks[:3]:
        print(f"\n[{c.chunk_id}]  chars {c.start}-{c.end}\n{c.text[:200]}...")