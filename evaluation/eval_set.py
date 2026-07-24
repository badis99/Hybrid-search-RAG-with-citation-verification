"""Labelled evaluation data: query -> gold chunk_id(s).

Bootstrapped the cheap way: pick a chunk, write a question it answers, record the
pair. Categories matter more than raw count — a set made only of exact-term
queries would flatter BM25 and tell you nothing about hybrid.
"""
from __future__ import annotations

# (query, gold_chunk_ids, category)
EVAL_QUERIES: list[tuple[str, list[str], str]] = [
    # --- paraphrase: the wording deliberately avoids the source's terms ---
    ("Which nation lifted the trophy at the very first tournament?",
     ["world-cup-1930-uruguay::0"], "paraphrase"),
    ("Who has found the net more times than anyone across World Cup history?",
     ["world-cup-goalscoring-records::0"], "paraphrase"),
    ("Which defeat is remembered as a national tragedy for the host nation?",
     ["world-cup-1950-maracanazo::0"], "paraphrase"),
    ("Which team came from two goals down to win the final?",
     ["germany-national-team::0"], "paraphrase"),
    ("Which side is known for its yellow shirts and attacking flair?",
     ["brazil-national-team::0"], "paraphrase"),
    ("What did the winners get to keep permanently after a third victory?",
     ["world-cup-trophy::0"], "paraphrase"),
    ("Who finally won the biggest prize after losing an earlier final?",
     ["lionel-messi::0", "world-cup-2022-qatar::0"], "paraphrase"),
    ("Which player is said to have dominated one tournament more than anyone?",
     ["world-cup-1986-argentina::0", "diego-maradona::0"], "paraphrase"),

    # --- exact term: rare tokens BM25 should nail and embeddings blur ---
    ("catenaccio", ["italy-national-team::0"], "exact-term"),
    ("Maracanazo", ["world-cup-1950-maracanazo::0"], "exact-term"),
    ("Mineirazo", ["world-cup-2014-germany::0"], "exact-term"),
    ("Tofiq Bahramov", ["world-cup-1966-england::0"], "exact-term"),
    ("Estadio Centenario", ["world-cup-1930-uruguay::0"], "exact-term"),
    ("Just Fontaine", ["world-cup-goalscoring-records::0"], "exact-term"),
    ("Miracle of Bern", ["germany-national-team::0"], "exact-term"),
    ("jogo bonito", ["brazil-national-team::0"], "exact-term"),
    ("Lusail Stadium", ["world-cup-2022-qatar::0"], "exact-term"),
    ("Silvio Gazzaniga", ["world-cup-trophy::0"], "exact-term"),
    ("Vittorio Pozzo", ["italy-national-team::0"], "exact-term"),

    # --- factual: plain questions mixing named entities and meaning ---
    ("How many World Cups has Brazil won?",
     ["brazil-national-team::0", "world-cup-overview::1"], "factual"),
    ("Who scored a hat-trick in a World Cup final?",
     ["world-cup-1966-england::0", "world-cup-goalscoring-records::0"], "factual"),
    ("Which countries co-hosted the first World Cup held in Asia?",
     ["world-cup-2002-brazil::0"], "factual"),
    ("How was the 2014 World Cup final decided?",
     ["world-cup-2014-germany::0"], "factual"),
    ("Which nation has won the Women's World Cup four times?",
     ["womens-world-cup::0"], "factual"),
    ("Who scored twice in the 1998 World Cup final?",
     ["world-cup-1998-france::0"], "factual"),
    ("Which player won the World Cup three times?", ["pele::0"], "factual"),
    ("Which club did Maradona lead to two league titles?",
     ["diego-maradona::0"], "factual"),
    ("How many teams will play at the 2026 World Cup?",
     ["world-cup-overview::0"], "factual"),
    ("Who was sent off in the 2006 World Cup final?",
     ["italy-national-team::0"], "factual"),
    ("Which goalkeeper was named best player of a World Cup?",
     ["world-cup-2002-brazil::0"], "factual"),
]

# No gold chunk exists — the correct behaviour is abstention, not retrieval.
OUT_OF_CORPUS: list[str] = [
    "Which country won the 2026 World Cup?",
    "Who is the current manager of Real Madrid?",
    "What was the score in the 2030 World Cup final?",
]
