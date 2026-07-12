from rag.ingest import load_corpus
docs = load_corpus()
print(len(docs), "docs")            # expect 20
for d in docs[:3]:
    print(d.doc_id, "|", d.text[:500])