import json
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class Doc:
    doc_id: str
    text: str
    source: str

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
