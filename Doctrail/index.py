import json
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from Doctrail.tokenizer import tokenize


def load_chunks(path: str = "data/processed/chunks.jsonl") -> List[Dict[str, Any]]:
    """Loads all chunk dictionaries from a JSONL file."""
    chunks: List[Dict[str, Any]] = []
    file_path = Path(path)
    if not file_path.exists():
        return chunks

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_bm25(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    """Tokenizes each chunk's text and builds an in-memory BM25Okapi index."""
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized_corpus)