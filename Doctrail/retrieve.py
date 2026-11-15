from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from Doctrail.tokenizer import tokenize


def bm25_search(
    query: str,
    chunks: List[Dict[str, Any]],
    bm25: BM25Okapi,
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Ranks chunks using BM25, filters out non-positive scores, and returns
    top-k results formatted with metadata and a 200-character snippet.
    """
    query_tokens = tokenize(query)
    if not query_tokens or not chunks:
        return []

    scores = bm25.get_scores(query_tokens)

    # Pair each chunk with its score and filter out zero or negative scores
    scored_results = [
        (score, chunk)
        for score, chunk in zip(scores, chunks)
        if score > 0.0
    ]

    # Sort descending by BM25 score
    scored_results.sort(key=lambda x: x[0], reverse=True)

    top_results: List[Dict[str, Any]] = []
    for score, chunk in scored_results[:k]:
        raw_text = chunk.get("text", "")
        # Normalize whitespace and take first 200 characters
        snippet = " ".join(raw_text.split())[:200]
        top_results.append({
            "chunk_id": chunk["chunk_id"],
            "rel_path": chunk["rel_path"],
            "page": chunk["page"],
            "score": round(float(score), 4),
            "snippet": snippet
        })

    return top_results