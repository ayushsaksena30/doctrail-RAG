def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion. Each ranking is a list of ids, best first.

    score(id) = sum over rankings of 1 / (k + rank), rank starting at 1.
    Uses ranks, not scores, because BM25 and cosine scores aren't comparable.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # ties broken by id so the output is deterministic
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))