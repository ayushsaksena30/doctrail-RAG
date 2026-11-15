import pytest
from pathlib import Path
from Doctrail.index import build_bm25, load_chunks
from Doctrail.retrieve import bm25_search


def test_bm25_unit_in_memory():
    mock_chunks = [
        {
            "chunk_id": "c1",
            "rel_path": "doc1.pdf",
            "page": 1,
            "text": "The quick brown fox jumps over the lazy dog."
        },
        {
            "chunk_id": "c2",
            "rel_path": "doc2.pdf",
            "page": 3,
            "text": "Attention is all you need for transformer architectures."
        },
        {
            "chunk_id": "c3",
            "rel_path": "doc3.pdf",
            "page": 2,
            "text": "A peer-to-peer electronic cash system avoids double spending."
        }
    ]

    bm25 = build_bm25(mock_chunks)

    # Test top match ranking
    results = bm25_search("transformer attention", mock_chunks, bm25, k=2)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "c2"
    assert results[0]["rel_path"] == "doc2.pdf"

    # Test zero-score filtering
    no_results = bm25_search("completely_unrelated_query_xyz", mock_chunks, bm25, k=5)
    assert len(no_results) == 0


@pytest.mark.skipif(
    not Path("data/processed/chunks.jsonl").exists(),
    reason="data/processed/chunks.jsonl missing; run ingest first"
)

def test_bm25_integration_corpus():
    chunks = load_chunks("data/processed/chunks.jsonl")
    assert len(chunks) > 0
    bm25 = build_bm25(chunks)

    # Test 1: Japanese query should retrieve bitcoin_jp.pdf
    jp_query = "二重支払い"
    jp_results = bm25_search(jp_query, chunks, bm25, k=5)
    assert len(jp_results) > 0
    jp_files = [r["rel_path"] for r in jp_results]
    assert any("bitcoin_jp.pdf" in f for f in jp_files), f"Expected bitcoin_jp.pdf in {jp_files}"

    # Test 2: English query should retrieve attention_en.pdf
    en_query = "attention heads"
    en_results = bm25_search(en_query, chunks, bm25, k=5)
    assert len(en_results) > 0
    en_files = [r["rel_path"] for r in en_results]
    assert any("attention_en.pdf" in f for f in en_files), f"Expected attention_en.pdf in {en_files}"