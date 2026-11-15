from rank_bm25 import BM25Okapi
from Doctrail.tokenizer import tokenize


def test_english_words():
    assert tokenize("Hello, World 2024!") == ["hello", "world", "2024"]


def test_japanese_bigrams():
    assert tokenize("検索精度") == ["検索", "索精", "精度"]


def test_single_cjk_char_kept():
    assert tokenize("猫") == ["猫"]


def test_mixed_script():
    assert tokenize("Bitcoin電子通貨") == ["bitcoin", "電子", "子通", "通貨"]


def test_fullwidth_normalized():
    assert tokenize("ＡＢＣ１２３") == ["abc123"]


def test_punctuation_breaks_runs():
    assert tokenize("検索、精度") == ["検索", "精度"]


def test_japanese_query_finds_japanese_chunk():
    docs = [
        "ビットコインは分散型の電子通貨システムである",
        "The transformer relies entirely on attention mechanisms",
        "テスラの売上高は前年から増加した",
    ]
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    scores = bm25.get_scores(tokenize("電子通貨"))
    assert scores.argmax() == 0