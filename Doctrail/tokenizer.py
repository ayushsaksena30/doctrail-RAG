import re
import unicodedata

# Hiragana + Katakana (incl. long-vowel mark), CJK ideographs, half-width katakana
_CJK = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f"
_RUN_RE = re.compile(rf"[{_CJK}]+|[A-Za-z0-9]+")
_CJK_START = re.compile(rf"[{_CJK}]")


def tokenize(text: str) -> list[str]:
    """Latin/digit runs -> lowercase words. CJK runs -> character bigrams.

    Why: Japanese has no spaces, so splitting on whitespace would leave a whole
    sentence as one token and BM25 could never match a query against it.
    """
    text = unicodedata.normalize("NFKC", text).lower()  # full-width -> normal
    tokens: list[str] = []
    for run in _RUN_RE.findall(text):
        if _CJK_START.match(run[0]):
            if len(run) == 1:
                tokens.append(run)  # a lone character can't form a bigram
            else:
                tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
        else:
            tokens.append(run)
    return tokens