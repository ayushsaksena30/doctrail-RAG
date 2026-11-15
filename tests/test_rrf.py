import pytest
from Doctrail.rrf import rrf


def test_rrf_order_and_scores():
    list1 = ["C", "A", "B"]
    list2 = ["B", "A"]
    result = rrf([list1, list2])
    assert [doc for doc, _ in result] == ["B", "A", "C"]
    scores = dict(result)
    assert scores["C"] == pytest.approx(1 / 61)
    assert scores["A"] == pytest.approx(2 / 62)
    assert scores["B"] == pytest.approx(1 / 63 + 1 / 61)


def test_single_list_keeps_order():
    assert [d for d, _ in rrf([["x", "y", "z"]])] == ["x", "y", "z"]


def test_ties_are_deterministic():
    result = rrf([["b"], ["a"]])
    assert [d for d, _ in result] == ["a", "b"]