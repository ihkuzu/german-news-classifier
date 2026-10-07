import pytest

from newsclf.metrics import accuracy, macro_f1, per_class, top_confusions

GOLD = ["a", "a", "a", "b", "b", "c"]
PRED = ["a", "a", "b", "b", "c", None]


def test_accuracy():
    assert accuracy(GOLD, PRED) == pytest.approx(3 / 6)
    assert accuracy([], []) == 0.0


def test_per_class_precision_recall_f1():
    scores = per_class(GOLD, PRED, ["a", "b", "c"])
    assert scores["a"] == pytest.approx(
        {"precision": 1.0, "recall": 2 / 3, "f1": 0.8, "support": 3}
    )
    assert scores["b"] == pytest.approx(
        {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    )
    assert scores["c"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}


def test_macro_f1_averages_classes_equally():
    assert macro_f1(GOLD, PRED, ["a", "b", "c"]) == pytest.approx((0.8 + 0.5 + 0.0) / 3)


def test_macro_f1_ignores_classes_without_examples():
    assert macro_f1(["a", "a"], ["a", "a"], ["a", "b"]) == 1.0


def test_missing_prediction_counts_as_wrong():
    assert accuracy(["a"], [None]) == 0.0


def test_top_confusions_are_ordered_by_count():
    gold = ["a", "a", "a", "b", "c"]
    pred = ["b", "b", "c", "a", None]
    assert top_confusions(gold, pred, k=2) == [("a", "b", 2), ("a", "c", 1)]
    assert ("c", "(none)", 1) in top_confusions(gold, pred, k=10)
