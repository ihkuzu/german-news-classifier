from __future__ import annotations

from collections import Counter


def accuracy(gold: list[str], pred: list[str | None]) -> float:
    if not gold:
        return 0.0
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def per_class(gold: list[str], pred: list[str | None], labels) -> dict[str, dict]:
    scores = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
    return scores


def macro_f1(gold: list[str], pred: list[str | None], labels) -> float:
    # classes without test examples would only add zeros
    scores = [s["f1"] for s in per_class(gold, pred, labels).values() if s["support"]]
    return sum(scores) / len(scores) if scores else 0.0


def top_confusions(
    gold: list[str], pred: list[str | None], k: int = 5
) -> list[tuple[str, str, int]]:
    pairs = Counter((g, p or "(none)") for g, p in zip(gold, pred) if g != p)
    return [(g, p, n) for (g, p), n in pairs.most_common(k)]
