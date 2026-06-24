"""Evaluation metrics with no dependency on the private research method."""

from collections.abc import Iterable


def accuracy_score(predictions: Iterable[int], labels: Iterable[int]) -> float:
    predictions = list(predictions)
    labels = list(labels)
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    if not labels:
        return 0.0
    return sum(p == y for p, y in zip(predictions, labels)) / len(labels)


def binary_f1_score(predictions: Iterable[int], labels: Iterable[int]) -> float:
    predictions = list(predictions)
    labels = list(labels)
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")

    true_positive = sum(p == 1 and y == 1 for p, y in zip(predictions, labels))
    false_positive = sum(p == 1 and y == 0 for p, y in zip(predictions, labels))
    false_negative = sum(p == 0 and y == 1 for p, y in zip(predictions, labels))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator
