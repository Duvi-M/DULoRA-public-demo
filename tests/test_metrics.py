import pytest

from dulora_demo.evaluator import accuracy_score, binary_f1_score


def test_accuracy_score() -> None:
    assert accuracy_score([1, 0, 1, 1], [1, 0, 0, 1]) == pytest.approx(0.75)


def test_binary_f1_score() -> None:
    assert binary_f1_score([1, 0, 1, 1], [1, 0, 0, 1]) == pytest.approx(0.8)


def test_metrics_reject_different_lengths() -> None:
    with pytest.raises(ValueError):
        accuracy_score([1], [1, 0])
