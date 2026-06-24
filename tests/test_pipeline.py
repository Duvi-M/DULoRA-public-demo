import random

import torch

from dulora_demo.data import build_synthetic_text_data
from dulora_demo.model import TinyTextClassifier
from dulora_demo.reproducibility import set_global_seed
from dulora_demo.trainer import train_demo_model


def test_global_seed_reproduces_python_and_torch_sequences() -> None:
    set_global_seed(11)
    first = (random.random(), torch.rand(3))

    set_global_seed(11)
    second = (random.random(), torch.rand(3))

    assert first[0] == second[0]
    assert torch.equal(first[1], second[1])


def test_offline_pipeline_runs_end_to_end() -> None:
    set_global_seed(7)
    data = build_synthetic_text_data(
        train_size=16,
        eval_size=8,
        sequence_length=6,
        batch_size=4,
    )
    model = TinyTextClassifier(
        vocab_size=data.vocab_size,
        embedding_dim=8,
        num_labels=data.num_labels,
    )

    summary = train_demo_model(
        model,
        data.train_loader,
        data.eval_loader,
        epochs=1,
    )

    assert summary.train_loss >= 0
    assert 0 <= summary.eval_accuracy <= 1
    assert 0 <= summary.eval_f1 <= 1
