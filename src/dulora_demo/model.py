"""Model builders for offline tests and optional Transformers/PEFT demos."""

from typing import Sequence

import torch
from torch import nn


class TinyTextClassifier(nn.Module):
    """Small classifier used for fast tests without network downloads."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 16,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids).mean(dim=1)
        return self.classifier(hidden)


def build_peft_sequence_classifier(
    model_name: str,
    num_labels: int,
    rank_pattern: dict[str, int] | None = None,
    target_modules: Sequence[str] = ("query", "value"),
    demo_rank: int = 2,
):
    """Build a standard PEFT model using a supplied public/demo rank pattern.

    This function demonstrates integration only. It does not calculate scores
    or derive an adaptive pattern.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    kwargs = {
        "r": demo_rank,
        "lora_alpha": 4,
        "lora_dropout": 0.05,
        "target_modules": list(target_modules),
        "bias": "none",
    }
    if rank_pattern:
        kwargs["rank_pattern"] = rank_pattern

    return get_peft_model(model, LoraConfig(**kwargs))
