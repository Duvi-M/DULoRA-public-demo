"""Dataset helpers for the public demo."""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class SyntheticTextData:
    train_loader: DataLoader
    eval_loader: DataLoader
    vocab_size: int
    num_labels: int


def build_synthetic_text_data(
    train_size: int = 32,
    eval_size: int = 16,
    sequence_length: int = 12,
    vocab_size: int = 64,
    num_labels: int = 2,
    batch_size: int = 8,
    seed: int = 7,
) -> SyntheticTextData:
    """Create deterministic token IDs for fast, offline pipeline checks."""
    generator = torch.Generator().manual_seed(seed)

    def make_dataset(size: int) -> TensorDataset:
        tokens = torch.randint(
            low=0,
            high=vocab_size,
            size=(size, sequence_length),
            generator=generator,
        )
        labels = (tokens.sum(dim=1) % num_labels).long()
        return TensorDataset(tokens, labels)

    train_dataset = make_dataset(train_size)
    eval_dataset = make_dataset(eval_size)

    return SyntheticTextData(
        train_loader=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        ),
        eval_loader=DataLoader(eval_dataset, batch_size=batch_size),
        vocab_size=vocab_size,
        num_labels=num_labels,
    )


def load_huggingface_text_dataset(
    dataset_name: str,
    tokenizer_name: str,
    train_size: int = 64,
    eval_size: int = 32,
    max_length: int = 96,
):
    """Load a small public text dataset for an optional online demonstration."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_split = dataset["train"].select(
        range(min(train_size, len(dataset["train"])))
    )
    eval_name = "test" if "test" in dataset else "validation"
    eval_split = dataset[eval_name].select(
        range(min(eval_size, len(dataset[eval_name])))
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_split = train_split.map(tokenize, batched=True)
    eval_split = eval_split.map(tokenize, batched=True)
    return train_split, eval_split, tokenizer
