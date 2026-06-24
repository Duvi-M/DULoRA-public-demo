"""Minimal PyTorch training loop for the public demonstration."""

from dataclasses import dataclass

import torch
from torch import nn

from .evaluator import accuracy_score, binary_f1_score


@dataclass(frozen=True)
class TrainingSummary:
    train_loss: float
    eval_accuracy: float
    eval_f1: float


def train_demo_model(
    model: nn.Module,
    train_loader,
    eval_loader,
    epochs: int = 1,
    learning_rate: float = 1e-2,
    device: str = "cpu",
) -> TrainingSummary:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    latest_loss = 0.0

    model.train()
    for _ in range(epochs):
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            latest_loss = float(loss.detach().cpu())

    predictions: list[int] = []
    references: list[int] = []
    model.eval()
    with torch.no_grad():
        for input_ids, labels in eval_loader:
            logits = model(input_ids.to(device))
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            references.extend(labels.tolist())

    return TrainingSummary(
        train_loss=latest_loss,
        eval_accuracy=accuracy_score(predictions, references),
        eval_f1=binary_f1_score(predictions, references),
    )
