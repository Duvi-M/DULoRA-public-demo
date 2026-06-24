"""Configuration-driven entry point for the public offline demo."""

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

from dulora_demo.data import build_synthetic_text_data
from dulora_demo.demo_allocator import RoundRobinDemoAllocator
from dulora_demo.model import TinyTextClassifier
from dulora_demo.reproducibility import set_global_seed
from dulora_demo.trainer import train_demo_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DULoRA public demo.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/public/demo.yaml"),
    )
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    allocator_cfg = config["demo_allocator"]
    set_global_seed(config["seed"])

    data = build_synthetic_text_data(
        train_size=dataset_cfg["train_size"],
        eval_size=dataset_cfg["eval_size"],
        sequence_length=dataset_cfg["sequence_length"],
        vocab_size=dataset_cfg["vocab_size"],
        num_labels=dataset_cfg["num_labels"],
        batch_size=training_cfg["batch_size"],
        seed=config["seed"],
    )
    allocator = RoundRobinDemoAllocator(
        min_rank=allocator_cfg["min_rank"],
        max_rank=allocator_cfg["max_rank"],
        rank_step=allocator_cfg["rank_step"],
    )
    allocation = allocator.allocate(
        allocator_cfg["layer_names"],
        allocator_cfg["total_budget"],
    )

    model = TinyTextClassifier(data.vocab_size, num_labels=data.num_labels)
    summary = train_demo_model(
        model,
        data.train_loader,
        data.eval_loader,
        epochs=training_cfg["epochs"],
        learning_rate=training_cfg["learning_rate"],
    )

    print(f"Strategy: {allocation.strategy}")
    print(f"Rank pattern: {allocation.rank_pattern}")
    print(f"Total demo rank: {allocation.total_rank}")
    print(f"Metrics: {summary}")


if __name__ == "__main__":
    main()
