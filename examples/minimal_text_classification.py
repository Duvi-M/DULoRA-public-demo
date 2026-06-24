"""Run the offline public demo from the repository root."""

from dulora_demo.data import build_synthetic_text_data
from dulora_demo.demo_allocator import RoundRobinDemoAllocator
from dulora_demo.model import TinyTextClassifier
from dulora_demo.reproducibility import set_global_seed
from dulora_demo.trainer import train_demo_model


def main() -> None:
    seed = 7
    set_global_seed(seed)
    data = build_synthetic_text_data(seed=seed)
    allocator = RoundRobinDemoAllocator(min_rank=1, max_rank=3)
    allocation = allocator.allocate(
        ["encoder.block_a", "encoder.block_b", "encoder.block_c"],
        total_budget=6,
    )

    model = TinyTextClassifier(
        vocab_size=data.vocab_size,
        num_labels=data.num_labels,
    )
    summary = train_demo_model(
        model=model,
        train_loader=data.train_loader,
        eval_loader=data.eval_loader,
    )

    print("Public allocator:", allocation.strategy)
    print("Demo rank pattern:", allocation.rank_pattern)
    print("Research algorithm included:", allocation.is_research_algorithm)
    print("Evaluation:", summary)


if __name__ == "__main__":
    main()
