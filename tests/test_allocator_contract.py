import pytest

from dulora_demo.demo_allocator import RoundRobinDemoAllocator


def test_allocator_returns_every_layer_and_respects_budget() -> None:
    allocator = RoundRobinDemoAllocator(min_rank=1, max_rank=3, rank_step=1)
    layers = ["layer_a", "layer_b", "layer_c"]

    result = allocator.allocate(layers, total_budget=7)

    assert set(result.rank_pattern) == set(layers)
    assert result.total_rank <= 7
    assert all(1 <= rank <= 3 for rank in result.rank_pattern.values())
    assert result.is_research_algorithm is False


def test_allocator_is_deterministic() -> None:
    allocator = RoundRobinDemoAllocator()
    layers = ["layer_a", "layer_b"]

    first = allocator.allocate(layers, total_budget=5)
    second = allocator.allocate(layers, total_budget=5)

    assert first == second


def test_allocator_rejects_impossible_minimum_budget() -> None:
    allocator = RoundRobinDemoAllocator(min_rank=2)

    with pytest.raises(ValueError, match="total_budget"):
        allocator.allocate(["layer_a", "layer_b"], total_budget=3)
