"""Deliberately simple allocator used by the public demonstration.

This module does not inspect gradients, parameters, activations, losses, or
model internals. It cycles through layer names in their supplied order and is
not the DULoRA research method.
"""

from dataclasses import dataclass
from typing import Sequence

from .allocator_interface import AllocationResult, RankAllocator


@dataclass
class RoundRobinDemoAllocator(RankAllocator):
    """Assign rank increments in a deterministic round-robin order."""

    min_rank: int = 1
    max_rank: int = 4
    rank_step: int = 1

    def __post_init__(self) -> None:
        if self.min_rank < 1:
            raise ValueError("min_rank must be positive")
        if self.max_rank < self.min_rank:
            raise ValueError("max_rank must be at least min_rank")
        if self.rank_step < 1:
            raise ValueError("rank_step must be positive")

    def allocate(
        self,
        layer_names: Sequence[str],
        total_budget: int,
    ) -> AllocationResult:
        names = list(dict.fromkeys(layer_names))
        if not names:
            return AllocationResult({}, strategy="round_robin_demo")

        minimum_budget = len(names) * self.min_rank
        if total_budget < minimum_budget:
            raise ValueError(
                f"total_budget must be at least {minimum_budget} for "
                f"{len(names)} layers"
            )

        ranks = {name: self.min_rank for name in names}
        remaining = total_budget - minimum_budget

        while remaining >= self.rank_step:
            changed = False
            for name in names:
                if remaining < self.rank_step:
                    break
                if ranks[name] + self.rank_step > self.max_rank:
                    continue
                ranks[name] += self.rank_step
                remaining -= self.rank_step
                changed = True
            if not changed:
                break

        return AllocationResult(
            rank_pattern=ranks,
            strategy="round_robin_demo",
            is_research_algorithm=False,
        )
