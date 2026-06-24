"""Public allocator contracts.

The research allocator is intentionally not included in this repository.
Implementations exposed here are demonstrations of the software interface only.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AllocationResult:
    """Result returned by a rank allocator."""

    rank_pattern: dict[str, int]
    strategy: str
    is_research_algorithm: bool = False

    @property
    def total_rank(self) -> int:
        return sum(self.rank_pattern.values())


class RankAllocator(ABC):
    """Interface shared by public demos and private research implementations."""

    @abstractmethod
    def allocate(
        self,
        layer_names: Sequence[str],
        total_budget: int,
    ) -> AllocationResult:
        """Return a rank for every supplied layer without exceeding the budget."""
