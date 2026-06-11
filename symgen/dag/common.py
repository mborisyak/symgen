"""Unified graph format: a topology is a topologically-sorted list of cells (`None` = input,
`(links, context)` = node referencing earlier cells by local index)."""
from typing import Protocol, Any, TypeAlias

import numpy as np

__all__ = [
  'Node',
  'Topology',
  'TopologyGenerator',
]

Node: TypeAlias = tuple[tuple[int, ...], dict[str, Any]] | None
Topology: TypeAlias = list[Node]

class TopologyGenerator(Protocol):
  def __call__(self, rng: np.random.Generator, n_in: int, n_out: int) -> Topology:
    ...
