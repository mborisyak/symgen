import numpy as np

from .common import Node, Topology

__all__ = [
  'RandomTopology',
]

class RandomTopology(object):
  """Each node links to 1..max_inputs distinct earlier cells; outputs link only to inputs and internal nodes."""
  def __init__(self, n_nodes: int, max_inputs: int):
    self.n_nodes = n_nodes
    self.max_inputs = max_inputs

  def _node(self, rng: np.random.Generator, pool: int) -> Node:
    if pool <= 0:
      return (), {}

    k = int(rng.integers(1, min(self.max_inputs, pool) + 1))
    links = tuple(int(c) for c in rng.choice(pool, size=k, replace=False))
    return links, {}

  def __call__(self, rng: np.random.Generator, n_in: int, n_out: int) -> Topology:
    graph: Topology = [None] * n_in

    for j in range(self.n_nodes):
      graph.append(self._node(rng, n_in + j))

    pool = n_in + self.n_nodes
    for _ in range(n_out):
      graph.append(self._node(rng, pool))

    return graph
