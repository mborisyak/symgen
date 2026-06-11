import numpy as np

from .common import Topology

__all__ = [
  'Trivial',
]

class Trivial(object):
  """No internal nodes; each of the `n_out` output cells links to all inputs."""
  def __call__(self, rng: np.random.Generator, n_in: int, n_out: int) -> Topology:
    graph: Topology = [None] * n_in

    links = tuple(range(n_in))
    for _ in range(n_out):
      graph.append((links, {}))

    return graph
