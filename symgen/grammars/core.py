from ..generator import op

__all__ = ['load', 'uniform_input']

def uniform_input(rng, memory):
  """Default leaf sampler: uniformly pick one of the node's local inputs (cells 0..len(memory)-1)."""
  return int(rng.integers(len(memory)))

load = op('load', uniform_input)
