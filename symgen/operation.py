from typing import Protocol, Any

import numpy as np

__all__ = [
  'Operation',
  'inspect_op',
]

class Operation(Protocol):
  def __call__(self, *args: np.ndarray[np.number], **kwargs: Any):
    ...

def inspect_op(op: Operation):
  import inspect

  signature = inspect.signature(op)
  assert all(p.kind != inspect.Parameter.VAR_POSITIONAL for _, p in signature.parameters.items()), \
    'functions with variable positional arguments are not valid operations'

  assert all(p.kind != inspect.Parameter.VAR_KEYWORD for _, p in signature.parameters.items()), \
    'functions with variable keyword arguments are not valid operations'

  arity = len([
    p for _, p in signature.parameters.items()
    if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or p.kind == inspect.Parameter.POSITIONAL_ONLY
  ])

  scope_variables = set([
    name for name, p in signature.parameters.items()
    if p.kind == inspect.Parameter.KEYWORD_ONLY
  ])

  return arity, scope_variables