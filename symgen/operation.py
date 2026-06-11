from typing import Protocol, Any
import inspect

import numpy as np

__all__ = [
  'Operation',
  'inspect_op',
  'bind',
]

class Operation(Protocol):
  def __call__(self, *args: np.ndarray[np.number], **kwargs: Any):
    ...

def inspect_op(op: Operation):
  """Return `(arity, arguments)`: positional params are stack operands, keyword-only params are baked args."""
  signature = inspect.signature(op)
  parameters = signature.parameters

  assert all(p.kind != inspect.Parameter.VAR_POSITIONAL for p in parameters.values()), \
    'operations may not take *args (varargs are forbidden)'

  assert all(p.kind != inspect.Parameter.VAR_KEYWORD for p in parameters.values()), \
    'operations may not take **kwargs'

  arity = sum(
    1 for p in parameters.values()
    if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
  )

  arguments = tuple(
    name for name, p in parameters.items()
    if p.kind == inspect.Parameter.KEYWORD_ONLY
  )

  return arity, arguments

def bind(arguments: tuple[str, ...], args, memory):
  """Map keyword-only params to values: `memory` gets the memory list, the rest are baked args."""
  baked = [name for name in arguments if name != 'memory']
  assert len(args) == len(baked), \
    f'operation expects arguments {baked}, got {len(args)}: {args}'

  values = iter(args)
  return {
    name: (memory if name == 'memory' else next(values))
    for name in arguments
  }
