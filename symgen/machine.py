import inspect

import numpy as np

from .lib import Operation, merge

def inspect_op(op: Operation):
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

class StackMachine(object):
  def __init__(self, *libraries: dict[str, Operation], max_stack_size: int | None=None):
    self.library = merge(*libraries)

    self.properties = {
      name: inspect_op(op)
      for name, op in self.library.items()
    }
    self.max_stack_size = max_stack_size

  def __call__(self, expression, inputs, *, out=None):
    n, *batch = inputs.shape
    max_stack_size = len(expression) if self.max_stack_size is None else self.max_stack_size
    max_stack_size = min(max_stack_size, len(expression))

    stack = np.ndarray(shape=(max_stack_size, *batch), dtype=inputs.dtype)
    index = 0
    memory = np.ndarray(shape=(max_stack_size, *batch), dtype=inputs.dtype)

    for op, arg in expression:
      arity, scope = self.properties[op]
      arguments = [stack[index - i - 1] for i in range(arity)]

      kwargs = {}
      if 'inputs' in scope:
        kwargs['inputs'] = inputs
      if 'memory' in scope:
        kwargs['memory'] = memory
      if 'argument' in scope:
        kwargs['argument'] = arg

      index -= arity

      if 'out' in scope:
        self.library[op](*arguments, **kwargs, out=stack[index])
        index += 1
      else:
        result = self.library[op](*arguments, **kwargs)
        if result is not None:
          stack[index] = result
          index += 1

    if out is not None:
      out[:] = stack[:index]
      return out
    else:
      return np.copy(stack[:index])

  def trace(self, expression, inputs):
    n, *batch = inputs.shape

    stack = np.ndarray(shape=(len(expression), *batch), dtype=inputs.dtype)
    index = 0
    memory = np.ndarray(shape=(len(expression), *batch), dtype=inputs.dtype)
    expression_lens = np.ndarray(shape=(len(expression), ), dtype=np.uint32)

    for op, arg in expression:
      arity, scope = self.properties[op]
      l = 0
      arguments = []
      for i in range(arity):
        arguments.append(stack[index - l - 1])
        l += expression_lens[index - l - 1]

      expression_lens[index] = l + 1

      kwargs = {}
      if 'inputs' in scope:
        kwargs['inputs'] = inputs
      if 'memory' in scope:
        kwargs['memory'] = memory
      if 'argument' in scope:
        kwargs['argument'] = arg

      if 'out' in scope:
        self.library[op](*arguments, **kwargs, out=stack[index])
      else:
        result = self.library[op](*arguments, **kwargs)
        if result is not None:
          stack[index] = result

      index += 1

    return stack[:index]