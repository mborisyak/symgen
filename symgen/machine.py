import numpy as np

from .operation import inspect_op
from .lib import Operation, merge

class StackMachine(object):
  def __init__(self, *libraries: dict[str, Operation], max_stack_size: int | None=None):
    self.library = merge(*libraries)

    self.properties = {
      name: inspect_op(op)
      for name, op in self.library.items()
    }
    self.max_stack_size = max_stack_size

  def __call__(self, expression, inputs=None, *, out=None):
    if inputs is None:
      n, batch = 0, ()
    else:
      n, *batch = inputs.shape

    max_stack_size = len(expression) if self.max_stack_size is None else self.max_stack_size
    max_stack_size = min(max_stack_size, len(expression))

    stack = np.ndarray(shape=(max_stack_size, *batch), dtype=inputs.dtype)
    index = 0
    memory = np.ndarray(shape=(max_stack_size, *batch), dtype=inputs.dtype)

    for op, *args in expression:
      arity, scope = self.properties[op]
      arguments = [stack[index - i - 1] for i in range(arity)]

      kwargs = {}
      if 'inputs' in scope:
        kwargs['inputs'] = inputs
      if 'memory' in scope:
        kwargs['memory'] = memory
      if 'argument' in scope:
        kwargs['argument'], = args

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

  def trace(self, expression, inputs=None):
    if inputs is None:
      inputs = np.ndarray(shape=(0, 1), dtype=np.float32)

    n, *batch = inputs.shape
    dtype = inputs.dtype

    stack = np.ndarray(shape=(len(expression), *batch), dtype=dtype)
    index = 0
    memory = np.ndarray(shape=(len(expression), *batch), dtype=dtype)
    expression_lens = np.ndarray(shape=(len(expression), ), dtype=np.uint32)

    for op, *arg in expression:
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
        kwargs['argument'], = arg

      if 'out' in scope:
        self.library[op](*arguments, **kwargs, out=stack[index])
      else:
        result = self.library[op](*arguments, **kwargs)
        if result is not None:
          stack[index] = result

      index += 1

    return stack[:index]