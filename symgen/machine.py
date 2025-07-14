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

  def parse(self, code: str):
    instructions = code.split()

    expression = list()

    for i, instruction in enumerate(instructions):
      if instruction in self.library:
        expression.append((instruction, ))

      elif instruction.startswith('(') and instruction.endswith(')'):
        addr = int(instruction[1:-1])
        expression.append(('variable', addr))

      elif instruction.startswith('[') and instruction.endswith(']'):
        addr = int(instruction[1:-1])
        expression.append(('load', addr))

      elif instruction.startswith('{') and instruction.endswith('}'):
        addr = int(instruction[1:-1])
        expression.append(('store', addr))

      else:
        try:
          value = float(instruction)
          expression.append(('const', value))

        except ValueError as e:
          raise ValueError(f'instruction is not understood: {instruction}') from e

    return expression

  def evaluate(self, expression, *inputs):
    if len(inputs) == 0:
      inputs = np.ndarray(shape=(0, 1), dtype=np.float32)
    else:
      inputs = np.stack(inputs, axis=0, dtype=float)

    return self(expression, inputs)

  def __call__(self, expression, inputs=None, *, out=None):
    if isinstance(expression, str):
      expression = self.parse(expression)

    if inputs is None:
      inputs = np.ndarray(shape=(0, 1), dtype=np.float32)

    if inputs.ndim == 1:
      expanded = True
      inputs = inputs[:, None]
    else:
      expanded = False

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

    if expanded:
      stack = np.squeeze(stack, axis=-1)

    if out is not None:
      out[:] = stack[:index]
      return out
    else:
      return np.copy(stack[:index])

  def trace(self, expression, inputs=None):
    if isinstance(expression, str):
      expression = self.parse(expression)

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