import numpy as np
import jax
import jax.numpy as jnp

from .operation import inspect_op, bind, Operation
from .assembly import pretty
from .lib import merge

class StackMachine(object):
  def __init__(self, *libraries: dict[str, Operation], max_stack_size: int | None=None):
    library = merge(*libraries)

    self.properties = {
      name: inspect_op(op)
      for name, op in library.items()
    }
    self.library = {
      name: (op if 'memory' in self.properties[name][1] else jax.jit(op))
      for name, op in library.items()
    }
    self.max_stack_size = max_stack_size

  @staticmethod
  def _address(body: str):
    """Integer body -> numeric cell; identifier body -> named cell."""
    return int(body) if body.isdigit() else body

  def parse(self, code: str):
    instructions = code.split()

    expression = list()

    for i, instruction in enumerate(instructions):
      if instruction in self.library:
        expression.append((instruction, ))

      elif instruction.startswith('(') and instruction.endswith(')'):
        expression.append(('load', self._address(instruction[1:-1])))

      elif instruction.startswith('[') and instruction.endswith(']'):
        expression.append(('store', self._address(instruction[1:-1])))

      else:
        try:
          value = float(instruction)
          expression.append(('const', value))

        except ValueError as e:
          raise ValueError(f'instruction is not understood: {instruction}') from e

    return expression

  def _resolve(self, expression, input_names):
    """Resolve named load/store addresses to integer cells (inputs first, then first-seen names)."""
    names = {name: i for i, name in enumerate(input_names)}
    next_cell = len(names)

    resolved = list()
    for op, *args in expression:
      if op in ('load', 'store') and len(args) > 0 and isinstance(args[0], str):
        name = args[0]
        if name not in names:
          names[name] = next_cell
          next_cell += 1
        resolved.append((op, names[name]))
      else:
        resolved.append((op, *args))

    return resolved

  def evaluate(self, expression, *inputs):
    if len(inputs) == 0:
      inputs = np.ndarray(shape=(0, 1), dtype=np.float32)
    else:
      inputs = np.stack(inputs, axis=0, dtype=float)

    return self(expression, inputs)

  @staticmethod
  def _n_cells(expression, n_in):
    """Number of memory cells: inputs plus any cell the program loads/stores."""
    addresses = [args[0] for op, *args in expression if op in ('load', 'store') and len(args) > 0]
    return max(n_in, 1 + max(addresses)) if len(addresses) > 0 else n_in

  def _prepare(self, expression, inputs, kwargs):
    """Return (resolved expression, inputs array, n_cells)."""
    if isinstance(expression, str):
      expression = self.parse(expression)

    if len(kwargs) > 0:
      if inputs is not None:
        raise ValueError('provide inputs either positionally or as keyword arguments, not both')
      input_names = list(kwargs.keys())
      inputs = np.stack([np.asarray(kwargs[name]) for name in input_names], axis=0)
    else:
      input_names = []

    expression = self._resolve(expression, input_names)

    if inputs is None:
      inputs = np.ndarray(shape=(0, 1), dtype=np.float32)

    return expression, inputs, self._n_cells(expression, inputs.shape[0])

  def _seed_memory(self, inputs, n_cells):
    """Memory is a list of jax arrays; cells 0..n_in-1 are the input rows."""
    memory = [None] * n_cells
    for i in range(inputs.shape[0]):
      memory[i] = inputs[i]
    return memory

  def _run(self, expression, inputs, n_cells):
    """Execute a resolved program on jax `inputs` and return the output stack as (n_out, *batch)."""
    batch = inputs.shape[1:]
    memory = self._seed_memory(inputs, n_cells)
    stack = []

    for op, *args in expression:
      arity, arguments = self.properties[op]
      operands = [stack.pop() for _ in range(arity)]

      result = self.library[op](*operands, **bind(arguments, args, memory))
      if result is not None:
        stack.append(result)

    if len(stack) > 0:
      return jnp.stack([jnp.broadcast_to(v, batch) for v in stack])
    else:
      return jnp.zeros((0, *batch), dtype=inputs.dtype)

  def __call__(self, expression, inputs=None, *, out=None, **kwargs):
    expression, inputs, n_cells = self._prepare(expression, inputs, kwargs)

    inputs = jnp.asarray(inputs)
    if inputs.ndim == 1:
      expanded = True
      inputs = inputs[:, None]
    else:
      expanded = False

    outputs = self._run(expression, inputs, n_cells)

    if expanded:
      outputs = jnp.squeeze(outputs, axis=-1)

    if out is not None:
      out[:] = np.asarray(outputs)
      return out
    else:
      return outputs

  def trace(self, expression, inputs=None, **kwargs):
    expression, inputs, n_cells = self._prepare(expression, inputs, kwargs)

    inputs = jnp.asarray(inputs)
    if inputs.ndim == 1:
      inputs = inputs[:, None]

    batch = inputs.shape[1:]
    memory = self._seed_memory(inputs, n_cells)
    stack = []
    records = []

    for op, *args in expression:
      arity, arguments = self.properties[op]
      operands = [stack.pop() for _ in range(arity)]

      result = self.library[op](*operands, **bind(arguments, args, memory))
      if result is not None:
        stack.append(result)
        records.append(result)
      else:
        records.append(operands[0] if len(operands) > 0 else jnp.zeros(batch, dtype=inputs.dtype))

    return jnp.stack([jnp.broadcast_to(v, batch) for v in records])

  def compile(self, program, n_in):
    """Compile a program into a fused, jitted callable f(inputs) -> outputs."""
    if isinstance(program, str):
      program = self._resolve(self.parse(program), [])

    n_cells = self._n_cells(program, n_in)
    return jax.jit(lambda inputs: self._run(program, jnp.asarray(inputs), n_cells))

  def show(self, program):
    """Render a program as a readable expression (see `symgen.assembly.pretty`)."""
    if isinstance(program, str):
      program = self._resolve(self.parse(program), [])

    return pretty(program, self.properties)
