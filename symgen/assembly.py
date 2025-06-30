import numpy as np

__all__ = [
  'Assembly'
]

### 1 or 1.0 --- float constant (push 1.0 into the stack);
### (1) --- read the 1st input cell and push it into the stack (indexing starts with 0);
### [1] --- read the 1st memory cell and push it into the stack (indexing starts with 0);
### {1} --- pop and store in the 1st memory cell.

CORE_OPERATORS: dict[str, str] = {
  'input': '({integer:d})',
  'memory': '[{integer:d}]',
  'store': '{{{integer:d}}}',
  'const': '{value:g}'
}

PRETTY_CORE_OPERATORS: dict[str, str] = {
  'input': 'x_{integer:d}',
  'memory': 'm_{integer:d}',
  'store': 'm_{integer:d} :=',
  'const': '{value:g}'
}

PRETTY_OPERATORS: dict[str, str] = {
  'add': '({arg1} + {arg2})',
  'mul': '{arg1} * {arg2}',
  'sub': '({arg1} - {arg2})',
  'div': '({arg1})/({arg2})',
  'square': '{arg1}**2',
  'cube': '{arg1}**3',
  'exp': 'exp({arg1})',
  'tanh': 'tanh({arg1})',
  'sigmoid': 'sigma({arg1})',
  'erf': 'erf({arg1})',
  'neg': '(-{arg1})'
}

LATEX_CORE_OPERATORS: dict[str, str] = {
  'input': 'x_{{{integer:d}}}',
  'memory': 'm_{{{integer:d}}}',
  'store': 'm_{{{integer:d}}} :=',
  'const': '{value:g}'
}

LATEX_OPERATORS: dict[str, str] = {
  'add': '\\left({arg1} + {arg2}\\right)',
  'mul': '{arg1} * {arg2}',
  'sub': '\\left({arg1} - {arg2}\\right)',
  'div': '\\frac{{{arg1}}}{{{arg2}}}',
  'square': '\\left({arg1}^2\\right)',
  'cube': '\\left({arg1}^3\\right)',
  'exp': '\\exp\\left({arg1}\\right)',
  'tanh': '\\tanh\\left({arg1}\\right)',
  'sigmoid': '\\sigma\\left({arg1}\\right)',
  'erf': '\\mathrm{{erf}}\\left({arg1}\\right)',
  'neg': '\\left(-{arg1}\\right)'
}

def merge(*libraries: dict[str, str]):
  library = dict()

  for lib in libraries:
    for k in lib:
      k_lower = k.lower()

      if k_lower in library:
        raise ValueError(f'operator {k_lower} is already in the library')

      library[k_lower] = lib[k]

  return library


class Assembly(object):
  def __init__(self, *libraries: dict[str, str]):
    self.ops = merge(*libraries)

    self.symbols = {}
    self.symbol_defines = {}
    self.op_names = {}
    self.op_codes = {}

    for i, op_name in enumerate(self.ops):
      symbol = f'SYMGEN_INSTRUCTION_{op_name.upper()}'
      self.symbols[op_name] = symbol
      self.symbol_defines[op_name] = f'#define {symbol} {i}'
      self.op_names[i] = op_name
      self.op_codes[op_name] = i

    self.core_op_names = {
      self.op_codes[k]: k
      for k in CORE_OPERATORS
    }

  def __repr__(self):
    return '\n\n'.join(
      f'{k}\n{code}'
      for k, code in self.ops.items()
    )

  def assemble(self, code: str):
    instructions = code.split()

    machine_code = np.ndarray(shape=(len(instructions), 2), dtype=np.int32)
    float_view = np.ndarray(shape=(len(instructions), 2), dtype=np.float32, buffer=machine_code)

    for i, instruction in enumerate(instructions):
      if instruction in self.op_codes:
        machine_code[i, 0] = self.op_codes[instruction]
        machine_code[i, 1] = 0
      elif instruction.startswith('(') and instruction.endswith(')'):
        addr = int(instruction[1:-1])
        machine_code[i, 0] = self.op_codes['input']
        machine_code[i, 1] = addr
      elif instruction.startswith('[') and instruction.endswith(']'):
        addr = int(instruction[1:-1])
        machine_code[i, 0] = self.op_codes['memory']
        machine_code[i, 1] = addr
      elif instruction.startswith('{') and instruction.endswith('}'):
        addr = int(instruction[1:-1])
        machine_code[i, 0] = self.op_codes['store']
        machine_code[i, 1] = addr
      else:
        try:
          value = float(instruction)
          machine_code[i, 0] = self.op_codes['const']
          float_view[i, 1] = value
        except ValueError as e:
          raise ValueError(f'instruction is not understood: {instruction}') from e

    return machine_code

  def disassemble(self, code: np.ndarray[np.int32]):
    operators = list()
    float_view = np.ndarray(shape=code.shape, dtype=np.float32, buffer=code)

    for i in range(code.shape[0]):
      op_code = int(code[i, 0])

      if op_code in self.core_op_names:
        op_name = self.core_op_names[op_code]
        operators.append(
          CORE_OPERATORS[op_name].format(value=float_view[i, 1], integer=code[i, 1])
        )
      elif op_code in self.op_names:
        operators.append(self.op_names[op_code])

      else:
        raise ValueError(f'unknown op code {op_code}')

    return ' '.join(operators)

  def format(self, code: np.ndarray[np.int32], core_operators, operators):
    arity = {
      k: v.count('POP()')
      for k, v in self.ops.items()
    }

    float_view = np.ndarray(shape=code.shape, dtype=np.float32, buffer=code)

    stack = []
    for i in range(code.shape[0]):
      op_name = self.op_names[code[i, 0]]

      if op_name in core_operators:
        stack.append(
          core_operators[op_name].format(value=float_view[i, 1], integer=code[i, 1])
        )
      else:
        if op_name in operators:
          arguments = {
            f'arg{i + 1}': stack.pop()
            for i in range(arity[op_name])
          }
          stack.append(operators[op_name].format(**arguments))
        else:
          arguments = ','.join([stack.pop() for _ in range(arity[op_name])])
          stack.append(f'{op_name}({arguments})')

    formatted = '; '.join(stack)
    return formatted

  def pretty(self, code: np.ndarray[np.int32]):
    return self.format(code, PRETTY_CORE_OPERATORS, PRETTY_OPERATORS)

  def latex(self, code: np.ndarray[np.int32]):
    return self.format(code, LATEX_CORE_OPERATORS, LATEX_OPERATORS)




