import string
import os

import numpy as np

from .assembly import Assembly
from . import compilation

ROOT = os.path.dirname(__file__)
TEMPLATE = os.path.join(ROOT, 'sym_eval.template.c')

STACK_KW = 'stack'
ARG_KW = 'argument'
INPUT_KW = 'input'
MEMORY_KW = 'memory'

### template identifier: (signature element, call argument, body identifier)
KEYWORDS = {
  'stack': ('number_t ** stack', '&stack', 'stack'),
  'argument': ('arg_t argument', 'instruction.argument', 'argument'),
  'input': ('const number_t * input', 'expression_input', 'input'),
  'memory': ('number_t * memory', 'memory', 'memory'),
}

__all__ = [
  'StackMachine'
]

def get_hash(library: Assembly, max_stack_size: int, debug: bool):
  import hashlib
  algo = hashlib.sha256()

  algo.update(repr(library).encode())
  algo.update('!'.encode())
  algo.update(repr(max_stack_size).encode())
  algo.update('!'.encode())
  algo.update(repr(debug).encode())

  return algo.hexdigest()

def generate(library: Assembly, max_stack_size: int=1024, debug: bool=False):
  commands = {}
  command_switches = {}

  for k, code in library.ops.items():
    template = string.Template(code)

    used_identifiers = set(template.get_identifiers())
    for identifier in used_identifiers:
      if identifier not in KEYWORDS:
        raise ValueError(
          f'Unknown identifier {identifier}, only the following identifiers are valid: {", ".join(KEYWORDS)}'
        )

    ### stack is always used (possibly implicitly)
    used_identifiers.add(STACK_KW)

    signature = []
    call_arguments = []
    code_substitutes = {}

    for identifier in KEYWORDS:
      if identifier in used_identifiers:
        signature_argument, call_argument, body_expression = KEYWORDS[identifier]
        signature.append(signature_argument)
        call_arguments.append(call_argument)
        code_substitutes[identifier] = body_expression

    signature = ', '.join(signature)
    call_arguments = ', '.join(call_arguments)

    function_name = f'symgen_instruction_{k}'

    command_switches[k] = (f'          case {library.symbols[k]}:\n'
                           f'            {function_name}({call_arguments});\n'
                           f'            break;')

    body = template.substitute(code_substitutes)
    *definitions, return_expression = body.split('\n')

    definitions = ''.join(f'  {line}\n' for line in definitions)

    if 'return' in return_expression or 'return;' in return_expression:
      if not return_expression.endswith(';'):
        return_expression = f'{return_expression};'
      commands[k] = (f'static inline void {function_name}({signature}) {{\n'
                     f'{definitions}'
                     f'  {return_expression}\n'
                     f'}}')
    else:
      commands[k] = (f'static inline void {function_name}({signature}) {{\n'
                      f'{definitions}'
                      f'  const number_t _result = {return_expression};\n'
                      f'  push(stack, _result);\n'
                      f'}}')

  with open(TEMPLATE, 'r') as f:
    module_template = string.Template(f.read())

  all_defines = '\n'.join(d for d in library.symbol_defines.values())
  all_commands = '\n\n'.join(commands.values())

  switch_table = '\n\n'.join((
    *('command switch table', ),
    *(command_switches.values())
  ))

  all_names = ', '.join([
    f'"{name}"'
    for name in library.ops
  ])

  stack_size = f'#define MAXIMAL_STACK_SIZE {max_stack_size}'
  if debug:
    debug_define = '#define SYMGEN_DEBUG'
  else:
    debug_define = '// SYMGEN_DEBUG off'

  machine_hash = get_hash(library, max_stack_size, debug)

  module_code = module_template.substitute({
    'DEFINES': all_defines,
    'COMMANDS': all_commands,
    'COMMAND_SWITCH': switch_table,
    'COMMAND_NAMES': all_names,
    'STACK_SIZE': stack_size,
    'HASH': f'"{machine_hash}"',
    'DEBUG': debug_define
  })

  return machine_hash, module_code

class StackMachine(object):
  def __init__(
    self, *libraries: dict[str, str],
    debug: bool=False,
    shared: bytes | str | None=None,
    source: bytes | str | None=None,
    max_stack_size: int=1024,
  ):
    self.assembly = Assembly(*libraries)

    self.shared, self.machine = compilation.ensure(
      generate=lambda : generate(self.assembly, max_stack_size=max_stack_size, debug=debug),
      get_hash= lambda: get_hash(self.assembly, max_stack_size=max_stack_size, debug=debug),
      name='sym_eval', shared=shared, source=source
    )

  def execute(self, bin_code, sizes, inputs, outputs):
    result = self.machine.stack_eval(bin_code, sizes, inputs, outputs)
    if result != 0:
      raise ValueError('machine has failed to execute')

    return outputs

  def evaluate(self, code, *arguments):
    bin_code = self.assembly.assemble(code)

    sizes = np.array([bin_code.shape[0], ], dtype=np.int32)
    inputs = np.array([[[float(x) for x in arguments]]], dtype=np.float32)
    outputs = np.zeros(shape=(1, 1, 1), dtype=np.float32)

    result = self.machine.stack_eval(bin_code, sizes, inputs, outputs)
    if result != 0:
      raise ValueError('machine has failed to execute')

    return np.reshape(outputs, shape=())



