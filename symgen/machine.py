import string

import os

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
  'compile',
  'link'
]

def get_hash(library: dict[str, str]):
  import hashlib
  hash = hashlib.sha256()

  for k, code in library.items():
    hash.update(k.encode('utf-8'))
    hash.update('|'.encode('utf-8'))
    hash.update(code.encode('utf-8'))

  return hash.hexdigest()

def generate(*libraries: dict[str, str], output: bytes | str | None = None):
  library = dict()

  for lib in libraries:
    for k in lib:
      k_lower = k.lower()

      if k_lower in library:
        raise ValueError(f'operator {k_lower} is already in the library')

      library[k_lower] = lib[k]

  order = dict((k, i) for i, k in enumerate(library.keys()))

  instruction_symbols = {}
  instruction_symbol_defines = {}
  for k, i in order.items():
    instruction_symbols[k] = f'SYMGEN_INSTRUCTION_{k.upper()}'
    instruction_symbol_defines[k] = f'#define SYMGEN_INSTRUCTION_{k.upper()} {i}'

  commands = {}
  command_switches = {}

  for k, code in library.items():
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

    command_switches[k] = (f'          case {instruction_symbols[k]}:\n'
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

  all_defines = '\n'.join((
    *('defines', ),
    *(d for d in instruction_symbol_defines.values())
  ))

  all_commands = '\n\n'.join((
    *('commands', ),
    *(commands.values())
  ))

  switch_table = '\n\n'.join((
    *('command switch table', ),
    *(command_switches.values())
  ))

  all_names = ', '.join([
    f'"{name}"'
    for name in library
  ])

  module_code = module_template.substitute({
    'DEFINES': all_defines,
    'COMMANDS': all_commands,
    'COMMAND_SWITCH': switch_table,
    'COMMAND_NAMES': all_names
  })

  hash = get_hash(library)

  if output is None:
    output = f'sym_eval-{hash}.c'

  with open(output, 'w') as f:
    f.write(module_code)

  return hash, output

LIB_FLAGS = ['--shared', '-fPIC']
OPT_FLAGS = ['-O3', '-march=native', '-mtune=native']

def compile(*libraries: dict[str, str], output: bytes | str | None=None, source: bytes | str | None=None, debug: bool=False):
  import sysconfig
  import subprocess as sp
  import tempfile

  import numpy as np

  if source is None:
    _, source_file = tempfile.mkstemp(suffix='.c')
  else:
    source_file = source

  try:
    hash, source_file = generate(*libraries, output=source_file)

    include_dirs = [*sysconfig.get_paths()['include'].split(' '), np.get_include()]
    include_dirs = [f'-I{l}' for l in include_dirs]
    cflags = sysconfig.get_config_vars().get('CFLAGS').split(' ')
    ldflags = sysconfig.get_config_vars().get('LDFLAGS').split(' ')
    debug = ('-DDEBUG', ) if debug else ()

    if output is None:
      suffix = sysconfig.get_config_vars().get('EXT_SUFFIX')
      output = f'sym_eval-{hash}{suffix}'

    print([source_file, *LIB_FLAGS, *OPT_FLAGS, *include_dirs, *cflags, *ldflags, *debug, '-o', output])

    process = sp.Popen(
      executable='gcc',
      args=['gcc', source_file, *LIB_FLAGS, *OPT_FLAGS, *include_dirs, *ldflags, *debug, '-o', output],
      stdout=sp.PIPE, stderr=sp.PIPE, stdin=None
    )

    stdout, stderr = process.communicate()
    if process.returncode != 0:
      print('===== stdout =====')
      print(stdout.decode('utf-8'))
      print('===== stderr =====')
      print(stderr.decode('utf-8'))
      raise Exception('Compilation failed')

  finally:
    if source is None:
      os.remove(source_file)

  return output

def link(shared: bytes | str | None):
  import importlib.util
  spec = importlib.util.spec_from_file_location("sym_eval", shared)
  mymodule = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mymodule)

  return mymodule