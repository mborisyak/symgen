from typing import TypeAlias, Sequence
import string
import os

import numpy as np

from .assembly import Assembly
from . import compilation

ROOT = os.path.dirname(__file__)
TEMPLATE = os.path.join(ROOT, 'sym_gen.template.c')

### template identifier: (signature element, call argument, body identifier)
KEYWORDS = {
  'stack': ('number_t ** stack', '&stack', 'stack'),
  'argument': ('arg_t argument', 'instruction.argument', 'argument'),
  'input': ('const number_t * input', 'expression_input', 'input'),
  'memory': ('number_t * memory', 'memory', 'memory'),
}

BRANCHING_VARIABLE = '_branching_random'

__all__ = [
  'generate',
  'Grammar', 'symbol',
  'GeneratorMachine'
]

class Symbol(object):
  __slots__ = ('name', 'arguments', 'keyword_arguments')

  def __init__(self, name: str, *arguments: str, **keyword_arguments):
    self.name = name
    self.arguments = arguments
    self.keyword_arguments = keyword_arguments

  def __add__(self, other):
    return Expansion(self, other)

  def __iter__(self):
    yield self

  def __hash__(self):
    return hash((self.name, *self.arguments))

  def __repr__(self):
    return f'{self.name}({", ".join(str(x) for x in self.arguments)})'

  def __eq__(self, other):
    if not isinstance(other, Symbol):
      return False

    return self.name == other.name and all(
      arg1 == arg2 for arg1, arg2 in zip(self.arguments, other.arguments)
    )

  def __len__(self):
    return 1

def symbol(name: str, *arguments: str | int | float, **kwargs: str | int | float):
  return Symbol(name, *arguments, **kwargs)

def definition(name: str, *arguments: str | int | float):
  assert len(set(arguments)) == len(arguments), 'repeating arguments'

  return Symbol(name, *arguments)

class Expansion(object):
  __slots__ = ('symbols',)

  def __init__(self, *variables: Symbol):
    self.symbols = variables

  def __add__(self, other: Symbol):
    return Expansion(*self.symbols, other)

  def __iter__(self):
    for sym in self.symbols:
      yield sym

  def __repr__(self):
    return ' + '.join(repr(sym) for sym in self.symbols)

  def __len__(self):
    return len(self.symbols)

TransitionTable: TypeAlias = dict[Expansion | Symbol, float]

class Grammar(object):
  def __init__(self, signatures: Sequence[Symbol], transitions: dict[Symbol, TransitionTable]):
    self.signatures = {
      sign.name: sign
      for sign in signatures
    }
    self.transitions = transitions

  def __repr__(self):
    return '{signatures}\n\n{transitions}'.format(
      signatures='\n'.join(repr(sym) for sym in self.signatures),
      transitions='\n'.join(
        '{condition} ->\n{expansions}'.format(
          condition=repr(condition),
          expansions='\n'.join(
            f'    {exp!r} with {p:f}'
            for exp, p in expansions.items()
          )
        )
        for condition, expansions in self.transitions.items()
      )
    )

def get_hash(assembly: Assembly, grammar: Grammar, seed_symbol: Symbol, debug: bool=False):
  import hashlib
  algo = hashlib.sha256()

  algo.update(repr(assembly).encode())
  algo.update(b'!')
  algo.update(repr(grammar).encode())
  algo.update(b'!')
  algo.update(repr(seed_symbol).encode())
  algo.update(b'!')
  algo.update(repr(debug).encode())

  return algo.hexdigest()

def function_signature(signature: Symbol):
  assert len(signature.keyword_arguments) == 0, 'a signature should not contain keyword arguments'

  if len(signature.arguments) > 0:
    arguments = ', '.join(f'int {arg}' for arg in signature.arguments)
    return (
      f'static int symgen_expand_{signature.name}'
      f'(pcg32_random_t * rng, InstructionStack * instruction_stack, int depth, VariableSet input_set, {arguments})'
    )
  else:
    return (
      f'static int symgen_expand_{signature.name}'
      f'(pcg32_random_t * rng, InstructionStack * instruction_stack, int depth, VariableSet input_set)'
    )

def match_arguments(invocation: Symbol, signature: Symbol, strict: bool=False):
  arguments = dict()

  for value, param in zip(invocation.arguments, signature.arguments):
    if param in arguments:
      raise ValueError(f'{param} is defined twice!')

    arguments[param] = value

  for param, value in invocation.keyword_arguments.items():
    if param in arguments:
      raise ValueError(f'{param} is defined twice!')

    arguments[param] = value

  missing_arguments = set(signature.arguments) - set(arguments.keys())

  if strict and len(missing_arguments) > 0:
    raise ValueError(f'the following arguments are missing: {missing_arguments}')

  for k in missing_arguments:
    arguments[k] = k

  return [arguments[k] for k in signature.arguments]

def function_invocation(
  invocation: Symbol, signature: Symbol, rng='rng', stack='instruction_stack', depth='depth - 1',
  input_set='input_set', memory_set='memory_set',
  strict: bool=False
):
  kwargs = {kw: arg for kw, arg in invocation.keyword_arguments.items()}

  ### special arguments
  input_set = kwargs.pop('input_set', input_set)
  depth = kwargs.pop('depth', depth)

  arguments = match_arguments(invocation, signature, strict=strict)

  if len(arguments) > 0:
    args = ', '.join(str(arg) for arg in arguments)
    return (
      f'symgen_expand_{invocation.name} ({rng}, {stack}, {depth}, {input_set}, {args})'
    )
  else:
    return (
      f'symgen_expand_{invocation.name} ({rng}, {stack}, {depth}, {input_set})'
    )

def push_instruction(assembly: Assembly, sym: Symbol):
  if len(sym.arguments) == 0:
    arg = '(arg_t) { .integer=0 }'
  elif len(sym.arguments) == 1:
    x, = sym.arguments
    if isinstance(x, float):
      arg = f'(arg_t) {{ .number={x} }}'
    elif isinstance(x, int):
      arg = f'(arg_t) {{ .integer={x} }}'
    else:
      arg = str(x)
  else:
    raise ValueError('Too many arguments for an assembly instruction!')

  return (
    'push_instruction(\n'
    f'        instruction_stack,\n'
    f'        (instruction_t) {{.command = {assembly.symbols[sym.name]}, .argument={arg} }}\n'
    f'      )'
  )


def generate_function(signature: Symbol, rules: dict[Symbol, TransitionTable], grammar: Grammar, assembly: Assembly):
  transitions = dict()

  for condition, rule in rules.items():
    likelihoods = [l for _, l in rule.items()]
    norm = sum(likelihoods)
    cumulative = {}

    s = 0.0
    for k, l in rule.items():
      s += l
      cumulative[k] = s / norm

    branches = []
    for i, (expansion, c) in enumerate(cumulative.items()):
      if len(expansion) > 0:
        jumps = []
        *intermediate_products, last_product = expansion

        if len(intermediate_products) > 0:
          jumps.append('      int status;')

        for product in intermediate_products:
          if product.name in assembly.symbols:
            jumps.append(
              f'      status = {push_instruction(assembly, product)};\n'
              f'      if (status != STATUS_OK) {{ return status; }};\n'
            )
          else:
            jumps.append(
              f'      status = {function_invocation(product, grammar.signatures[product.name])};\n'
              f'      if (status != STATUS_OK) {{ return status; }};\n'
            )

        if last_product.name in assembly.symbols:
          jumps.append(
            f'      return {push_instruction(assembly, last_product)};'
          )
        else:
          jumps.append(
            f'      return {function_invocation(last_product, grammar.signatures[last_product.name])};'
          )
        jumps = '\n'.join(jumps)
      else:
        jumps = 'return STATUS_OK;'

      if i < len(cumulative) - 1:
        branches.append(
          f'if ({BRANCHING_VARIABLE} < {c}) {{\n{jumps}\n    }}'
        )
      else:
        branches.append(
          f' {{\n{jumps}\n    }}'
        )

    transitions[condition] = ' else '.join(branches)

  body = ' else '.join(
    'if ({condition}) {{\n    {transition}\n  }}'.format(
      condition=' && '.join(f'({cond})' for cond in condition.arguments) if len(condition.arguments) > 0 else '1',
      transition=transitions[condition]
    )
    for condition in rules
  )

  delcr = function_signature(signature)
  definition = (
    f'{delcr} {{\n'
    f'  DEBUG_PRINT("expand {signature.name}\\n");\n'
    f'  if (depth < 0) {{\n'
    f'    return ERROR_MAX_DEPTH;\n'
    f'  }}\n\n'
    f'  const double {BRANCHING_VARIABLE} = uniform(rng);\n'
    f'  {body};\n'
    f'  return ERROR_UNPROCESSED_CONDITION;\n'
    f'}}'
  )

  declaration = f'{delcr};'

  return declaration, definition

def generate(assembly: Assembly, grammar: Grammar, seed_symbol: Symbol, debug: bool=False):
  # for nonterminal, rules in grammar.items():
  #   assert nonterminal not in assembly.symbols, \
  #     f'elementary operation {nonterminal} can be expanded'
  #
  #   for expansion in rules:
  #     for product in expansion:
  #       assert product in grammar or product in assembly.symbols, \
  #         f'token {product} is neither non-terminal nor operation'


  with open(TEMPLATE, 'r') as f:
    module_template = string.Template(f.read())

  declarations = []
  definitions = []
  for _, signature in grammar.signatures.items():
    rules = {
      condition: rule
      for condition, rule in grammar.transitions.items()
      if condition.name == signature.name
    }
    declaration, definition = generate_function(signature, rules, grammar, assembly)
    declarations.append(declaration)
    definitions.append(definition)

  code_hash = get_hash(assembly, grammar, seed_symbol, debug=debug)


  module_code = module_template.substitute({
    'DECLARATIONS': '\n'.join(declarations),
    'DEFINITIONS': '\n\n'.join(definitions),
    'DEFINES': '\n'.join(assembly.symbol_defines.values()),
    'SEED_SYMBOL': function_invocation(
      seed_symbol, signature=grammar.signatures[seed_symbol.name],
      rng='&rng', stack='&instruction_stack', depth='max_depth',
      strict=True
    ),
    'HASH': f'"{code_hash}"',
    'DEBUG': '#define SYMGEN_DEBUG' if debug else '// debug off'
  })

  return code_hash, module_code

class GeneratorMachine(object):
  def __init__(
    self, *libraries: dict[str, str],
    grammar: Grammar,
    seed_symbol: Symbol,
    debug: bool=False,
    shared: bytes | str=None,
    source: bytes | str=None
  ):
    self.assembly: Assembly = Assembly(*libraries)
    self.grammar = grammar

    self.shared, self.machine = compilation.ensure(
      generate=lambda: generate(self.assembly, grammar, seed_symbol, debug=debug),
      get_hash=lambda: get_hash(self.assembly, grammar, seed_symbol, debug=debug),
      name='sym_gen', shared=shared, source=source
    )

  def generate(
    self, seed_1: int, seed_2: int,
    max_inputs: int=1, max_depth: int=10,
    instruction_limit: int=1024, expression_limit: int=1024
  ):
    instructions = np.ndarray(shape=(instruction_limit, 2), dtype=np.int32)
    instruction_sizes = np.ndarray(shape=(expression_limit, ), dtype=np.int32)

    generated = self.machine.expr_gen(seed_1, seed_2, instructions, instruction_sizes, max_inputs, max_depth)

    total = np.sum(instruction_sizes[:generated])

    return instructions[:total], instruction_sizes[:generated]



