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
CUMULATIVE_VARIABLE = '_cumulative'
BITS_IN_BYTE = 8

__all__ = [
  'generate',
  'Grammar', 'symbol', 'op',
  'Invocation', 'Symbol',
  'GeneratorMachine'
]

import re
C_ARG_DECLARATION_RE = re.compile(r'([^[]*\s+)?(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)(\s*\[.*])?\s*')

def argument_name(declaration):
  matched = C_ARG_DECLARATION_RE.fullmatch(declaration)
  if matched is None:
    raise ValueError(f'{declaration} is not a valid argument declaration')

  return matched.group('name')

class Symbol(object):
  __slots__ = ('name', 'arguments', 'argument_names')

  def __init__(self, name: str, *arguments: str):
    self.name = name
    self.arguments = arguments
    self.argument_names = [argument_name(arg) for arg in arguments]

  def __hash__(self):
    return hash((self.name, *self.arguments))

  def __eq__(self, other):
    if isinstance(other, Symbol):
      return (
        self.name == other.name and
        self.arguments == other.arguments
      )
    else:
      return False

  def __repr__(self):
    return f'symbol {self.name}({", ".join(str(x) for x in self.arguments)})'

  def when(self, *conditions: str) -> 'Condition':
    return Condition(self, *conditions)

  def __call__(self, *args: str | int | float, domain=None, depth=None, **kwargs: str | int | float) -> 'Invocation':
    return Invocation(self, *args, **kwargs, domain=None, depth=None)

  def __add__(self, other):
    if isinstance(other, Symbol):
      return Expansion(self(), other())

    elif isinstance(other, Invocation):
      return Expansion(self(), other)

    elif isinstance(other, Expansion):
      return Expansion(self(), *other.invocations)

    else:
      raise ValueError(
        'expansion should include only instances Invocation or Symbol (eqv. to invocation w/o arguments)'
      )

def symbol(name: str):
  def constructor(*arguments: str | int | float):
    return Symbol(name, *arguments)
  return constructor

def op(name: str):
  return Symbol(name)

class Condition(object):
  __slots__ = ('definition', 'arguments')

  def __init__(self, definition: Symbol, *arguments: str):
    self.definition = definition
    self.arguments = arguments

  def __repr__(self):
    return f'{self.definition.name}.when({", ".join(str(x) for x in self.arguments)})'

  def __hash__(self):
    return hash((self.definition.name, *self.arguments))

  def __eq__(self, other):
    if isinstance(other, Condition):
      return self.definition == other.definition and self.arguments == other.arguments
    else:
      return False

  @property
  def name(self):
      return self.definition.name

class Invocation(object):
  __slots__ = ('definition', 'arguments', 'keyword_arguments', 'domain', 'depth')

  def __init__(
    self, definition: Symbol, *arguments: str | int | float,
    domain: str | None = None,
    depth: str | None = None,
    **keyword_arguments: str | int | float,
  ):
    self.definition = definition
    self.arguments = arguments
    self.keyword_arguments = keyword_arguments
    self.domain = domain
    self.depth = depth

  def __hash__(self):
    kwargs = (
      *(f'{k}={v}' for k, v in self.keyword_arguments.items()),
      *(() if self.domain is None else (f'domain={self.domain}', )),
      *(() if self.depth is None else (f'depth={self.domain}',))
    )
    ### tuple of a tuple for making it compatible with a single-item Expansion
    return hash((
      (self.definition, *self.arguments, *kwargs),
    ))

  def __eq__(self, other):
    if isinstance(other, Invocation):
      return (
        self.definition == other.definition and
        self.arguments == other.arguments and
        self.keyword_arguments == other.keyword_arguments and
        self.domain == other.domain and
        self.depth == other.depth
      )

    else:
      return False


  def __add__(self, other):
    if isinstance(other, Invocation):
      return Expansion(self, other)

    elif isinstance(other, Expansion):
      return Expansion(self, *other.invocations)

    elif isinstance(other, Symbol):
      return Expansion(self, other())

    else:
      raise ValueError(
        'expansion should include only instances Invocation or Symbol (cast into invocation w/o arguments)'
      )

  def __repr__(self):
    args = (
      *(str(x) for x in self.arguments),
      *(f'{k}={v}' for k, v in self.keyword_arguments.items()),
      *(() if self.domain is None else (f'domain={self.domain}',)),
      *(() if self.depth is None else (f'depth={self.domain}',))
    )

    return f'{self.definition.name}({", ".join(args)})'

  @property
  def name(self):
    return self.definition.name

class Expansion(object):
  __slots__ = ('invocations',)

  def __init__(self, *variables: Invocation):
    self.invocations = variables

  def __add__(self, other: Invocation | Symbol):
    if isinstance(other, Symbol):
      return Expansion(*self.invocations, other())

    elif isinstance(other, Invocation):
      return Expansion(*self.invocations, other)

    elif isinstance(other, Expansion):
      return Expansion(*self.invocations, *other.invocations)

    else:
      raise ValueError(
        'expansion should include only instances Invocation or Symbol (cast into invocation w/o arguments)'
      )

  def __iter__(self):
    for sym in self.invocations:
      yield sym

  def __repr__(self):
    return ' + '.join(repr(sym) for sym in self.invocations)

  def __len__(self):
    return len(self.invocations)

  def __hash__(self):
    return hash(tuple(
      (invocation.definition, *invocation.arguments, *(f'{k}={v}' for k, v in invocation.keyword_arguments.items()))
      for invocation in self.invocations
    ))

  def __eq__(self, other):
    if isinstance(other, Expansion):
      return len(self) == len(other) and all(
        self_inv == other_inv for self_inv, other_inv in zip(self, other)
      )

    else:
      return False

TransitionTable: TypeAlias = dict[Expansion | Invocation | Symbol, float] | Expansion | Invocation | Symbol
UpcastedTransitionTable: TypeAlias = dict[Expansion, float | str]

class Grammar(object):
  transitions : dict[Symbol, dict[Condition, UpcastedTransitionTable]]

  def __init__(self, rules: dict[Condition | Symbol, TransitionTable]):
    self.transitions = dict()

    for condition in rules:
      table = dict()

      if isinstance(rules[condition], dict):
        original_table = rules[condition]
      elif rules[condition] is None:
        original_table = {}
      elif isinstance(rules[condition], (Expansion, Invocation, Symbol)):
        original_table = {rules[condition]: 1.0}
      else:
        raise ValueError('transition table can be either dict, a single Expansion/Invocation/Symbol, or None.')

      for expansion, prob in original_table.items():
        if isinstance(expansion, Symbol):
          expansion = Expansion(expansion(), )
        elif isinstance(expansion, Invocation):
          expansion = Expansion(expansion, )
        elif isinstance(expansion, Expansion):
          pass
        else:
          raise ValueError(
            f'Expected either an Expansion (symbol1(...) + symbol2(...)), '
            f'a single Symbol or a single Invocation, got {expansion}.'
          )

        expansion = Expansion(*(
          inv() if isinstance(inv, Symbol) else inv
          for inv in expansion
        ))
        table[expansion] = prob

      if isinstance(condition, Symbol):
        condition = condition.when()
      elif isinstance(condition, Condition):
        pass
      else:
        raise ValueError(
          f'Expected either a Condition (symbol.when(...)) or a Symbol (eqv. to empty condition), got {condition}.'
        )

      definition = condition.definition

      if not isinstance(definition, Symbol):
        raise ValueError(
          f'Expected a Symbol, got {definition}.'
        )

      if definition not in self.transitions:
        self.transitions[definition] = dict()

      self.transitions[definition][condition] = table

  def __repr__(self):
    return '\n'.join(
      '{condition} ->\n{expansions}'.format(
        condition=repr(condition),
        expansions='\n'.join(
          f'    {exp!r} with {p}'
          for exp, p in expansions.items()
        )
      )
      for symbol, rules in self.transitions.items()
      for condition, expansions in self.transitions[symbol].items()
    )

def get_hash(
  assembly: Assembly, grammar: Grammar, seed_symbol: Invocation,
  maximal_number_of_inputs: int, stack_limit: int, debug: bool=False
):
  import hashlib
  algo = hashlib.sha256()

  algo.update(repr(assembly).encode())
  algo.update(b'!')
  algo.update(repr(grammar).encode())
  algo.update(b'!')
  algo.update(repr(seed_symbol).encode())
  algo.update(b'!')
  algo.update(repr(debug).encode())
  algo.update(b'!')
  algo.update(repr(maximal_number_of_inputs).encode())
  algo.update(b'!')
  algo.update(repr(stack_limit).encode())

  return algo.hexdigest()

def argument_declaration(arg: str):
  tokens = [t for t in arg.split(' ') if len(t) > 0]

  *modifiers, name = tokens

  if len(modifiers) > 0:
    return arg
  else:
    return f'int {arg}'

def function_signature(signature: Symbol):
  if len(signature.arguments) > 0:
    arguments = ', '.join(argument_declaration(arg) for arg in signature.arguments)

    return (
      f'static int symgen_expand_{signature.name}'
      f'(pcg32_random_t * rng, InstructionStack * instruction_stack, '
      f'const int depth, const BitSet domain, {arguments})'
    )
  else:
    return (
      f'static int symgen_expand_{signature.name}'
      f'(pcg32_random_t * rng, InstructionStack * instruction_stack, '
      f'const int depth, const BitSet domain)'
    )

def match_arguments(invocation: Invocation, scope: Symbol | None=None):
  signature = invocation.definition
  arguments = dict()

  for value, param in zip(invocation.arguments, signature.argument_names):
    if param in arguments:
      raise ValueError(f'{param} is defined twice!')

    arguments[param] = value

  for param, value in invocation.keyword_arguments.items():
    if param in arguments:
      raise ValueError(f'{param} is defined twice, possibly due to collision of args and kwargs.')

    if param not in signature.argument_names:
      raise ValueError(f'Unknown argument {param} of symbol {signature}!')

    arguments[param] = value

  missing_arguments = set(signature.argument_names) - set(arguments.keys())

  if scope is None:
    if len(missing_arguments) > 0:
      raise ValueError(f'the following arguments are missing: {missing_arguments}')
  else:
    for k in missing_arguments:
      if k in scope.argument_names:
        arguments[k] = k
      else:
        raise ValueError(
          f'unspecified argument {k} (declared as {signature}) '
          f'is not present in the scope {scope}.'
        )

  return [arguments[k] for k in signature.argument_names]

def function_invocation(
  invocation: Invocation, scope: Symbol | None=None,
  rng='rng', stack='instruction_stack', depth='depth - 1', domain='domain'
):
  ### special arguments
  depth = depth if invocation.depth is None else invocation.depth
  domain = domain if invocation.domain is None else invocation.domain

  args = [f'{depth}', f'{domain}']
  for arg in match_arguments(invocation, scope=scope):
    args.append(str(arg))
  args = ', '.join(args)

  return f'symgen_expand_{invocation.definition.name} ({rng}, {stack}, {args})'

def push_instruction(assembly: Assembly, instruction: Invocation):
  if len(instruction.keyword_arguments) + len(instruction.arguments) > 1:
    raise ValueError('Too many arguments for an assembly instruction!')

  if instruction.depth is not None:
    raise ValueError("Assembly instructions don't accept depth argument")

  if instruction.domain is not None:
    raise ValueError("Assembly instructions don't accept domain argument")

  if len(instruction.keyword_arguments) == 1:
    key, = instruction.keyword_arguments.keys()
    value = instruction.keyword_arguments[key]

    if key == 'number':
      arg = f'(arg_t) {{ .number={value} }}'
    elif key == 'integer':
      arg = f'(arg_t) {{ .integer={value} }}'
    else:
      raise ValueError('assembly instructions can only use "number" or "integer" keyword arguments.')

  elif len(instruction.arguments) == 1:
    x, = instruction.arguments
    if isinstance(x, float):
      arg = f'(arg_t) {{ .number={x} }}'
    elif isinstance(x, int):
      arg = f'(arg_t) {{ .integer={x} }}'
    else:
      arg = str(x)
  else:
    arg = '(arg_t) { .integer=0 }'

  return (
    'push_instruction(\n'
    f'        instruction_stack,\n'
    f'        (instruction_t) {{.command = {assembly.symbols[instruction.definition.name]}, .argument={arg} }}\n'
    f'      )'
  )

def generate_function(signature: Symbol, rules: dict[Condition, TransitionTable], assembly: Assembly):
  bodies = list()

  for condition, rule in rules.items():
    if len(rule) == 0:
      bodies.append(
        f'if ({condition}) {{ return STATUS_OK; }}\n'
      )
      continue

    cumulative_floats = 0.0
    cumulative_exprs = []
    for l in rule.values():
      if isinstance(l, (int, float)):
        cumulative_floats += l
      else:
        cumulative_exprs.append(l)

    if len(cumulative_exprs) == 0:
      normalization = cumulative_floats
    else:
      normalization = f'{cumulative_floats} + ' + ' + '.join(f'({expr})' for expr in cumulative_exprs)

    cumulatives: list[str] = []
    if isinstance(normalization, float):
      c = 0.0
      for l in rule.values():
        c += l
        cumulatives.append(str(c / normalization))

      normalization = None
    else:
      for l in rule.values():
        cumulatives.append(f'({CUMULATIVE_VARIABLE} += {l})')

    branches = []
    for i, expansion, c in zip(range(len(rule)), rule.keys(), cumulatives):
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
              f'      status = {function_invocation(product, scope=condition.definition)};\n'
              f'      if (status != STATUS_OK) {{ return status; }};\n'
            )

        if last_product.name in assembly.symbols:
          jumps.append(
            f'      return {push_instruction(assembly, last_product)};'
          )
        else:
          jumps.append(
            f'      return {function_invocation(last_product, scope=condition.definition)};'
          )
        jumps = '\n'.join(jumps)
      else:
        jumps = 'return STATUS_OK;'

      if i < len(rule) - 1:
        branches.append(
          f'if ({BRANCHING_VARIABLE} < {c}) {{\n{jumps}\n    }}'
        )
      else:
        branches.append(
          f' {{\n{jumps}\n    }}'
        )

    transitions = ' else '.join(branches)

    bodies.append(
      'if ({condition}) {{\n'
      '    const double {branchinng_variable} = {normalization}pcg32_uniform(rng);\n'
      '    {comulative_declaration}\n'
      '    {transition}\n'
      '  }}'.format(
        condition=' && '.join(f'({cond})' for cond in condition.arguments) if len(condition.arguments) > 0 else '1',
        transition=transitions,
        comulative_declaration=f'double {CUMULATIVE_VARIABLE} = 0;' if len(cumulative_exprs) > 0 else '// no need for cumulative variable',
        branchinng_variable=BRANCHING_VARIABLE,
        cumulative_variable=CUMULATIVE_VARIABLE,
        normalization='' if normalization is None else f'({normalization}) * ',
      )
    )

  body = ' else '.join(bodies)

  delcr = function_signature(signature)
  definition = (
    f'{delcr} {{\n'
    f'  DEBUG_PRINT("expand {signature.name}\\n");\n'
    f'  if (depth < 0) {{\n'
    f'    return ERROR_MAX_DEPTH;\n'
    f'  }}\n\n'
    f'  {body};\n'
    f'  return ERROR_UNPROCESSED_CONDITION;\n'
    f'}}'
  )

  declaration = f'{delcr};'

  return declaration, definition

def generate(
  assembly: Assembly, grammar: Grammar, seed_symbol: Symbol | Invocation,
  maximal_number_of_inputs: int=1024, stack_limit: int=1024, debug: bool=False
):
  import ctypes

  if isinstance(seed_symbol, Symbol):
    seed_symbol = seed_symbol()

  with open(TEMPLATE, 'r') as f:
    module_template = string.Template(f.read())

  declarations = []
  definitions = []
  for sym, rules in grammar.transitions.items():
    declaration, definition = generate_function(sym, rules, assembly)
    declarations.append(declaration)
    definitions.append(definition)

  code_hash = get_hash(assembly, grammar, seed_symbol, maximal_number_of_inputs, stack_limit, debug=debug)

  ### memory limit is the same as stack limit
  set_limit = max(maximal_number_of_inputs, stack_limit, 1)
  bitmask_size = ctypes.sizeof(ctypes.c_char) * BITS_IN_BYTE
  bit_set_size = (set_limit // bitmask_size) + (0 if set_limit % bitmask_size == 0 else 1)

  module_code = module_template.substitute({
    'DECLARATIONS': '\n'.join(declarations),
    'DEFINITIONS': '\n\n'.join(definitions),
    'DEFINES': '\n'.join(assembly.symbol_defines.values()),
    'SEED_SYMBOL': function_invocation(
      seed_symbol,
      rng='&rng', stack='&instruction_stack', depth='max_depth', domain='all_variables',
      scope=None
    ),
    'HASH': f'"{code_hash}"',
    'DEBUG': '#define SYMGEN_DEBUG' if debug else '// debug off',
    'BIT_SET_SIZE': str(bit_set_size),
    'MEMORY_LIMIT': str(stack_limit)
  })

  return code_hash, module_code

class GeneratorMachine(object):
  def __init__(
    self, *libraries: dict[str, str],
    grammar: Grammar,
    seed_symbol: Symbol | Invocation,
    debug: bool=False,
    maximal_number_of_inputs: int=1024,
    stack_limit: int = 1024,
    shared: bytes | str=None,
    source: bytes | str=None
  ):
    self.assembly: Assembly = Assembly(*libraries)
    self.grammar = grammar

    if isinstance(seed_symbol, Symbol):
      seed_symbol = seed_symbol()

    self.shared, self.machine = compilation.ensure(
      generate=lambda: generate(self.assembly, grammar, seed_symbol, maximal_number_of_inputs, stack_limit, debug=debug),
      get_hash=lambda: get_hash(self.assembly, grammar, seed_symbol, maximal_number_of_inputs, stack_limit, debug=debug),
      name='sym_gen', shared=shared, source=source
    )
    self.maximal_number_of_inputs = maximal_number_of_inputs

  def generate(
    self, seed_1: int, seed_2: int,
    max_inputs: int | None=None,
    max_depth: int=1024,
    instruction_limit: int=1024,
    expression_limit: int=1024,
    max_expression_length: int=128
  ):
    instructions = np.ndarray(shape=(instruction_limit, 2), dtype=np.int32)
    instruction_sizes = np.ndarray(shape=(expression_limit, ), dtype=np.int32)

    if max_inputs is None:
      max_inputs = self.maximal_number_of_inputs

    generated = self.machine.expr_gen(seed_1, seed_2, instructions, instruction_sizes, max_inputs, max_depth, max_expression_length)

    total = np.sum(instruction_sizes[:generated])

    return instructions[:total], instruction_sizes[:generated]



