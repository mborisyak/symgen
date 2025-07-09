from typing import TypeAlias, Sequence, Callable, Any
import inspect
import string

import numpy as np

__all__ = [
  'generate',
  'Grammar', 'symbol', 'op',
  'Invocation', 'Symbol',
  'GeneratorMachine'
]

class Symbol(object):
  __slots__ = ('name', 'arguments', 'argument_names')

  def __init__(self, name: str, *arguments: str):
    self.name = name
    self.arguments = arguments

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

  def when(self, condition) -> 'Condition':
    signature = inspect.signature(condition)

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for _, p in signature.parameters.items()):
      return Condition(self, condition)
    else:
      assert not any(p.kind == inspect.Parameter.VAR_POSITIONAL in self.arguments for _, p in signature.parameters.items())
      assert all(name in self.arguments for name in signature.parameters)
      return Condition(self, condition)

  def __call__(self, *args, **kwargs) -> 'Invocation':
    assert len(args) <= len(self.arguments), 'too many arguments'

    arguments = dict(zip(args, self.arguments))
    for k, v in kwargs.items():
      assert k not in arguments, f'The argument {k} is already passed as a positional argument'
      arguments[k] = v

    return Invocation(self, arguments)

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
  def constructor(*arguments: str):
    return Symbol(name, *arguments)
  return constructor

def op(name: str):
  return Symbol(name)

def condition_str(condition: Callable[..., Any]):
  params = ', '.join(
    f'**{name}' if p.kind == inspect.Parameter.VAR_KEYWORD else name
    for name, p in inspect.signature(condition).parameters.items()
  )

  return f'({params}) -> {condition.__code__.hex()}'

class Condition(object):
  __slots__ = ('definition', 'condition')

  def __init__(self, definition: Symbol, condition: callable):
    self.definition = definition
    self.condition = condition

  def __repr__(self):
    return f'{self.definition.name}.when({condition_str(self.condition)})'

  def __hash__(self):
    code = self.condition.__code__
    return hash((self.definition.name, code.co_code, code.co_consts, code.co_names, code.co_varnames))

  def __eq__(self, other):
    if isinstance(other, Condition):
      return self.definition == other.definition and conditions_eq(self.condition, other.condition)
    else:
      return False

  @property
  def name(self):
      return self.definition.name

class Invocation(object):
  __slots__ = ('definition', 'arguments')

  def __init__(self, definition: Symbol, arguments: dict[str, Callable[..., Any]]):
    self.definition = definition
    self.arguments = arguments

  def __hash__(self):
    arguments = (
      (k, c.__code__.co_code, c.__code__.co_consts, c.__code__.co_names, c.__code__.co_varnames)
      for k, c in self.arguments.items()
    )
    return hash(((self.definition, *arguments), ))

  def __eq__(self, other):
    if isinstance(other, Invocation):
      if self.definition != other.definition:
        return False

      if self.arguments.keys() != other.arguments.keys():
        return False

      return all(conditions_eq(self.arguments[k], other.arguments[k]) for k in self.arguments)

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
    args = ','.join(
      f'{k}={condition_str(v)}' for k, v in self.arguments.items()
    )

    return f'{self.definition.name}({args})'

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
    return hash(tuple(self.invocations))

  def __eq__(self, other):
    if isinstance(other, Expansion):
      return len(self) == len(other) and all(
        self_inv == other_inv for self_inv, other_inv in zip(self, other)
      )

    else:
      return False

TransitionTable: TypeAlias = dict[Expansion | Invocation | Symbol, float | Callable[..., float]] | Expansion | Invocation | Symbol
UpcastedTransitionTable: TypeAlias = dict[Expansion, float | Callable[..., float]]

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
  invocation: Invocation, scope: Symbol | None=None, rng='rng', stack='stack', depth='depth - 1', domain='domain',
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
    f'        (instruction_t) {{.command = {assembly.symbols[instruction.definition.name]}, .argument={arg} }},\n'
    f'        stack\n'
    f'      )'
  )

def generate_expansion(expansion: Expansion, scope: Symbol, assembly: Assembly):
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
          f'      status = {function_invocation(product, scope=scope)};\n'
          f'      if (status != STATUS_OK) {{ return status; }};\n'
        )

    if last_product.name in assembly.symbols:
      jumps.append(
        f'      return {push_instruction(assembly, last_product)};'
      )
    else:
      jumps.append(
        f'      return {function_invocation(last_product, scope=scope)};'
      )
    jumps = '\n'.join(jumps)
  else:
    jumps = 'return STATUS_OK;'

  return jumps

def get_total_likelihood(rule):
  total_floats = 0.0
  exprs = []

  for l in rule.values():
    if isinstance(l, (int, float)):
      total_floats += l
    else:
      exprs.append(l)

  if len(exprs) == 0:
    normalization = total_floats
  else:
    normalization = f'{total_floats} + ' + ' + '.join(f'({expr})' for expr in exprs)

  return normalization

def generate_op(op: str, assembly: Assembly, maximal_arity: int):
  from string import Template
  code = Template(assembly.ops[op]).substitute({
    'argument': 'argument',
    'memory': 'memory[argument.integer * n_trace_samples + i]',
    'input': 'inputs[argument.integer * n_trace_samples + i]',
  })


  arguments = [f'arg_{i}[i]' for i in range(maximal_arity)]
  terms = code.split('POP()')

  if len(terms) > maximal_arity + 1:
    raise ValueError('The function consumes more arguments than max. arity!')

  substituted = list()
  for term, arg in zip(terms[:-1], arguments):
    substituted.append(term)
    substituted.append(arg)

  substituted.append(terms[-1])
  substituted = ''.join(substituted)

  *definitions, return_expression = substituted.split('\n')
  definitions = ''.join(f'    {line.strip()}\n' for line in definitions)

  if 'return' in return_expression:
    body = (
      f'  for (int i = 0; i < n_trace_samples; ++i) {{\n'
      f'    {definitions}\n'
      f'    {return_expression};\n'
      f'  }}'
    )
  else:
    body = (
      f'  for (int i = 0; i < n_trace_samples; ++i) {{\n'
      f'{definitions}\n'
      f'    output[i] = {return_expression};\n'
      f'  }}'
    )

  signature = (
    'static void symgen_{op}(\n'
    '  arg_t argument, {args}, \n'
    '  number_t * output, number_t * inputs, number_t * memory, \n'
    '  unsigned int n_trace_samples\n'
    ')'
  ).format(
    op=op,
    args=', '.join([f'number_t * arg_{i}' for i in range(maximal_arity)]),
  )

  return f'{signature}{{\n{body}\n}}'


def generate_function(signature: Symbol, rules: dict[Condition, TransitionTable], assembly: Assembly):
  condition_total_likelihoods = dict()
  for condition, rule in rules.items():
    condition_total_likelihoods[condition] = get_total_likelihood(rule)

  condition_variables = {}
  for i, condition in enumerate(rules):
    if len(condition.arguments) > 0:
      condition_variables[f'{CONDITION_VARIABLE}_{i}'] = ' && '.join(f'({cond})' for cond in condition.arguments)
    else:
      condition_variables[f'{CONDITION_VARIABLE}_{i}'] = '1'

  conditional_clauses = list()

  for i, (condition, rule) in enumerate(rules.items()):
    transitions = list()

    for expansion, likelihood in rule.items():
      expansion = generate_expansion(expansion, scope=condition.definition, assembly=assembly)
      transitions.append(
        f'if ({BRANCHING_VARIABLE} <= ({CUMULATIVE_VARIABLE} += {likelihood})) {{\n'
        f'{expansion}\n'
        f'    }}'
      )


    conditional_clauses.append(
      '  if ({condition}) {{\n'
      '    {transitions}\n'
      '  }}'.format(
        condition=f'{CONDITION_VARIABLE}_{i}',
        transitions=' else '.join(transitions),
        branchinng_variable=BRANCHING_VARIABLE,
        cumulative_variable=CUMULATIVE_VARIABLE,
        condition_likelihood=str(condition_total_likelihoods[condition]),
      )
    )

  body = '\n\n'.join(conditional_clauses)
  condition_computation = '\n'.join(
    f'  const int {k} = {v};' for k, v in condition_variables.items()
  )
  condition_likelihoods = '\n'.join(
    f'  const double {TOTAL_LIKELIHOOD_VARIABLE}_{i} = {condition_total_likelihoods[condition]};'
    for i, condition in enumerate(rules)
  )

  total_likelihood_declr = 'const double {variable} = {total};'.format(
    variable=TOTAL_LIKELIHOOD_VARIABLE,
    total=' + '.join(
      f'({CONDITION_VARIABLE}_{i} ? {TOTAL_LIKELIHOOD_VARIABLE}_{i} : 0.0)'
      for i, _ in enumerate(rules)
    )
  )

  delcr = function_signature(signature)
  definition = (
    f'{delcr} {{\n'
    f'  DEBUG_PRINT("expand {signature.name}\\n");\n'
    f'{condition_computation}\n'
    f'{condition_likelihoods}\n'
    f'  {total_likelihood_declr}\n'
    f'  if ({TOTAL_LIKELIHOOD_VARIABLE} <= 0.0) {{ return ERROR_UNPROCESSED_CONDITION; }}\n\n'
    f'  double {CUMULATIVE_VARIABLE} = 0.0;\n'
    f'  const double {BRANCHING_VARIABLE} = ({TOTAL_LIKELIHOOD_VARIABLE}) * pcg32_uniform(rng);\n'
    f'  DEBUG_PRINT("r = %.3lf from %.3lf\\n", {BRANCHING_VARIABLE}, {TOTAL_LIKELIHOOD_VARIABLE});\n'
    f'  if (depth < 0) {{\n'
    f'    return ERROR_MAX_DEPTH;\n'
    f'  }}\n\n'
    f'{body};\n'
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

  arities = {k: v.count('POP()') for k, v in assembly.ops.items()}
  max_arity = max(arities.values())

  common_op_type = 'typedef void (*op_t)(arg_t, {inputs}, number_t *, number_t *, number_t *, unsigned int);'.format(
    inputs=', '.join('number_t *' for _ in range(max_arity))
  )

  op_table = 'const static op_t ops[{n}] = {{\n{ops}\n}};'.format(
    n=len(assembly.ops),
    ops=',\n'.join(f'  symgen_{op}' for op in assembly.ops)
  )

  arities_declr = 'const int arities[] = {{{array}}};'.format(
    array=', '.join([str(arity) for _, arity in arities.items()])
  )

  outputs = 'const int number_of_outputs[] = {{{array}}};'.format(
    array=', '.join(['0' if op == 'store' else '1' for op in assembly.ops])
  )

  op_functions = '\n\n'.join(
    generate_op(op, assembly, max(arities.values()))
    for op in assembly.ops
  )

  module_code = module_template.substitute({
    'DECLARATIONS': '\n'.join(declarations),
    'DEFINITIONS': '\n\n'.join(definitions),
    'DEFINES': '\n'.join(assembly.symbol_defines.values()),
    'ARITIES': arities_declr,
    'MAX_ARITY': str(max_arity),
    'COMMON_OP_TYPE': common_op_type,
    'OP_TABLE': op_table,
    'NUMBER_OF_OUTPUTS': outputs,
    'SEED_SYMBOL': function_invocation(
      seed_symbol, rng='&rng', stack='&stack', depth='max_depth', domain='all_variables', scope=None
    ),
    'OPS': op_functions,
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
    max_expression_length: int=128,
    trace_samples: np.ndarray[np.float32] | int=32,
  ):
    instructions = np.ndarray(shape=(instruction_limit, 2), dtype=np.int32)
    instruction_sizes = np.ndarray(shape=(expression_limit, ), dtype=np.int32)

    if max_inputs is None:
      max_inputs = self.maximal_number_of_inputs

    if isinstance(trace_samples, int):
      rng = np.random.RandomState(seed_1)
      trace_samples = np.ndarray(shape=(max_inputs, trace_samples), dtype=np.float32)
      trace_samples[:] = rng.normal(size=trace_samples.shape)

    assert trace_samples.shape[0] == max_inputs

    trace_buffer = np.ndarray(shape=(instruction_limit, trace_samples.shape[1]), dtype=np.float32)

    generated = self.machine.expr_gen(
      seed_1, seed_2, instructions, instruction_sizes, max_inputs, max_depth, max_expression_length,
      trace_samples, trace_buffer,
    )

    total = np.sum(instruction_sizes[:generated])

    return instructions[:total], instruction_sizes[:generated]



