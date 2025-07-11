from typing import TypeAlias, Callable, Any, Sequence
import inspect

import random
import numpy as np

from .operation import Operation, inspect_op
from .lib import merge

__all__ = [
  'symbol', 'op',
  'Invocation', 'Symbol',
  'GeneratorMachine'
]

Scope: TypeAlias = tuple[Sequence[str], Sequence[str]] | None

def get_scope(f) -> Scope:
  import inspect

  if not callable(f):
    return None

  parameters = inspect.signature(f).parameters
  assert all(p.kind != inspect.Parameter.VAR_POSITIONAL for _, p in parameters.items()), \
    'functions with variable positional arguments are not allowed'

  assert all(p.kind != inspect.Parameter.POSITIONAL_ONLY for _, p in parameters.items()), \
    'functions with positional only arguments are not allowed'

  if any(p.kind == inspect.Parameter.VAR_KEYWORD for _, p in parameters.items()):
    return None

  without_default_value = tuple(name for name, p in parameters.items() if p.default == inspect.Parameter.empty)
  with_default_value = tuple(name for name, p in parameters.items() if p.default != inspect.Parameter.empty)

  return without_default_value, with_default_value

def apply_with_scope(f, scope: Scope, **kwargs):
  if not callable(f):
    return f

  if scope is None:
    return f(**kwargs)

  without_default_value, with_default_value = scope
  args = dict()

  for var_name in without_default_value:
    if var_name not in kwargs:
      raise ValueError(f'missing required scope variable {var_name}')
    args[var_name] = kwargs[var_name]

  for var_name in with_default_value:
    if var_name in kwargs:
      args[var_name] = kwargs[var_name]

  return f(**args)

class Op(object):
  __slots__ = ('name', 'arguments', 'scopes')

  def __init__(self, name: str, *arguments: Any):
    self.name = name
    self.arguments = arguments
    self.scopes = [get_scope(argument) for argument in arguments]


  def __add__(self, other):
    if isinstance(other, Symbol):
      return Expansion(self, other())

    elif isinstance(other, Invocation) or isinstance(other, Op):
      return Expansion(self, other)

    elif isinstance(other, Expansion):
      return Expansion(self, *other.invocations)

    else:
      raise ValueError(
        'expansion should include only instances Invocation or Symbol (eqv. to invocation w/o arguments)'
      )

class Symbol(object):
  __slots__ = ('name', 'arguments', )

  def __init__(self, name, *arguments: str):
    self.name = name
    self.arguments = arguments

  def __repr__(self):
    return f'symbol({self.name})({", ".join(str(x) for x in self.arguments)})'

  def when(self, condition: Callable[..., bool] | None=None) -> 'Condition':
    return Condition(self, condition)

  def __call__(self, *args: Any, **kwargs: Any) -> 'Invocation':
    assert len(args) <= len(self.arguments), 'too many arguments'

    arguments = dict(zip(self.arguments, args))
    for k, v in kwargs.items():
      assert k not in arguments, f'The argument {k} is already passed as a positional argument'
      arguments[k] = v

    return Invocation(self, arguments)

  def __add__(self, other):
    if isinstance(other, Symbol):
      return Expansion(self(), other())

    elif isinstance(other, Invocation) or isinstance(other, Op):
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

def op(name: str, *arguments: Any):
  return Op(name, *arguments)

def condition_str(condition: Callable[..., Any]):
  if not callable(condition):
    return repr(condition)

  params = ', '.join(
    f'**{name}' if p.kind == inspect.Parameter.VAR_KEYWORD else name
    for name, p in inspect.signature(condition).parameters.items()
  )

  if hasattr(condition, '__code__'):
    return f'({params}) -> {condition.__code__.hex()}'
  else:
    return f'({params}) -> {hash(condition)}'

class Condition(object):
  __slots__ = ('definition', 'condition', 'scope')

  def __init__(self, definition: Symbol, condition: Callable[..., bool] | None):
    self.definition = definition
    self.condition = condition
    self.scope = get_scope(condition)

  def __repr__(self):
    return f'{self.definition.name}.when({condition_str(self.condition)})'

  @property
  def name(self):
      return self.definition.name

class Invocation(object):
  __slots__ = ('definition', 'arguments', 'scopes')

  def __init__(self, definition: Symbol, arguments: dict[str, Callable[..., Any] | Any]):
    self.definition = definition
    self.arguments = arguments
    self.scopes = {k: get_scope(v) for k, v in arguments.items()}

  def __add__(self, other):
    if isinstance(other, Invocation):
      return Expansion(self, other)

    elif isinstance(other, Expansion):
      return Expansion(self, *other.invocations)

    elif isinstance(other, Symbol):
      return Expansion(self, other())

    elif isinstance(other, Op):
      return Expansion(self, other)

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

  def __init__(self, *variables: Invocation | Op):
    self.invocations = variables

  def __add__(self, other: Invocation | Symbol | Op):
    if isinstance(other, Symbol):
      return Expansion(*self.invocations, other())

    elif isinstance(other, Invocation) or isinstance(other, Op):
      return Expansion(*self.invocations, other)

    elif isinstance(other, Expansion):
      return Expansion(*self.invocations, *other.invocations)

    else:
      raise ValueError(
        'expansion should include only instances of Invocation, Symbol (cast into invocation w/o arguments) or operation'
      )

  def __iter__(self):
    for sym in self.invocations:
      yield sym

  def __repr__(self):
    return ' + '.join(repr(sym) for sym in self.invocations)

  def __len__(self):
    return len(self.invocations)

ExpansionLike: TypeAlias = Expansion | Invocation | Symbol | Op
TransitionTable: TypeAlias = dict[ExpansionLike, float | Callable[..., float]] | ExpansionLike
UpcastedTransitionTable: TypeAlias = dict[Expansion, tuple[float | Callable[..., float], Scope]]
NormalizedGrammar: TypeAlias = dict[Symbol, dict[Condition, UpcastedTransitionTable]]

def normalize_grammar(rules: dict[Condition | Symbol, TransitionTable]) -> NormalizedGrammar:
  transitions: NormalizedGrammar = dict()

  for condition in rules:
    table: UpcastedTransitionTable = dict()

    if isinstance(rules[condition], dict):
      original_table = rules[condition]
    elif rules[condition] is None:
      original_table = {}
    elif isinstance(rules[condition], (Expansion, Invocation, Symbol, Op)):
      original_table = {rules[condition]: 1.0}
    else:
      raise ValueError('transition table can be either dict, a single Expansion/Invocation/Symbol, or None.')

    for expansion, prob in original_table.items():
      if isinstance(expansion, Op):
        expansion = Expansion(expansion, )
      elif isinstance(expansion, Symbol):
        expansion = Expansion(expansion(), )
      elif isinstance(expansion, Invocation):
        expansion = Expansion(expansion, )
      elif isinstance(expansion, Expansion):
        pass
      else:
        raise ValueError(
          f'Expected either an Expansion (symbol1(...) + symbol2(...)), '
          f'a single Operation, a single Symbol or a single Invocation, got {expansion}.'
        )

      expansion = Expansion(*(
        inv() if isinstance(inv, Symbol) else inv
        for inv in expansion
      ))
      table[expansion] = (prob, get_scope(prob))

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

    if definition not in transitions:
      transitions[definition] = dict()

    transitions[definition][condition] = table

  return transitions

def check_condition(condition, **kwargs):
  if condition.condition is None:
    return True

  elif condition.arguments is None:
    return condition.condition(**kwargs)

  else:
    return condition.condition(*(kwargs[arg] for arg in condition.arguments))

def invoke(
  rng: random.Random,
  invocation: Invocation, local_scope: dict[str, Any], global_scope: dict[str, Any], *,
  inputs: np.ndarray[np.float32], stack: list[np.ndarray[np.float32]] | None, memory: dict[int, np.ndarray[np.float32]]
):
  arguments = {}
  global_scope_updated = {}

  for k in invocation.arguments:
    f, scope = invocation.arguments[k], invocation.scopes[k]
    value = apply_with_scope(f, scope, **local_scope, **global_scope, rng=rng, inputs=inputs, stack=stack, memory=memory)

    if k in global_scope:
      global_scope_updated[k] = value
    else:
      arguments[k] = value

  for k in invocation.definition.arguments:
    if k not in arguments:
      if k in local_scope:
        arguments[k] = local_scope[k]
      else:
        raise ValueError(f'Invocation context does not contain argument {k}.')

  for k in global_scope:
    if k not in global_scope_updated:
      global_scope_updated[k] = global_scope[k]

  return invocation.definition(**arguments), global_scope_updated

def sample(rng: random.Random, likelihoods):
  norm = sum(likelihoods)
  u = rng.uniform(0, norm)
  c = 0.0
  for i, l in enumerate(likelihoods):
    c += l
    if c >= u:
      return i

  return len(likelihoods) - 1


class GeneratorMachine(object):
  def __init__(
    self, *libraries: dict[str, Operation],
    rules: dict[Condition | Symbol, TransitionTable],
  ):
    self.library = merge(*libraries)
    self.properties = {
      name: inspect_op(operation)
      for name, operation in self.library.items()
    }
    self.grammar = normalize_grammar(rules)
    self.op_scopes = {
      k: get_scope(op)
      for k, op in self.library.items()
    }

  def __call__(
    self, rng: random.Random, seed_symbol: Symbol | Invocation, *,
    trace: np.ndarray[np.float32] | None=None,
    stack: list[np.ndarray[np.float32]] | None=None,
    memory: dict[int, np.ndarray[np.float32]] | None = None,
    **global_scope
  ):
    if isinstance(seed_symbol, Symbol):
      seed_symbol = seed_symbol()

    result = []
    if trace is not None:
      stack = list() if stack is None else [x for x in stack]
      memory = dict() if memory is None else {k: v for k, v in memory.items()}
    else:
      stack = None
      memory = None

    transition_rules = self.grammar[seed_symbol.definition]
    active_tables = [
      table
      for condition, table in transition_rules.items()
      if condition.condition is None or apply_with_scope(
        condition.condition, condition.scope, **seed_symbol.arguments, **global_scope,
        rng=rng, inputs=trace, stack=stack, memory=memory
      )
    ]

    if len(active_tables) == 0:
      raise ValueError(f'Uncaught condition {seed_symbol} (global: {global_scope}).')

    active_rules = [(expansion, prob) for table in active_tables for expansion, prob in table.items()]

    if len(active_rules) == 0:
      return [], stack, memory

    likelihoods = [
      apply_with_scope(prob, scope, **seed_symbol.arguments, **global_scope, rng=rng, stack=stack, memory=memory) if callable(prob) else prob
      for _, (prob, scope) in active_rules
    ]

    index = sample(rng, likelihoods)

    expansion, _ = active_rules[index]

    for term in expansion:
      if isinstance(term, Op):
        assert term.name in self.library, f'unknown op {term.name}'
        arguments = [
          apply_with_scope(
            v, scope, **seed_symbol.arguments, **global_scope,
            rng=rng, stack=stack, memory=memory, inputs=trace
          )
          for v, scope in zip(term.arguments, term.scopes)
        ]
        result.append((term.name, *arguments))

        if trace is not None:
          arity, scope = self.properties[term.name]
          args = [stack.pop() for _ in range(arity)]

          kwargs = {}
          if 'inputs' in scope:
            kwargs['inputs'] = trace
          if 'memory' in scope:
            kwargs['memory'] = memory
          if 'argument' in scope:
            kwargs['argument'], = arguments

          if 'out' in scope:
            _, *batch = trace.shape
            out = np.ndarray(shape=batch, dtype=np.float32)
            self.library[term.name](*args, **kwargs, out=out)
            stack.append(out)
          else:
            out = self.library[term.name](*args, **kwargs)
            if out is not None:
              stack.append(out)
      else:
        concrete_invocation, global_scope_updated = invoke(
          rng, term, seed_symbol.arguments, global_scope,
          inputs=trace, stack=stack, memory=memory
        )

        terms, stack, memory = self(
          rng, seed_symbol=concrete_invocation, **global_scope_updated,
          trace=trace, stack=stack, memory=memory
        )
        result.extend(terms)

    return result, stack, memory