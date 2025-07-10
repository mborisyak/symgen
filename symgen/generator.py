from typing import TypeAlias, Callable, Any
import inspect

import random
import numpy as np

from .lib import merge, Operation

__all__ = [
  'symbol',
  'Invocation', 'Symbol',
  'GeneratorMachine'
]

class Symbol(object):
  __slots__ = ('name', 'arguments', 'argument_names')

  def __init__(self, *arguments: str):
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

  def when(self, condition: Callable[..., bool] | None=None) -> 'Condition':
    signature = inspect.signature(condition)

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for _, p in signature.parameters.items()):
      return Condition(self, condition)
    else:
      assert not any(p.kind == inspect.Parameter.VAR_POSITIONAL in self.arguments for _, p in signature.parameters.items())
      assert all(name in self.arguments for name in signature.parameters)
      return Condition(self, condition)

  def __call__(self, *args: Any, **kwargs: Any) -> 'Invocation':
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

def symbol(*arguments: str):
  return Symbol(*arguments)

def condition_str(condition: Callable[..., Any]):
  params = ', '.join(
    f'**{name}' if p.kind == inspect.Parameter.VAR_KEYWORD else name
    for name, p in inspect.signature(condition).parameters.items()
  )

  if hasattr(condition, '__code__'):
    return f'({params}) -> {condition.__code__.hex()}'
  else:
    return f'({params}) -> {hash(condition)}'

class Condition(object):
  __slots__ = ('definition', 'condition', 'arguments')

  def __init__(self, definition: Symbol, condition: Callable[..., bool] | None):
    self.definition = definition
    self.condition = condition

    if condition is None:
      self.arguments = ()
    else:
      signature = inspect.signature(condition)
      assert all(
        p.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL)
        for _, p in signature.parameters.items()
      )

      if any(p.kind == inspect.Parameter.VAR_KEYWORD for _, p in signature.parameters.items()):
        self.arguments = None
      else:
        self.arguments = tuple(name for name in signature.parameters)

  def __repr__(self):
    return f'{self.definition.name}.when({condition_str(self.condition)})'

  def __hash__(self):
    return hash((self.definition.name, self.condition))

  def __eq__(self, other):
    if isinstance(other, Condition):
      return self.definition == other.definition and self.condition == other.condition
    else:
      return False

  @property
  def name(self):
      return self.definition.name

class Invocation(object):
  __slots__ = ('definition', 'arguments')

  def __init__(self, definition: Symbol, arguments: dict[str, Callable[..., Any] | Any]):
    self.definition = definition
    self.arguments = arguments

  def __hash__(self):
    arguments = ((k, c) for k, c in self.arguments.items())
    return hash(((self.definition, *arguments), ))

  def __eq__(self, other):
    if isinstance(other, Invocation):
      if self.definition != other.definition:
        return False

      if self.arguments.keys() != other.arguments.keys():
        return False

      return all(self.arguments[k] == other.arguments[k] for k in self.arguments)

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

  def __init__(self, *variables: Invocation | Operation):
    self.invocations = variables

  def __add__(self, other: Invocation | Symbol | Operation):
    if isinstance(other, Symbol):
      return Expansion(*self.invocations, other())

    elif isinstance(other, Invocation) or callable(other):
      return Expansion(*self.invocations, other)

    elif isinstance(other, Expansion):
      return Expansion(*self.invocations, *other.invocations)

    else:
      raise ValueError(
        'expansion should include only instances of Invocation or Symbol (cast into invocation w/o arguments)'
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
      (inv.definition, *((k, v) for k, v in inv.arguments.items()))
      for inv in self.invocations
    ))

  def __eq__(self, other):
    if isinstance(other, Expansion):
      return len(self) == len(other) and all(
        self_inv == other_inv for self_inv, other_inv in zip(self, other)
      )

    else:
      return False

TransitionTable: TypeAlias = dict[Expansion | Invocation | Symbol, float | Callable[..., float]] | Expansion | Invocation | Symbol
UpcastedTransitionTable: TypeAlias = dict[Expansion, float | Callable[..., float]]
NormalizedGrammar: TypeAlias = dict[Symbol, dict[Condition, UpcastedTransitionTable]]

def normalize_grammar(rules: dict[Condition | Symbol, TransitionTable]) -> NormalizedGrammar:
  transitions: NormalizedGrammar = dict()

  for condition in rules:
    table: UpcastedTransitionTable = dict()

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

def apply_with_scope(f, **kwargs):
  signature = inspect.signature(f)
  return f(**{k: kwargs[k] for k in signature.parameters})

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
    self.grammar = normalize_grammar(rules)

  def __call__(self, rng: random.Random, seed_symbol: Symbol | Invocation):
    if isinstance(seed_symbol, Symbol):
      seed_symbol = seed_symbol()

    result = []

    transition_rules = self.grammar[seed_symbol.definition]
    active_rules = [
      (expansion, prob)
      for condition, table in transition_rules
      for expansion, prob in table
      if check_condition(**seed_symbol.arguments)
    ]

    likelihoods = [
      apply_with_scope(prob, **seed_symbol.arguments) if callable(prob) else prob
      for _, prob in active_rules
    ]

    index = sample(rng, likelihoods)

    expansion, _ = active_rules[index]

    for invocation in expansion:
      concrete_invocation = invocation.definition(**{
        k: apply_with_scope(v, **seed_symbol.arguments) if callable(v) else v
        for k, v in invocation.arguments
      })

      result.extend(
        self(rng, seed_symbol=concrete_invocation)
      )

    return result



