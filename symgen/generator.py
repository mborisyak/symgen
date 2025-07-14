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

Scope: TypeAlias = Sequence[str] | None

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

  scope = tuple(name for name, p in parameters.items())

  return scope

def apply_with_scope(f, scope: Scope, *contexts):
  if not callable(f):
    return f

  if scope is None:
    return f(**{k: v for context in contexts for k, v in context.items()})

  args = dict()

  for var_name in scope:
    for context in contexts:
      if var_name in context:
        args[var_name] = context[var_name]
        continue

  return f(**args)

def merge_local_definitions(
  local: dict[str, Callable[..., Any]], local_scope: dict[str, Scope],
  additional: dict[str, Callable[..., Any]]
):
  combined = {k: v for k, v in local.items()}
  combined_scopes = {k: v for k, v in local_scope.items()}

  for k in additional:
    if k in combined:
      raise ValueError(f'local context variable {k} is already defined')
    else:
      combined[k] = additional[k]
      combined_scopes[k] = get_scope(additional[k])

  return combined, combined_scopes

def get_local_context(local, local_scopes, *contexts):
  local_context = {}
  for k in local:
    local_context[k] = apply_with_scope(local[k], local_scopes[k], local_context, *contexts)

  return local_context

class Op(object):
  __slots__ = ('name', 'argument', 'scope', 'local', 'local_scopes', 'checks', 'check_scopes')

  def __init__(
    self, name: str, argument: Any,
    local: dict[str, Callable[..., Any]], local_scopes: dict[str, Scope],
    checks: Sequence[Callable[..., bool]], check_scopes: Sequence[Scope]
  ):
    self.name = name
    self.argument = argument
    self.scope = get_scope(argument)
    self.local = local
    self.local_scopes = local_scopes
    self.checks = checks
    self.check_scopes = check_scopes

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

  def __call__(self, *contexts):
    if self.argument is None:
      return (self.name, )
    else:
      local_context = get_local_context(self.local, self.local_scopes, *contexts)

      argument = apply_with_scope(self.argument, self.scope, *contexts, local_context)
      return self.name, argument

  def where(self, **local: Callable[..., Any]):
    merged, merged_scopes = merge_local_definitions(self.local, self.local_scopes, local)
    return Op(self.name, self.argument, merged, merged_scopes, self.checks, self.check_scopes)

  def assure(self, *checks):
    scopes = [get_scope(check) for check in checks]

    return Op(
      self.name, self.argument, self.local, self.local_scopes,
      [*self.checks, *checks], [*self.check_scopes, *scopes]
    )

  def check(self, *contexts):
    local_context = get_local_context(self.local, self.local_scopes, *contexts)

    return all(
      apply_with_scope(check, scope, *contexts, local_context)
      for check, scope in zip(self.checks, self.check_scopes)
    )

class Symbol(object):
  __slots__ = ('name', 'local', 'local_scopes', 'auto_updates', 'auto_update_scopes', 'checks', 'check_scopes')

  def __init__(
    self, name,
    local: dict[str, Callable[..., Any]], local_scopes: dict[str, Scope],
    auto_updates: dict[str, Callable[..., Any]], auto_update_scopes: dict[str, Scope],
    checks: Sequence[Callable[..., bool]], check_scopes: Sequence[Scope]
  ):
    self.name = name

    self.local = local
    self.local_scopes = local_scopes

    self.auto_updates = auto_updates
    self.auto_update_scopes = auto_update_scopes

    self.checks = checks
    self.check_scopes = check_scopes

  def __repr__(self):
    return f'symbol({self.name})'

  def when(self, condition: Callable[..., bool] | None=None) -> 'Condition':
    return Condition(self, condition, {}, {}, self.checks, self.check_scopes)

  def where(self, **local):
    combined, combined_scopes = merge_local_definitions(self.local, self.local_scopes, local)
    return Symbol(
      self.name, combined, combined_scopes,
      self.auto_updates, self.auto_update_scopes,
      self.checks, self.check_scopes
    )

  def assure(self, *checks):
    scopes = [get_scope(check) for check in checks]

    return Symbol(
      self.name, self.local, self.local_scopes,
      self.auto_updates, self.auto_update_scopes,
      [*self.checks, *checks], [*self.check_scopes, *scopes]
    )

  def auto(self, **updates: Callable[..., Any]):
    scopes = {k: get_scope(v) for k, v in updates.items()}

    return Symbol(
      self.name, self.local, self.local_scopes,
      {**self.auto_updates, **updates}, {**self.auto_update_scopes, **scopes},
      self.checks, self.check_scopes
    )

  def __call__(self, **kwargs: Any) -> 'Invocation':
    arguments = {**kwargs}
    scopes = {k: get_scope(v) for k, v in kwargs.items()}

    for k in self.auto_updates:
      if k not in arguments:
        arguments[k] = self.auto_updates[k]
        scopes[k] = self.auto_update_scopes[k]

    return Invocation(self, arguments, scopes, self.local, self.local_scopes, self.checks, self.check_scopes)

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
  return Symbol(name, {}, {}, {}, {}, [], [])

def op(name: str, argument: Any = None):
  return Op(name, argument, {}, {}, [], [])

def condition_str(condition: Callable[..., Any]):
  if not callable(condition):
    return repr(condition)

  params = ', '.join(
    f'**{name}' if p.kind == inspect.Parameter.VAR_KEYWORD else name
    for name, p in inspect.signature(condition).parameters.items()
  )

  if hasattr(condition, '__code__'):
    return f'({params}) -> {condition.__code__}'
  else:
    return f'({params}) -> {hash(condition)}'

class Condition(object):
  __slots__ = ('definition', 'condition', 'scope', 'local', 'local_scopes', 'checks', 'check_scopes')

  def __init__(
    self, definition: Symbol, condition: Callable[..., bool] | None,
    local: dict[str, Callable[..., Any]], local_scopes: dict[str, Scope],
    checks: Sequence[Callable[..., bool]], check_scopes: Sequence[Scope]
  ):
    self.definition = definition
    self.condition = condition
    self.scope = get_scope(condition)

    self.local = local
    self.local_scopes = local_scopes

    self.checks = checks
    self.check_scopes = check_scopes

  def where(self, **local):
    combined, combined_scopes = merge_local_definitions(self.local, self.local_scopes, local)
    return Condition(self.definition, self.condition, combined, combined_scopes, self.checks, self.check_scopes)

  def __repr__(self):
    return f'{self.definition!r}.when({condition_str(self.condition)})'

  @property
  def name(self):
      return self.definition.name

  def __call__(self, *contexts) -> bool:
    if self.condition is None:
      return True
    else:
      local_context = get_local_context(self.local, self.local_scopes, *contexts)
      return apply_with_scope(self.condition, self.scope, local_context, *contexts)

  def assure(self, *checks):
    scopes = [get_scope(check) for check in checks]

    return Condition(
      self.name, self.condition, self.local, self.local_scopes,
      [*self.checks, *checks], [*self.check_scopes, *scopes]
    )

  def check(self, *contexts):
    # if len(self.checks) == 0:
    #   return True

    local_context = get_local_context(self.local, self.local_scopes, *contexts)

    results = [
      apply_with_scope(check, scope, *contexts, local_context)
      for check, scope in zip(self.checks, self.check_scopes)
    ]

    return all(results)

class NonTerminal(object):
  __slots__ = ('definition', 'context', 'local_context', 'checks', 'check_scopes')

  def __init__(self, definition, context, local_context, checks, check_scopes):
    self.definition = definition
    self.context = context
    self.local_context = local_context
    self.checks = checks
    self.check_scopes = check_scopes

  def check(self, *contexts):
    return all(
      apply_with_scope(check, scope, *contexts, self.local_context)
      for check, scope in zip(self.checks, self.check_scopes)
    )

  def __repr__(self):
    return f'{self.definition.name}({self.context, self.local_context})'

class Invocation(object):
  __slots__ = ('definition', 'arguments', 'scopes', 'local', 'local_scopes', 'checks', 'check_scopes')

  def __init__(
    self, definition: Symbol,
    arguments: dict[str, Callable[..., Any] | Any], scopes: dict[str, Scope],
    local: dict[str, Callable[..., Any]], local_scopes: dict[str, Scope],
    checks: Sequence[Callable[..., bool]], check_scopes: Sequence[Scope]
  ):
    self.definition = definition

    self.arguments = arguments
    self.scopes = scopes

    self.local = local
    self.local_scopes = local_scopes

    self.checks = checks
    self.check_scopes = check_scopes

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

  def __call__(self, context: dict[str, Any], auto_context: dict[str, Any]) -> NonTerminal:
    context_updated = {}
    local_context = get_local_context(self.local, self.local_scopes, context, auto_context)

    for k in self.arguments:
      f, scope = self.arguments[k], self.scopes[k]
      context_updated[k] = apply_with_scope(f, scope, context, auto_context, local_context)

    for k in context:
      if k not in context_updated:
        context_updated[k] = context[k]

    return NonTerminal(
      self.definition, context=context_updated, local_context=local_context,
      checks=self.checks, check_scopes=self.check_scopes
    )

  def assure(self, *checks):
    scopes = [get_scope(check) for check in checks]

    return Invocation(
      self.definition, arguments=self.arguments, scopes=self.scopes,
      local=self.local, local_scopes=self.local_scopes,
      checks=[*self.checks, *checks], check_scopes=[*self.check_scopes, *scopes]
    )

  def where(self, **local):
    combined, combined_scopes = merge_local_definitions(self.local, self.local_scopes, local)
    return Invocation(
      self.definition, self.arguments, self.scopes,
      combined, combined_scopes,
      self.checks, self.check_scopes
    )

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
      condition = condition.when().where(**condition.local)
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

    if definition.name not in transitions:
      transitions[definition.name] = dict()

    transitions[definition.name][condition] = table

  return transitions

def sample(rng: random.Random, likelihoods):
  norm = sum(likelihoods)
  u = rng.uniform(0, norm)
  c = 0.0
  for i, l in enumerate(likelihoods):
    c += l
    if c >= u:
      return i

  return len(likelihoods) - 1

class CheckFailed(Exception):
  pass

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
    self, rng: random.Random, seed: Symbol | Invocation | NonTerminal, *,
    inputs: np.ndarray[np.float32] | None=None, attempts: int | None = None
  ):
    return self.generate(rng, seed, inputs=inputs, attempts=attempts)

  def generate(
    self, rng: random.Random, seed: Symbol | Invocation | NonTerminal, *,
    inputs: np.ndarray[np.float32] | None = None,
    attempts: int | None = None
  ):
    if inputs is None:
      stack = None
      memory = None
    else:
      stack = []
      memory = {}

    if isinstance(seed, Symbol):
      seed = seed()({}, {'rng': rng, 'stack': stack, 'memory': memory, 'inputs': inputs})

    elif isinstance(seed, Invocation):
      seed = seed({}, {'rng': rng, 'stack': stack, 'memory': memory, 'inputs': inputs})

    return self._generate(
      rng, seed,
      inputs=inputs, stack=stack, memory=memory,
      attempts=attempts
    )

  def _expand_operation(
    self, rng: random.Random, term: Op, context: dict[str, Any],
    inputs: np.ndarray[np.float32] | None=None,
    stack: list[np.ndarray[np.float32]] | None=None,
    memory: dict[int, np.ndarray[np.float32]] | None = None,
    attempts: int | None=None
  ):
    assert term.name in self.library, f'unknown op {term.name}'

    attempts = 1 if attempts is None else attempts

    for _ in range(attempts):
      attempt_stack = None if stack is None else stack.copy()
      attempt_memory = None if memory is None else memory.copy()
      attempt_autocontext = {'rng': rng, 'stack': attempt_stack, 'memory': attempt_memory, 'inputs': inputs}

      operation, *operation_args = term(context, attempt_autocontext)

      if inputs is not None:
        arity, scope = self.properties[term.name]
        args = [attempt_stack.pop() for _ in range(arity)]

        kwargs = {}
        if 'inputs' in scope:
          kwargs['inputs'] = inputs
        if 'memory' in scope:
          kwargs['memory'] = attempt_memory
        if 'argument' in scope:
          kwargs['argument'], = operation_args

        if 'out' in scope:
          _, *batch = inputs.shape
          out = np.ndarray(shape=batch, dtype=inputs.dtype)
          self.library[term.name](*args, **kwargs, out=out)
          attempt_stack.append(out)
        else:
          out = self.library[term.name](*args, **kwargs)
          if out is not None:
            attempt_stack.append(out)

        # attempt_autocontext['stack'] = attempt_stack
        # attempt_autocontext['memory'] = attempt_memory

      if term.check(context, attempt_autocontext):
        return (operation, *operation_args), attempt_stack, attempt_memory

    raise ValueError('Maximal number of attempts reached.')

  def _generate(
    self, rng: random.Random, seed: NonTerminal, *,
    inputs: np.ndarray[np.float32] | None=None,
    stack: list[np.ndarray[np.float32]] | None=None,
    memory: dict[int, np.ndarray[np.float32]] | None = None,
    attempts: int | None=None
  ):
    _rng = random.Random(rng.getrandbits(16))

    if inputs is not None:
      stack = list() if stack is None else [x for x in stack]
      memory = dict() if memory is None else {k: v for k, v in memory.items()}
    else:
      stack = None
      memory = None

    auto_context = {
      'rng': _rng,
      'inputs': inputs,
      'stack': stack,
      'memory': memory,
    }

    transition_rules = self.grammar[seed.definition.name]
    active_tables = [
      (condition, table)
      for condition, table in transition_rules.items()
      if condition(seed.context, seed.local_context, auto_context)
    ]

    if len(active_tables) == 0:
      raise ValueError(f'Uncaught condition {seed}.')

    active_rules = [(condition, expansion, prob) for condition, table in active_tables for expansion, prob in table.items()]

    if len(active_rules) == 0:
      return [], stack, memory

    likelihoods = [
      apply_with_scope(prob, scope, seed.context, seed.local_context, auto_context)
      for _, _, (prob, scope) in active_rules
    ]

    if attempts is None:
      attempts = 1

    for attempt in range(attempts):
      result = []

      index = sample(_rng, likelihoods)
      active_condition, expansion, _ = active_rules[index]

      attempt_stack = None if stack is None else stack.copy()
      attempt_memory = None if memory is None else memory.copy()

      for term in expansion:
        if isinstance(term, Op):
          assert term.name in self.library, f'unknown op {term.name}'

          op, attempt_stack, attempt_memory = self._expand_operation(
            _rng, term, seed.context, inputs=inputs, stack=attempt_stack, memory=attempt_memory,
            attempts=attempts
          )
          result.append(op)

        elif isinstance(term, Invocation):
          attempt_auto_context = {'rng': _rng, 'inputs': inputs, 'stack': attempt_stack, 'memory': attempt_memory}
          nonterminal = term(seed.context, attempt_auto_context)

          terms, attempt_stack, attempt_memory = self._generate(
            _rng, seed=nonterminal,
            inputs=inputs, stack=attempt_stack, memory=attempt_memory, attempts=attempts
          )
          result.extend(terms)

        else:
          raise ValueError('Improperly normalized transition table!')

      attempt_auto_context = {'rng': _rng, 'inputs': inputs, 'stack': attempt_stack, 'memory': attempt_memory}

      if seed.check(seed.context, attempt_auto_context):
        if active_condition.check(seed.context, attempt_auto_context):
          return result, attempt_stack, attempt_memory

    raise ValueError('Maximal number of generation attempts reached.')