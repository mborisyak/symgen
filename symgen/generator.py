from typing import TypeAlias, Callable, Any, Sequence, Iterable
import inspect

import numpy as np
import jax
import jax.numpy as jnp

from .operation import Operation, inspect_op, bind
from .lib import merge
from .dag import Topology, TopologyGenerator, Trivial

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
        break

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
      if isinstance(argument, (tuple, list)):
        return (self.name, *argument)
      else:
        return (self.name, argument)

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

  def seed(self, **arguments) -> 'NonTerminal':
    local_context = get_local_context(self.local, self.local_scopes, arguments)
    return NonTerminal(self, arguments, local_context, self.checks, self.check_scopes)


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
      self.definition, self.condition, self.local, self.local_scopes,
      [*self.checks, *checks], [*self.check_scopes, *scopes]
    )

  def check(self, *contexts):
    if len(self.checks) == 0:
      return True

    local_context = get_local_context(self.local, self.local_scopes, *contexts)

    results = [
      apply_with_scope(check, scope, *contexts, local_context)
      for check, scope in zip(self.checks, self.check_scopes)
    ]

    return all(results)

class NonTerminal(object):
  __slots__ = ('definition', 'parameters', 'local', 'checks', 'check_scopes')

  def __init__(self, definition: Symbol, parameters: dict[str, Any], local: dict[str, Any], checks, check_scopes):
    self.definition = definition
    self.parameters = parameters
    self.local = local

    self.checks = checks
    self.check_scopes = check_scopes

  def check(self, auto_context):
    return all(
      apply_with_scope(check, scope, self.local, self.parameters, auto_context)
      for check, scope in zip(self.checks, self.check_scopes)
    )

  def __repr__(self):
    return f'{self.definition.name}({self.parameters})'

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
      context_updated[k] = apply_with_scope(f, scope, local_context, context, auto_context)

    for k in context:
      if k not in context_updated:
        context_updated[k] = context[k]

    return NonTerminal(
      self.definition, parameters=context_updated, local=local_context,
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
      condition = condition.when().where(**condition.local).assure(*condition.checks)
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

def sample(rng: np.random.Generator, likelihoods):
  norm = sum(likelihoods)
  u = rng.uniform(0, norm)
  c = 0.0
  for i, l in enumerate(likelihoods):
    c += l
    if c > u:
      return i

  return len(likelihoods) - 1

class CheckFailed(Exception):
  pass

def reachable(roots: Iterable[int], deps: Callable[[int], Iterable[int]]) -> set[int]:
  """Cells reachable from `roots` following `deps`."""
  seen: set[int] = set()
  stack = list(roots)
  while stack:
    cell = stack.pop()
    if cell in seen:
      continue
    seen.add(cell)
    stack.extend(deps(cell))
  return seen

def lower(graph: Topology, expressions: dict[int, list[tuple]], n_out: int) -> list[tuple]:
  """Flatten per-node local programs into one stack-machine program, compacting memory cells."""
  n_in = next((i for i, node in enumerate(graph) if node is not None), len(graph))

  remap = {i: i for i in range(n_in)}
  for new_cell, cell in enumerate(sorted(expressions), start=n_in):
    remap[cell] = new_cell

  program: list[tuple] = []

  for cell in sorted(expressions):
    links, _ = graph[cell]
    for op, *args in expressions[cell]:
      if op == 'load':
        program.append(('load', remap[int(links[args[0]])]))
      else:
        program.append((op, *args))
    program.append(('store', remap[cell]))

  for cell in range(len(graph) - n_out, len(graph)):
    program.append(('load', remap[cell]))

  return program

def restore_topology(program: list[tuple]) -> Topology:
  """Reconstruct the topology from a lowered program (inverse of `lower`, best-effort)."""
  nodes: dict[int, tuple[int, ...]] = {}
  body: list[int] = []
  max_cell = -1

  for op, *args in program:
    if op == 'load':
      cell = int(args[0])
      body.append(cell)
      max_cell = max(max_cell, cell)
    elif op == 'store':
      cell = int(args[0])
      max_cell = max(max_cell, cell)
      nodes[cell] = tuple(dict.fromkeys(body))            # distinct links, first-appearance order
      body = []

  return [
    (nodes[cell], {}) if cell in nodes else None
    for cell in range(max_cell + 1)
  ]

class GeneratorMachine(object):
  def __init__(
    self, *libraries: dict[str, Operation],
    rules: dict[Condition | Symbol, TransitionTable],
    topology: 'TopologyGenerator' = Trivial(),
  ):
    library = merge(*libraries)
    self.properties = {
      name: inspect_op(operation)
      for name, operation in library.items()
    }
    self.op_scopes = {
      k: get_scope(op)
      for k, op in library.items()
    }
    self.library = {
      name: (op if 'memory' in self.properties[name][1] else jax.jit(op))
      for name, op in library.items()
    }
    self.grammar = normalize_grammar(rules)
    self.topology = topology

  def __call__(
    self, rng: np.random.Generator, seed: Symbol | Invocation | NonTerminal, *,
    inputs, n_out: int = 1, attempts: int | None = None
  ):
    return self.generate(rng, seed, inputs=inputs, n_out=n_out, attempts=attempts)

  def generate(
    self, rng: np.random.Generator, seed: Symbol | Invocation | NonTerminal, *,
    inputs, n_out: int = 1, attempts: int | None = None
  ):
    """Sample a topology, generate each node's expression into shared memory, and lower to a program."""
    inputs = jnp.asarray(inputs)
    n_in = inputs.shape[0]

    graph = self.topology(rng, n_in, n_out)
    outputs = list(range(len(graph) - n_out, len(graph)))

    links_of = lambda cell: () if graph[cell] is None else graph[cell][0]
    needed = reachable(outputs, links_of)

    memory = [None] * len(graph)                          # between-node memory (list of jax arrays)
    for i in range(n_in):
      memory[i] = inputs[i]                               # seed input cells

    expressions: dict[int, list[tuple]] = {}
    actual_deps: dict[int, set[int]] = {}
    for cell, node in enumerate(graph):
      if node is None or cell not in needed:
        continue                                          # input cell, or a node no output reaches

      links, node_context = node
      local_memory = [memory[g] for g in links]           # copy-in: general memory, local layout

      expression, value = self._generate_node(
        rng, seed, memory=local_memory, node_context=node_context, attempts=attempts
      )
      memory[cell] = value                                # copy-out: commit
      expressions[cell] = expression
      actual_deps[cell] = {links[args[0]] for op, *args in expression if op == 'load'}

    live = reachable(outputs, lambda cell: actual_deps.get(cell, ()))
    program = lower(graph, {cell: expressions[cell] for cell in expressions if cell in live}, n_out)

    return program

  def _generate_node(
    self, rng: np.random.Generator, seed: Symbol | Invocation | NonTerminal, *,
    memory: list, node_context: dict, attempts: int | None = None
  ):
    auto_context = {'rng': rng, 'stack': [], 'memory': memory, **node_context}

    if isinstance(seed, Symbol):
      nonterminal = seed()({}, auto_context)
    elif isinstance(seed, Invocation):
      nonterminal = seed({}, auto_context)
    else:
      nonterminal = seed

    expression, stack, _ = self._generate(
      rng, nonterminal, stack=[], memory=memory, node_context=node_context, attempts=attempts
    )

    assert len(stack) == 1, \
      f'a node expression must net exactly one value, got {len(stack)} for {seed}'

    return expression, stack[-1]

  def _expand_operation(
    self, rng: np.random.Generator, term: Op, context: dict[str, Any], *,
    stack: list, memory: list, node_context: dict, attempts: int | None = None
  ):
    assert term.name in self.library, f'unknown op {term.name}'

    attempts = 1 if attempts is None else attempts

    for _ in range(attempts):
      attempt_stack = stack.copy()
      attempt_memory = memory.copy()
      attempt_autocontext = {'rng': rng, 'stack': attempt_stack, 'memory': attempt_memory, **node_context}

      operation, *operation_args = term(context, attempt_autocontext)

      arity, arguments = self.properties[term.name]
      operands = [attempt_stack.pop() for _ in range(arity)]

      out = self.library[term.name](*operands, **bind(arguments, operation_args, attempt_memory))
      if out is not None:
        attempt_stack.append(out)

      if term.check(context, attempt_autocontext):
        return (operation, *operation_args), attempt_stack, attempt_memory

    raise ValueError('Maximal number of attempts reached.')

  def _generate(
    self, rng: np.random.Generator, seed: NonTerminal, *,
    stack: list, memory: list, node_context: dict, attempts: int | None = None
  ):
    _rng = np.random.default_rng(rng.integers(0, np.iinfo(int).max, size=(4, )))

    stack = [x for x in stack]
    memory = memory.copy()

    auto_context = {'rng': _rng, 'stack': stack, 'memory': memory, **node_context}

    transition_rules = self.grammar[seed.definition.name]
    active_tables = [
      (condition, table)
      for condition, table in transition_rules.items()
      if condition(seed.parameters, auto_context)
    ]

    if len(active_tables) == 0:
      raise ValueError(f'Uncaught condition {seed}.')

    active_rules = [(condition, expansion, prob) for condition, table in active_tables for expansion, prob in table.items()]

    if len(active_rules) == 0:
      return [], stack, memory

    likelihoods = [
      apply_with_scope(prob, scope, seed.parameters, auto_context)
      for _, _, (prob, scope) in active_rules
    ]

    if attempts is None:
      attempts = 1

    for attempt in range(attempts):
      result = []

      index = sample(_rng, likelihoods)
      active_condition, expansion, _ = active_rules[index]

      attempt_stack = stack.copy()
      attempt_memory = memory.copy()

      for term in expansion:
        if isinstance(term, Op):
          assert term.name in self.library, f'unknown op {term.name}'

          op, attempt_stack, attempt_memory = self._expand_operation(
            _rng, term, seed.parameters,
            stack=attempt_stack, memory=attempt_memory, node_context=node_context,
            attempts=attempts
          )
          result.append(op)

        elif isinstance(term, Invocation):
          attempt_auto_context = {'rng': _rng, 'stack': attempt_stack, 'memory': attempt_memory, **node_context}
          nonterminal = term(seed.parameters, attempt_auto_context)

          terms, attempt_stack, attempt_memory = self._generate(
            _rng, seed=nonterminal,
            stack=attempt_stack, memory=attempt_memory, node_context=node_context,
            attempts=attempts
          )
          result.extend(terms)

        else:
          raise ValueError('Improperly normalized transition table!')

      attempt_auto_context = {'rng': _rng, 'stack': attempt_stack, 'memory': attempt_memory, **node_context}

      if seed.check(attempt_auto_context):
        if active_condition.check(seed.parameters, attempt_auto_context):
          return result, attempt_stack, attempt_memory

    raise ValueError('Maximal number of generation attempts reached.')