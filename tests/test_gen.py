import random
from typing import NamedTuple
import math

import numpy as np
from treescope import display

import symgen

def test_dsl():
  from symgen.generator import symbol, op, Condition

  s = symbol('s')

  invocation = s(k=lambda i, j: i + j).where(j=lambda i: i + 1)
  result = invocation({'i': 1}, {'i': 1})
  print(result)
  assert set(result.arguments.keys()) == {'i', 'k'}
  assert result.arguments['i'] == 1
  assert result.arguments['k'] == 3

  condition = s.when(lambda k: k == 5).where(j=lambda i: 2 * i).where(k=lambda j: j + 1)
  assert isinstance(condition, Condition)
  assert condition({'i': 2})
  assert not condition({'i': 1})
  assert not condition({'i': 3})


  stack = [
    np.random.normal(size=(128, ))
  ]
  result = op('mul', lambda stack: 1 / np.std(stack[-1]))({'stack' : stack})
  assert len(result) == 2
  assert result[0] == 'mul'
  assert np.abs(result[1] - 1 / np.std(stack[0])) < 1.0e-3

  assert s.assure(lambda stack: np.std(stack) < 2.0)(stack=stack).check(stack=stack)
  assert not s.assure(lambda stack: np.std(stack) < 0.1)(stack=stack).check(stack=stack)

  assert s.assure(lambda std: std < 2.0).where(std=lambda stack: np.std(stack))(stack=stack).check(stack=stack)
  assert not s.assure(lambda std: std < 0.1).where(std=lambda stack: np.std(stack))(stack=stack).check(stack=stack)

  s_auto = s.auto(i=lambda i, j: i + j + 1)

  assert s_auto(j=lambda j: j + 1)(None, i=1, j=10).arguments['i'] == 12
  assert s_auto(j=lambda j: j + 1, i=lambda i, j: i + 2 * j + 2)(None, i=1, j=10).arguments['i'] == 23

def test_auto():
  import symgen
  from symgen.generator import GeneratorMachine, symbol, op

  lib = symgen.lib.merge(symgen.lib.core, symgen.lib.std)

  limit = 3

  expr = symbol('expr').auto(depth=lambda depth: depth + 1)

  rules = {
    expr.when(lambda depth: depth < limit): expr + expr + op('add'),
    expr.when(lambda depth: depth >= limit): op('variable', 0),
  }

  generator = symgen.GeneratorMachine(lib, rules=rules)

  expression = generator(random.Random(1234), expr(depth=0))

  print(expression)

  assert len(expression) == 2 ** (limit + 1) - 1

def test_local():
  import symgen
  from symgen.generator import GeneratorMachine, symbol, op

  lib = symgen.lib.merge(symgen.lib.core, symgen.lib.std)
  limit = 3

  expr = symbol('expr').where(t1=lambda i: 2 * i + 1).auto(counter=lambda counter: counter + 1)
  rules = {
    expr.when(lambda i: i < limit): expr(i=lambda t1, t2: t1 + t2).where(t2=lambda t1, i: t1 - 3 * i - 1),
    expr.when(lambda t1: t1 >= 2 * limit + 1): op('const', lambda t1, counter: counter + t1),
  }

  generator = symgen.GeneratorMachine(lib, rules=rules)
  expression = generator(random.Random(1234), expr(i=0, counter=0))

  assert len(expression) == 1
  (_, arg), = expression

  assert arg == limit + 2 * limit + 1

def test_fibonacci():
  from symgen.generator import symbol
  s = symbol('s')

  inv = s(i=lambda i, j: i + j, j=lambda i: i)
  state = s(i=1, j=1)

  for _ in range(10):
    print(state)
    state = inv(None, **state.arguments)

def test_context_passing():
  from symgen.generator import symbol
  s = symbol('s')

  inv = s(i=lambda i, j: i + j)
  state = dict(i=1, j=2)
  state = inv(dict(i=1, j=2), defaults=state)

  assert state.arguments['i'] == 3
  assert state.arguments['j'] == 2

def test_tracing():
  from symgen.generator import GeneratorMachine, symbol, op

  def display(*, argument):
    means, stds, memory_means, memory_stds = argument
    stack_ = ', '.join(f'{m:.2f} +- {s:.2f}' for m, s in zip(means, stds))
    memory_ = ', '.join(f'{k}: {memory_means[k]:.2f} +- {memory_stds[k]:.2f}' for k in memory_means)
    print(f'[{stack_}], {{{memory_}}}')

  def statistics(means, stds, memory_means, memory_stds):
    return means, stds, memory_means, memory_stds

  debug_lib = {
    'tracer': display
  }

  expr = symbol('expr').where(
    means=lambda stack: [float(np.mean(v)) for v in stack],
    stds=lambda stack:  [float(np.std(v)) for v in stack],
    memory_means = lambda memory: {k: float(np.mean(v)) for k, v in memory.items()},
    memory_stds = lambda memory: {k: float(np.std(v)) for k, v in memory.items()},
  )
  inspect = op('tracer', statistics)

  rules = {
    expr: (
      inspect + op('variable', 0) + inspect + op('variable', 1) + inspect +
      op('store', 1) + inspect + op('store', 0) + inspect +
      op('load', 1) + inspect + op('load', 0) + inspect + op('add') + inspect
    )
  }

  inputs = np.random.normal(size=(2, 1024))
  inputs[0] *= 2
  inputs[1] *= 0.5

  inputs[0] -= 1
  inputs[1] += 1.5

  generator = GeneratorMachine(symgen.lib.std, symgen.lib.core, debug_lib, rules=rules)
  expression = generator(random.Random(1), expr(), inputs=inputs)

  means = [term[1][0] for term in expression if term[0] == 'tracer']
  stds = [term[1][1] for term in expression if term[0] == 'tracer']
  mmeans = [term[1][2] for term in expression if term[0] == 'tracer']
  mstds = [term[1][3] for term in expression if term[0] == 'tracer']

  assert [len(x) for x in means] == [0, 1, 2, 1, 0, 1, 2, 1]
  assert [len(x) for x in stds] == [0, 1, 2, 1, 0, 1, 2, 1]
  assert [len(x) for x in mmeans] == [0, 0, 0, 1, 2, 2, 2, 2]
  assert [len(x) for x in mstds] == [0, 0, 0, 1, 2, 2, 2, 2]

def test_invocation():
  from symgen.generator import symbol
  s = symbol('s')

  inv = s(i=2, j=3, c=4)

  print(inv({}, {}))

  print(s()(dict(i=2, j=3, c=4), {} ))

def test_scope():
  import symgen
  from symgen.generator import GeneratorMachine, symbol, op

  lib = symgen.lib.merge(symgen.lib.core, symgen.lib.std)

  expr = symbol('expr')
  unop = symbol('unop')
  biop = symbol('biop')
  constant = symbol('constant')

  def subsample(rng: random.Random, domain):
    n = rng.randint(1, len(domain))
    return rng.sample(domain, n)


  rules = {
    expr.when(lambda i, domain, stack: i > 0 and len(domain) > 1): {
      expr(i=lambda i: i - 1, domain=subsample) + unop: 1.0,
    },
    expr.when(lambda i, domain: i > 0 and len(domain) > 1): {
      expr(i=lambda i: i - 1, domain=subsample) + expr(i=lambda i: i - 1, domain=subsample) + biop: 1.0,
      constant: lambda i: 1 / (i + 1),
    },
    expr.when(lambda domain: len(domain) == 1): op('variable', lambda domain: domain[0]),
    expr.when(lambda i, domain: i <= 0 and len(domain) > 0): op('variable', lambda domain: domain[0]),

    unop.when(lambda stack: np.std(stack[-1]) < 1.0): op('exp'),
    unop.when(lambda stack: np.std(stack[-1]) >= 1.0): op('const', lambda std: 1 / std).where(std=lambda stack: np.std(stack[-1])) + op('mul'),

    biop: {
      op('add'): 1.0,
      op('mul'): 1.0,
    },

    constant: {
      op('const', 0.0): 1.0,
      op('const', 1.0): 1.0,
      op('const', 2.0): 1.0,
    }
  }

  generator = GeneratorMachine(lib, rules=rules)
  machine = symgen.StackMachine(lib, max_stack_size=1024)

  trace = np.random.normal(size=(3, 8)).astype(np.float32)

  print(np.std(trace, axis=-1))

  for i in range(15):
    result, stack, memory = generator.generate(random.Random(i), expr(i=3, domain=[0, 1, 2]), inputs=trace)
    r = machine(result, inputs=trace)

    print(result)

    assert len(stack) == 1
    assert np.all(np.abs(r[0] - stack[0]) < 1.0e-6)

def test_attempt():
  import symgen
  from symgen.generator import GeneratorMachine, symbol, op

  lib = symgen.lib.merge(symgen.lib.core, symgen.lib.std)

  expr = symbol('expr')
  c = symbol('c')

  rules = {
    expr.assure(lambda std: std < 1.0).where(std=lambda stack: np.std(stack[-1])): op('variable', 0) + c + op('mul'),
    c: op('const', lambda rng: rng.expovariate())
  }

  generator = GeneratorMachine(lib, rules=rules)
  trace = np.random.normal(size=(1, 128)).astype(np.float32)
  trace = (1 + 1.0e-3) * trace / np.std(trace)

  rng = random.Random(12345)

  for i in range(15):
    result, stack, memory = generator.generate(rng, expr(), inputs=trace, attempts=16)

    assert len(result) == 3
    _, const, _ = result

    assert const[0] == 'const'
    assert const[1] < 1.0

  n, m = 1024, 3
  raises = 0
  for i in range(n):
    try:
      result, stack, memory = generator.generate(rng, expr(), inputs=trace, attempts=m)
    except ValueError:
      raises += 1

  expected = 0
  for _ in range(n):
    expected += 0 if any(rng.expovariate() < 1.0 for _ in range(m)) else 1

  print(f'{raises} / {expected}')
  assert 0.5 * expected < raises < 2 * expected
