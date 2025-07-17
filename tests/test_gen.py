import random

import numpy as np

import symgen

def test_dsl():
  from symgen.generator import symbol, op, Condition

  s = symbol('s')

  invocation = s(k=lambda i, j: i + j).where(j=lambda i: i + 1)
  result = invocation({'i': 1}, {'i': 1})
  print(result)
  assert set(result.parameters.keys()) == {'i', 'k'}
  assert result.parameters['i'] == 1
  assert result.parameters['k'] == 3

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

  s_auto = s.auto(i=lambda i, j: i + j + 1)

  assert s_auto(j=lambda j: j + 1)(dict(i=1, j=10), {}).parameters['i'] == 12
  assert s_auto(j=lambda j: j + 1, i=lambda i, j: i + 2 * j + 2)(dict(i=1, j=10), {}).parameters['i'] == 23

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

  expr = symbol('expr').where(t1=lambda i: 2 * i + 1).auto(counter=lambda counter: counter + 1, t1=lambda t1: t1)
  rules = {
    expr.when(lambda i: i < limit): expr(i=lambda t1, t2: t1 + t2).where(t2=lambda t1, i: t1 - 3 * i - 1),
    expr.when(lambda t1: t1 >= 2 * (limit - 1) + 1): op('const', lambda counter, t1: counter + t1),
  }

  generator = symgen.GeneratorMachine(lib, rules=rules)
  expression = generator(random.Random(1234), expr.seed(i=0, counter=0, t1=1))

  assert len(expression) == 1
  (_, arg), = expression

  assert arg == limit + 2 * (limit - 1)  + 1

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

def test_self_normalizing_grammar():
  from symgen import symbol, op, GeneratorMachine, StackMachine
  from scipy import optimize

  def aff_add(x, y, *, argument):
    wx, wy, b = argument
    return wx * x + wy * y + b

  def add_normalization(rng: random.Random, stack):
    wx = rng.normalvariate()
    wy = rng.normalvariate()

    z = wx * stack[-1] + wy * stack[-2]
    mz, sz = np.mean(z), np.std(z)

    return wx / sz, wy / sz, -mz / sz

  def aff_mul(x, y, *, argument):
    cx, cy, w, b = argument
    return w * (x + cx) * (y + cy) + b

  def mul_normalization(rng: random.Random, stack):
    cx, cy = rng.normalvariate(), rng.normalvariate()

    z = (stack[-1] + cx) * (stack[-2] + cy)
    mz, sz = np.mean(z), np.std(z)

    return cx, cy, 1 / sz, -mz / sz

  def aff_div(x, y, *, argument):
    cx, cy, w, b = argument
    return w * (x + cx) / (y + cy) + b

  def div_normalization(rng: random.Random, stack):
    cx, cy = rng.normalvariate(), rng.expovariate(1.0)
    z = (stack[-1] + cx) / (stack[-2] + cy)
    mz, sz = np.mean(z), np.std(z)

    return cx, cy, 1 / sz, -mz / sz

  def aff_exp(x, *, argument):
    c, w, b = argument
    return np.exp(c * x + w) + b

  def exp_normalization(rng: random.Random, stack):
    c = rng.normalvariate()
    z = np.exp(c * stack[-1])
    mz, sz = np.mean(z), np.std(z)

    return c, -np.log(sz), -mz / sz

  def pos_exp_normalization(rng: random.Random, stack):
    c, w, b = exp_normalization(rng, stack)
    return c, w, max(0.0, b)

  def aff_log(x, *, argument):
    c, w, b = argument

    return w * np.log(x + c) + b


  def log_normalization(rng: random.Random, stack):
    c = rng.expovariate(1.0)

    z = np.log(stack[-1] + c)
    mz, sz = np.mean(z), np.std(z)

    return c, 1 / sz, -mz / sz

  def aff_square(x, *, argument):
    c, w, b = argument

    return w * np.square(x + c) + b

  def square_normalization(rng: random.Random, stack):
    c = rng.normalvariate()

    z = np.square(stack[-1] + c)
    mz, sz = np.mean(z), np.std(z)

    return c, 1 / sz, -mz / sz

  def pos_square_normalization(rng: random.Random, stack):
    c, w, b = square_normalization(rng, stack)

    return c, w, max(0.0, b)

  normalized_ops = {
    'aff_add': aff_add,
    'aff_mul': aff_mul,
    'aff_div': aff_div,
    'aff_exp': aff_exp,
    'aff_log': aff_log,
    'aff_square': aff_square
  }

  expression = symbol('expression')
  unbounded = symbol('unbounded').auto(depth=lambda depth: depth - 1)
  positive = symbol('positive').auto(depth=lambda depth: depth - 1)

  rules = {
    expression: (
        unbounded(mem=[]) + op('store', 0) +
        unbounded(mem=[0, ]) + op('store', 1) +
        unbounded(mem=[0, 1]) + op('store', 2) +
        unbounded(mem=[0, 1, 2]) #+ unbounded(mem=[0, 1, 2])
    ),
    unbounded.when(lambda depth: depth > 0): {
      unbounded() + unbounded() + op('aff_add', add_normalization): 1.0,
      unbounded() + unbounded() + op('aff_mul', mul_normalization): 2.0,
      positive() + unbounded() + op('aff_div', div_normalization): 2.0,
      unbounded() + op('aff_exp', exp_normalization): 1.0,
      positive() + op('aff_log', log_normalization): 1.0,
      unbounded() + op('aff_square', square_normalization): 1.0
    },
    positive.when(lambda depth: depth > 0): {
      unbounded() + op('aff_exp', pos_exp_normalization): 1.0,
      unbounded() + op('aff_square', pos_square_normalization): 1.0
    },
    unbounded.when(lambda depth: depth <= 0): {
      op('variable', lambda rng, inputs: rng.randint(0, inputs.shape[0] - 1)): 2.0,
      op('load', lambda rng, mem: rng.randint(0, len(mem) - 1)): (lambda mem: len(mem))
    },
    positive.when(lambda depth: depth <= 0): unbounded() + op('aff_exp', pos_exp_normalization),
  }
  libs = (symgen.lib.core, symgen.lib.std, normalized_ops)
  machine = StackMachine(*libs)
  generator = GeneratorMachine(*libs, rules=rules)

  py_rng = random.Random(12345678)
  np_rng = np.random.default_rng(12345678)

  grid_x = np.linspace(-5, 5, num=129)
  grid_y = np.linspace(-5, 5, num=127)
  grid = np.stack(np.meshgrid(grid_x, grid_y, indexing='ij'), axis=0)

  n = 6
  inputs = np_rng.normal(size=(2, 1024))
  expressions = [
    generator(py_rng, seed=expression(depth=3), inputs=grid)
    for _ in range(n * n)
  ]

  results = []
  for expr in expressions:
    # assert len(expr) <= 2 * (2 ** 6 - 1)
    result = machine(expr, inputs=grid)

    # assert result.shape == (2, 1024)

    if not np.all(np.isfinite(result)):
      trace = machine.trace(expr, inputs=grid)

      for tr, term in zip(trace, expr):
        if not np.all(np.isfinite(tr)):
          print(term)

    results.append(result)

  import matplotlib.pyplot as plt
  fig = plt.figure(figsize=(2 * n, 2 * n))
  axes = fig.subplots(n, n, squeeze=False).ravel()

  for i in range(n * n):
    # axes[i].scatter(results[i][0], results[i][1])
    c = axes[i].contourf(grid_x, grid_y, results[i][0].T, cmap=plt.cm.cividis)
    # plt.colorbar(c, ax=axes[i])
    axes[i].axis('off')

  fig.tight_layout()
  fig.savefig('self-normalizing-grammar.png')
  plt.close(fig)

