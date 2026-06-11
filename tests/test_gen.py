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

def test_context_passing():
  from symgen.generator import symbol
  s = symbol('s')

  inv = s(i=lambda i, j: i + j)
  state = inv(s.seed(i=1, j=2).parameters, {})

  assert state.parameters['i'] == 3
  assert state.parameters['j'] == 2

def test_invocation():
  from symgen.generator import symbol
  s = symbol('s')

  inv = s(i=2, j=3, c=4)

  print(inv({}, {}))

  print(s()(dict(i=2, j=3, c=4), {} ))

def _tree_grammar():
  """A node is a depth-bounded add/mul tree over `load` leaves."""
  from symgen import symbol, op, load
  expr = symbol('expr').auto(depth=lambda depth: depth - 1)
  rules = {
    expr.when(lambda depth: depth > 0): {
      expr() + expr() + op('add'): 1.0,
      expr() + expr() + op('mul'): 1.0,
      load: 2.0,
    },
    expr.when(lambda depth: depth <= 0): load,
  }
  return expr, rules

def test_node_single_output():
  from symgen import GeneratorMachine, StackMachine, RandomTopology

  expr, rules = _tree_grammar()
  libs = (symgen.lib.core, symgen.lib.std)
  gen = GeneratorMachine(*libs, rules=rules, topology=RandomTopology(n_nodes=4, max_inputs=2))
  machine = StackMachine(*libs)

  rng = np.random.default_rng(1)
  inputs = rng.normal(size=(2, 32)).astype(np.float32)

  for _ in range(10):
    program = gen.generate(rng, expr(depth=2), inputs=inputs, n_out=1)
    out = machine(program, inputs)
    assert out.shape == (1, 32)
    assert np.all(np.isfinite(out))

def test_restore_topology():
  """The topology reconstructed from a lowered program is acyclic, and the program runs."""
  from symgen import GeneratorMachine, StackMachine, RandomTopology, restore_topology

  expr, rules = _tree_grammar()
  libs = (symgen.lib.core, symgen.lib.std)
  gen = GeneratorMachine(*libs, rules=rules, topology=RandomTopology(n_nodes=6, max_inputs=3))
  machine = StackMachine(*libs)

  rng = np.random.default_rng(7)
  inputs = rng.normal(size=(3, 48)).astype(np.float32)

  for _ in range(10):
    program = gen.generate(rng, expr(depth=3), inputs=inputs, n_out=2)

    graph = restore_topology(program)
    for cell, node in enumerate(graph):
      if node is None:
        continue
      links, _ = node
      assert all(g < cell for g in links)                # acyclic: links are earlier cells

    out = machine(program, inputs)
    assert out.shape == (2, 48)
    assert np.all(np.isfinite(out))

def test_rejection_sampling():
  """`assure` + attempts: every node's output must have std < 1."""
  from symgen import symbol, op, load, GeneratorMachine, StackMachine, RandomTopology

  expr = symbol('expr')
  rules = {
    expr.assure(lambda stack: np.std(stack[-1]) < 1.0):
      load + op('const', lambda rng: rng.exponential()) + op('mul'),
  }
  libs = (symgen.lib.core, symgen.lib.std)
  gen = GeneratorMachine(*libs, rules=rules, topology=RandomTopology(n_nodes=3, max_inputs=1))
  machine = StackMachine(*libs)

  rng = np.random.default_rng(123)
  inputs = rng.normal(size=(1, 256)).astype(np.float32)
  inputs = inputs / np.std(inputs)                     # std == 1

  for _ in range(10):
    program = gen.generate(rng, expr(), inputs=inputs, n_out=1, attempts=64)
    out = machine(program, inputs)
    assert np.all(np.isfinite(out))
    assert np.std(out) < 1.0 + 1.0e-4

def test_self_normalizing_grammar():
  from symgen import symbol, op, load, GeneratorMachine, StackMachine, RandomTopology
  from symgen.grammars import normalized

  unbounded = symbol('unbounded').auto(depth=lambda depth: depth - 1)
  positive = symbol('positive').auto(depth=lambda depth: depth - 1)

  rules = {
    unbounded.when(lambda depth: depth > 0): {
      positive(depth=lambda depth: depth): 2.0,
      unbounded(): 2.0,

      unbounded() + unbounded() + normalized.add: 1.0,
      unbounded() + unbounded() + normalized.mul: 1.0,
      positive() + unbounded() + normalized.div: 1.0,
      positive() + normalized.log: 1.0,
      unbounded() + normalized.square: 1.0,
      unbounded() + normalized.tanh: 1.0,
    },
    positive.when(lambda depth: depth > 0): {
      positive(): 2.0,
      positive() + positive() + normalized.pos_add: 2.0,
      positive() + positive() + normalized.pos_mul: 2.0,
      positive() + normalized.pos_exp: 1.0,
      unbounded() + normalized.pos_square: 0.5,
      positive() + normalized.pos_log: 1.0,
      unbounded() + normalized.softplus: 1.0,
      positive() + normalized.sqrt: 1.0,
    },
    unbounded.when(lambda depth: depth <= 0): load,
    positive.when(lambda depth: depth <= 0): unbounded() + normalized.softplus,
  }

  libs = (symgen.lib.core, symgen.lib.std, symgen.lib.affine)
  machine = StackMachine(*libs)
  generator = GeneratorMachine(*libs, rules=rules, topology=RandomTopology(n_nodes=10, max_inputs=4))

  np_rng = np.random.default_rng(12345678)

  grid_x = np.linspace(-5, 5, num=129)
  grid_y = np.linspace(-5, 5, num=127)
  grid = np.stack(np.meshgrid(grid_x, grid_y, indexing='ij'), axis=0)
  grid = (grid - grid.mean(axis=(1, 2), keepdims=True)) / grid.std(axis=(1, 2), keepdims=True)

  n = 6
  results = []
  finite = 0
  for _ in range(n * n):
    program = generator(np_rng, unbounded(depth=9), inputs=grid, n_out=1)
    result = machine(program, grid)
    assert result.shape == (1, *grid.shape[1:])
    if np.all(np.isfinite(result)):
      finite += 1
    results.append(result)

  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig = plt.figure(figsize=(2 * n, 2 * n))
  axes = fig.subplots(n, n, squeeze=False).ravel()
  for i in range(n * n):
    field = results[i][0]
    if np.all(np.isfinite(field)):
      lo, hi = np.percentile(field, [1, 99])
      axes[i].contourf(grid_x, grid_y, np.clip(field, lo, hi).T, cmap=plt.cm.cividis)
    axes[i].axis('off')
  fig.tight_layout()
  fig.savefig('self-normalizing-grammar.png')
  plt.close(fig)

  assert finite >= int(0.8 * n * n), f'only {finite}/{n * n} finite'
