import random
from typing import NamedTuple
import math

import numpy as np

import symgen

class P(NamedTuple):
  x: float = 1
  y: float = 1

def test_mega_grammar():
  import symgen
  from symgen.generator import Grammar, symbol, op

  libraries = (symgen.lib.core, symgen.lib.std)
  expr = symbol('expr')()

  normal = symbol('normal')('complexity', 'mem_from', 'mem_to')
  small = symbol('small')('complexity', 'mem_from', 'mem_to')
  large = symbol('large')('complexity', 'mem_from', 'mem_to')

  normal_positive = symbol('normal_positive')('complexity', 'mem_from', 'mem_to')
  small_positive = symbol('small_positive')('complexity', 'mem_from', 'mem_to')
  large_positive = symbol('large_positive')('complexity', 'mem_from', 'mem_to')

  constant_depth = 3

  constant_grammar = symgen.grammars.scale.ConstantGrammar()

  rules = {
    expr:
      normal(3, -1, -1) + op('store')(0) + normal(3, 0, 1) + op('store')(1) +
        normal(3, 0, 2) + op('store')(2) + normal(3, 0, 3) + op('store')(3) +
        normal(5, 0, 4),

    **constant_grammar.rules(),

    **symgen.grammars.scale.scale_grammar(
      small, normal, large,
      small_positive, normal_positive, large_positive,
      constant_grammar.small(complexity=constant_depth),
      constant_grammar.normal(complexity=constant_depth),
      constant_grammar.large(complexity=constant_depth),
      constant_grammar.small_positive(complexity=constant_depth),
      constant_grammar.normal_positive(complexity=constant_depth),
      constant_grammar.large_positive(complexity=constant_depth),
    ),

    small: {
      normal + op('sigmoid'): 1.0,
      normal + op('erf'): 1.0,
      normal + op('tanh'): 1.0
    },

    normal.when('mem_from >= 0'): {
      op('input')('RANDOM_INPUT()'): 'domain.size',
      op('memory')('RANDOM_MEMORY_RANGE(mem_from, mem_to)'): 'mem_to - mem_from',
    },

    ### just in case
    normal: {
      op('input')('RANDOM_INPUT()'): 1.0,
    },

    large: {
      normal() + op('exp'): 1.0,
      normal() + op('square'): 1.0,
      normal() + normal() + op('mul'): 3.0,
    },

    small_positive: normal + op('sigmoid'),

    normal_positive.when('mem_from >= 0'): {
      op('input')('RANDOM_INPUT()') + op('softplus'): 'domain.size',
      op('memory')('RANDOM_MEMORY_RANGE(mem_from, mem_to)') + op('softplus'): 'mem_to - mem_from',
    },

    normal_positive: {
      op('input')('RANDOM_INPUT()') + op('softplus'): 'domain.size',
    },

    large_positive: {
      normal() + op('exp'): 1.0,
      normal() + op('square'): 1.0,
    }
  }

  generator = symgen.GeneratorMachine(
    *libraries, grammar=Grammar(rules), seed_symbol=expr(), maximal_number_of_inputs=2,
    source='mega.c'
  )

  instructions, instruction_sizes = generator.generate(
    1234567, 765432, max_depth=1024, instruction_limit=32 * 1024, expression_limit=1024, max_expression_length=1024,
    max_inputs=2
  )
  offset = 0
  disassembled = []
  for s in instruction_sizes:
    formula = generator.assembly.pretty(instructions[offset:offset + s])
    disassembled.append(formula)
    print(formula)
    offset += s

  n, = instruction_sizes.shape
  m_ = 16
  m = m_ * m_
  xs1 = np.linspace(-3, 3, num=m_, dtype=np.float32)
  xs2 = np.linspace(-3, 3, num=m_, dtype=np.float32)
  Xs1, Xs2 = np.meshgrid(np.linspace(-3, 3, num=m_, dtype=np.float32), np.linspace(-3, 3, num=m_, dtype=np.float32), indexing='ij')
  xs = np.stack([Xs1.ravel(), Xs2.ravel()], axis=-1)
  xs = np.broadcast_to(xs[None, :, :], shape=(n, m, 2))
  print(xs.shape)
  ys = np.ndarray(shape=(n, m, 1), dtype=np.float32)

  machine = symgen.StackMachine(*libraries)
  machine.execute(instructions, instruction_sizes, xs, ys)

  import matplotlib.pyplot as plt
  k = 5
  fig = plt.figure(figsize=(3 * k, 3 * k))
  axes = fig.subplots(k, k, squeeze=False).ravel()

  for i in range(k * k):
    # axes[i].set_title(f'{disassembled[i]}')
    axes[i].contourf(xs1, xs2, ys[i, :, 0].reshape((m_, m_)))

  fig.tight_layout()
  fig.savefig('samples.png')
  plt.close()

  print('===')
  for i in range(n):
    if not np.all(np.isfinite(ys[i, :, :])):
      print(disassembled[i])
      if np.any(np.isnan(ys[i, :, :])):
        index = np.where(np.any(np.isnan(ys[i, :, :]), axis=-1))
        print('nan at', xs[i][index])

      if np.any(np.isinf(ys[i, :, :])):
        index = np.where(np.any(np.isinf(ys[i, :, :]), axis=-1))
        print('nan at', xs[i][index])

def test_const_grammar():
  from symgen import symbol, op, GeneratorMachine, StackMachine, Grammar

  libraries = (symgen.lib.core, symgen.lib.std)



  generator = symgen.GeneratorMachine(
    *libraries, grammar=Grammar(rules), seed_symbol=normal(constant_depth), maximal_number_of_inputs=2,
    source='const.c'
  )

  instructions, instruction_sizes = generator.generate(
    1234567, 765432, max_depth=1024, instruction_limit=1024 * 1024, expression_limit=1024, max_expression_length=1024,
    max_inputs=2
  )

  n, = instruction_sizes.shape
  m = 128
  xs = np.random.normal(size=(n, 1, 2)).astype(np.float32)
  print(xs.shape)
  ys = np.ndarray(shape=(n, 1, 1), dtype=np.float32)

  machine = symgen.StackMachine(*libraries)
  machine.execute(instructions, instruction_sizes, xs, ys)

  offset = 0
  disassembled = []
  for i, s in enumerate(instruction_sizes):
    formula = generator.assembly.pretty(instructions[offset:offset + s])
    disassembled.append(formula)
    print(ys[i, 0, 0,], '=', formula)
    offset += s

  print('\n=============\n')
  for i in range(n):
    if not np.all(np.isfinite(ys[i, :, :])):
      print(ys[i, 0, 0], '=', disassembled[i])

  print('\n=============\n')

  index = np.argsort(np.abs(ys[:, 0, 0]))
  for i in index[-10:]:
    print(ys[i, 0, 0], '=', disassembled[i])

  import matplotlib.pyplot as plt
  fig = plt.figure(figsize=(6, 9))
  axes = fig.subplots()
  ys = ys[:, 0, 0]
  axes.hist(ys, histtype='step', bins=100)

  fig.tight_layout()
  fig.savefig('consts.png')
  plt.close()

def test_simple_grammar():
  import symgen
  from symgen.generator import GeneratorMachine, symbol, op

  lib = symgen.lib.merge(symgen.lib.core, symgen.lib.std)

  expr = symbol('expr')('i')
  constant = symbol('constant')()

  rules={
    expr.when(lambda i: i > 0): {
      expr(lambda i: i - 1) + expr(lambda i: i - 1) + op('add'): 1.0,
      expr(lambda i: i - 1) + expr(lambda i: i - 1) + op('mul'): 1.0,
      constant: lambda i: 1 / (i + 1),
    },
    expr.when(lambda i: i <= 0) : constant(),
    constant: {
      op('const', 0.0): 1.0,
      op('const', 1.0): 1.0,
      op('const', 2.0): 1.0,
    }
  }

  generator = GeneratorMachine(lib, rules=rules,)

  result = generator(random.Random(123), expr(3))

  print(result)

def test_scope():
  import symgen
  from symgen.generator import GeneratorMachine, symbol, op

  lib = symgen.lib.merge(symgen.lib.core, symgen.lib.std)

  expr = symbol('expr')('i')
  unop = symbol('unop')()
  biop = symbol('biop')()
  constant = symbol('constant')()

  def subsample(rng: random.Random, domain):
    n = rng.randint(1, len(domain))
    return rng.sample(domain, n)


  rules = {
    expr.when(lambda i, domain, stack: i > 0 and len(domain) > 1): {
      expr(lambda i: i - 1, domain=subsample) + unop: 1.0,
    },
    expr.when(lambda i, domain: i > 0 and len(domain) > 1): {
      expr(lambda i: i - 1, domain=subsample) + expr(lambda i: i - 1, domain=subsample) + biop: 1.0,
      constant: lambda i: 1 / (i + 1),
    },
    expr.when(lambda domain: len(domain) == 1): op('variable', lambda domain: domain[0]),
    expr.when(lambda i, domain: i <= 0 and len(domain) > 0): op('variable', lambda domain: domain[0]),

    unop.when(lambda stack: np.std(stack[-1]) < 1.0): op('exp'),
    unop.when(lambda stack: np.std(stack[-1]) >= 1.0): op('const', lambda stack: 1 / np.std(stack[-1])) + op('mul'),

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
    result, stack, memory = generator(random.Random(i), expr(3), domain=[0, 1, 2], trace=trace)
    r = machine(result, inputs=trace)

    print(result)

    assert len(stack) == 1
    assert np.all(np.abs(r[0] - stack[0]) < 1.0e-6)