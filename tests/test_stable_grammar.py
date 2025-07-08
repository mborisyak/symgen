from typing import NamedTuple
import math

import numpy as np

import symgen

def test_stable_grammar():
  import symgen
  from symgen.generator import Grammar, symbol, op

  libraries = (symgen.lib.core, symgen.lib.std, symgen.lib.stable)
  expr = symbol('expr')()

  normal = symbol('normal')('complexity', 'mem_from', 'mem_to')
  positive = symbol('positive')('complexity', 'mem_from', 'mem_to')
  constant = symbol('constant')()
  positive_constant = symbol('positive_constant')()

  constant_depth = 3

  constant_grammar = symgen.grammars.scale.ConstantGrammar()

  rules = {
    expr: normal(5, -1, -1),
      # normal(3, -1, -1) + op('store')(0) + normal(3, 0, 1) + op('store')(1) +
      #   normal(3, 0, 2) + op('store')(2) + normal(3, 0, 3) + op('store')(3) +
      #   normal(5, 0, 4),

    normal.when('complexity > 0'): {
      normal('complexity - 2'): 2.0,
      positive('complexity - 1'): 2.0,

      normal('complexity - 1') + op('neg'): 1.0,
      positive('complexity - 1') + op('log'): 1.0,
      positive_constant + positive('complexity - 1') + op('nlog_p_c'): 1.0,

      normal('complexity - 1') + normal('complexity - 1') + op('nadd_n_n'): 1.0,
      positive('complexity - 1') + normal('complexity - 1') + op('nadd_n_p'): 1.0,
      positive('complexity - 1') + positive('complexity - 1') + op('nadd_p_p'): 1.0,
      constant + normal('complexity - 1') + normal('complexity - 1') + op('nadd_n_n_1_c'): 1.0,
      constant + positive('complexity - 1') + normal('complexity - 1') + op('nadd_n_p_1_c'): 1.0,
      positive_constant + positive('complexity - 1') + normal('complexity - 1') + op('nadd_p_p_1_c'): 1.0,

      normal('complexity - 1') + normal('complexity - 1') + op('nmul_n_n'): 1.0,
      positive('complexity - 1') + normal('complexity - 1') + op('nmul_n_p'): 1.0,
      positive('complexity - 1') + positive('complexity - 1') + op('nmul_p_p'): 1.0,

      constant + positive('complexity - 1') + op('ntanh_n_c'): 1.0,
    },

    positive.when('complexity > 0'): {
      positive('complexity - 2'): 2.0,
      # normal('complexity - 1') + op('exp'): 2.0,
      positive('complexity - 1') + positive('complexity - 1') + op('padd_p_p'): 1.0,
      positive_constant + positive('complexity - 1') + positive('complexity - 1') + op('padd_p_p_1_c'): 1.0,
      constant + normal('complexity - 1') + op('psquare_n_c'): 1.0,
      positive_constant + positive('complexity - 1') + op('psqrt_p_c'): 1.0,
      positive_constant + positive('complexity - 1') + op('pinv_p_c'): 1.0,

      positive('complexity - 1') + positive('complexity - 1') + op('pmul_p_p'): 1.0,
    },

    normal.when('mem_from >= 0'): {
      op('input')('RANDOM_INPUT()'): 'domain.size',
      op('memory')('RANDOM_MEMORY_RANGE(mem_from, mem_to)'): 'mem_to - mem_from',
    },
    ### just in case
    normal: {
      op('input')('RANDOM_INPUT()'): 1.0,
    },

    positive: {
      normal() + op('exp'): 1.0,
      constant + normal() + op('psquare_n_c'): 1.0
    },

    constant: op('const')(number='RANDOM_NORMAL()'),
    positive_constant: op('const')(number='RANDOM_LOG_NORMAL()'),
  }

  generator = symgen.GeneratorMachine(
    *libraries, grammar=Grammar(rules), seed_symbol=expr(), maximal_number_of_inputs=2,
    source='mega.c'
  )

  instructions, instruction_sizes = generator.generate(
    1234567, 765432, max_depth=1024, instruction_limit=32 * 1024, expression_limit=32 * 1024, max_expression_length=1024,
    max_inputs=2
  )
  offset = 0
  disassembled = []
  for s in instruction_sizes:
    formula = generator.assembly.pretty(instructions[offset:offset + s])
    disassembled.append(formula)
    # print(formula)
    offset += s

  n, = instruction_sizes.shape
  m_ = 32
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
    c = axes[i].contourf(xs1, xs2, ys[i, :, 0].reshape((m_, m_)))
    plt.colorbar(c, ax=axes[i])

  fig.tight_layout()
  fig.savefig('samples.png')
  plt.close()

  print('===')
  for i in range(n):
    if not np.all(np.isfinite(ys[i, :, :])) or not np.all(np.abs(ys[i, :, :]) < 20.0):
      print(disassembled[i])
      if np.any(np.isnan(ys[i, :, :])):
        index = np.where(np.any(np.isnan(ys[i, :, :]), axis=-1))
        print('nan at', xs[i][index])

      if np.any(np.isinf(ys[i, :, :])):
        index = np.where(np.any(np.isinf(ys[i, :, :]), axis=-1))
        print('nan at', xs[i][index])

      if np.any(np.abs(ys[i, :, :]) > 20.0):
        index = np.where(np.any(np.abs(ys[i, :, :]) > 20.0, axis=-1))
        print('large at', xs[i][index])
