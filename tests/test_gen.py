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
  import math
  import symgen
  from symgen.generator import GeneratorMachine, symbol

  lib = symgen.lib.merge(symgen.lib.core, symgen.lib.std)

  expr = symbol('expr')('i')
  constant = symbol('constant')()

  rules={
    expr: {
      expr + expr + lib['add']: 0.1,
      expr + expr + lib['mul']: 0.2,
      constant: 0.6,
    },
    constant: {
      (lib['const'], 0.0): 0.2,
      (lib['const'], 1.0): 0.2,
      (lib['const'], 2.0): 0.2,
      (lib['const'], math.pi): 0.2,
      (lib['const'], math.e): 0.2,
    }
  }

  generator = GeneratorMachine(lib, rules=rules,)

  result = generator(random.Random(123), expr(3))

  print(result)

def test_grammar_hash():
  from symgen.generator import Grammar, symbol, op

  libraries = (symgen.lib.core, symgen.lib.std)

  expr = symbol('expr')('i', 'j')
  mult = symbol('multiply')()
  constant = symbol('constant')()

  grammar = Grammar(
    rules={
      expr.when('i > 0'): {
        expr(j=0, i='i - 1') + expr('i - 1') + op('add'): 'i',
        expr('i - 1') + expr('i - 1') + mult: 1,
        constant: 2,
      },
      expr.when('i == 0'): {
        constant + constant + op('add'): 1,
        constant + constant + op('mul'): 1,
        constant: 3,
      },
      mult: {
        op('mul'): 1
      },
      constant.when('domain.size > 0'): {
        constant(domain='DOMAIN_RANDOM_SUBSET()', depth='depth'): 2,
        op('input')('RANDOM_INPUT()'): 5,
        op('const')(0.0): 1,
        op('const')(1.0): 1,
        op('const')(math.pi): 1,
      },
      constant: {
        op('const')(0.0): 1,
        op('const')(1.0): 1,
        op('const')(math.pi): 1,
      }
    }
  )

  generator = symgen.GeneratorMachine(
    *libraries, grammar=grammar,
    seed_symbol=expr('4', '-1'),
    source='sym_gen.c', debug=True
  )

  instructions, instruction_sizes = generator.generate(
    1234567, 765432, max_depth=10, instruction_limit=1024, expression_limit=8, max_inputs=2,
  )
  print(instructions)
  print(instruction_sizes)
  c = 0
  for s in instruction_sizes:
    print(generator.assembly.disassemble(instructions[c:c+s]))
    c += s

def test_grammar_args():
  from symgen.generator import Grammar, symbol, op

  libraries = (symgen.lib.core, symgen.lib.std)

  expr = symbol('expr')('float x')

  grammar = Grammar(
    rules={
      expr.when('x > 0.0'): {
        expr(x='x - 1') + op('const')(2.0) + op('mul'): 1.0,
        expr() + op('const')(1.0) + op('add'): 0.5
      },
      expr: op('const')('(arg_t) { .number = x}')
    }
  )

  generator = symgen.GeneratorMachine(
    *libraries, grammar=grammar,
    seed_symbol=expr('(float) 3.5'),
    source='sym_gen.c', debug=False
  )

  instructions, instruction_sizes = generator.generate(
    1234567, 765432, max_depth=20, instruction_limit=1024, normal_limit=8, max_inputs=2
  )
  c = 0
  print()
  for s in instruction_sizes:
    print(generator.assembly.disassemble(instructions[c:c+s]))
    c += s

def test_grammar_memory():
  from symgen.machine import StackMachine
  from symgen.generator import Grammar, symbol, op

  libraries = (symgen.lib.core, symgen.lib.std)

  expr = symbol('expr')('allocated', 'limit')
  subexpr = symbol('subexpr')('i')
  summation = symbol('sum')('i', 'limit')

  grammar = Grammar(
    rules={
      expr.when('allocated < limit'): {
        subexpr(i='allocated') + expr('allocated + 1'): 1,
      },
      expr: {
        summation('0', 'limit'): 1
      },
      subexpr: {
        op('input')('RANDOM_INPUT()') + op('input')('RANDOM_INPUT()') + op('mul') + op('store')('i'): 1,
      },
      summation.when('i == 0'): {
        op('memory')('0') + op('memory')('1') + op('add') + summation('2', 'limit'): 1
      },
      summation.when('i < limit'): {
        op('memory')('i') + op('add') + summation('i + 1', 'limit'): 1
      },
      summation: {}
    }
  )

  generator = symgen.GeneratorMachine(
    *libraries, grammar=grammar,
    seed_symbol=expr('0', '5'),
    source='sym_gen.c', debug=False
  )

  instructions, instruction_sizes = generator.generate(
    1234567, 765432, max_depth=10, instruction_limit=1024, normal_limit=8, max_inputs=4
  )
  c = 0
  print()
  for s in instruction_sizes:
    print(generator.assembly.disassemble(instructions[c:c+s]))
    c += s

  machine = StackMachine(*libraries)
  n_expr = instruction_sizes.shape[0]
  n_evals = 3
  inputs = np.random.randint(low=0, high=10, size=(n_expr, n_evals, 4)).astype(np.float32)
  outputs = np.zeros(shape=(n_expr, n_evals, 1), dtype=np.float32)
  machine.execute(instructions, instruction_sizes, inputs, outputs)

  print(outputs)

def test_tmp():
  import tempfile

  file = tempfile.NamedTemporaryFile('w', prefix='test', delete=True, delete_on_close=False)
  print(file.name)

  with open(file.name, 'w') as f:
    f.write('Hello!')

def test_arity():
  libraries = (symgen.lib.core, symgen.lib.std)
  assembly = symgen.assembly.Assembly(*libraries)

  arities = {
    op_name: code.count('POP()')
    for op_name, code in assembly.ops.items()
  }

  code = ['const', 'const', 'add', 'const', 'add', 'const', 'const', 'add', 'mul']
  print()
  print(' '.join(code))

  lens = []
  for op in code:
    n = arities[op]
    l = 0
    for j in range(n):
      l += lens[len(lens) - l - 1]

    if op == 'store':
      pass
    else:
      l += 1

    lens.append(l)

  print(' '.join(str(l) for l in lens))

  for i, op in enumerate(code):
    n = arities[op]
    l = 0
    args = []
    for j in range(n):
      arg_len = lens[i - l - 1]
      arg = code[i - l - arg_len : i - l]
      args.append(arg)
      l += arg_len

    print(op, ', '.join([' '.join(arg) for arg in args]))