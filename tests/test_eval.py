import math

import symgen
import numpy as np

def test_name():
  import re
  def get_name(x: str):
    name_re = re.compile(r'([^[]*\s+)?(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)(\s*\[.*])?\s*')
    return name_re.fullmatch(x).group('name')

  assert get_name('abc') == 'abc'
  assert get_name('long _abc') == '_abc'
  assert get_name('unsigned long _0abc') == '_0abc'
  assert get_name('unsigned long * _0abc') == '_0abc'
  assert get_name('unsigned long _0abc[]') == '_0abc'
  assert get_name('unsigned long abc77[]') == 'abc77'
  assert get_name('unsigned long abc77[5]') == 'abc77'
  assert get_name('unsigned long abc78[onother array[6]]') == 'abc78'
  assert get_name('unsigned long __[static 1024]') == '__'

def test_normal():
  u = np.random.uniform(size=(1024 * 1024, 10))
  sigma = np.sqrt(1 / 12)
  u = (np.sum(u, axis=-1) - 5) / np.sqrt(10) / sigma

  print(np.mean(u), np.std(u))

def test_eval():
  machine = symgen.StackMachine(
    symgen.lib.core, symgen.lib.std, debug=True, source='test_eval.c'
  )

  def evaluate(expression, *args):
    result = machine.evaluate(expression, *args)
    if len(args) > 0:
      arguments = ', '.join(f'({i}) = {x:.2f}' for i, x in enumerate(args))
      print(f'{expression} = {result} where {arguments}')
    else:
      print(f'{expression} = {result}')
    return result

  assert abs(evaluate('1.0 2.0 add') - 3.0) < 1.0e-6
  assert abs(evaluate('(0) 1.5 add {0} [0] [0] mul', 1.0) - 6.25) < 1.0e-6
  evaluate('0.3989422804014327 2 (0) (0) mul div neg exp mul {0} [0] log [0] mul', 3.0)
  assert abs(evaluate('(0) (0) mul (1) (1) mul add sqrt', 3.0, 4.0) - 5.0) < 1.0e-6

  assert abs(evaluate('1.0 2.0 div', ) - 2.0) < 1.0e-6

  binary = machine.assembly.assemble('(0) (0) mul (1) (1) mul add sqrt')
  print(binary.T)
  disassembled = machine.assembly.disassemble(binary)
  print(disassembled)

  print()

def test_execute():
  from symgen import StackMachine, lib
  machine = StackMachine(
    lib.core, lib.std, debug=True, source='test_eval.c'
  )
  binary = machine.assembly.assemble('(0) (0) mul (1) (1) mul add sqrt (0) (1) add')
  sizes = np.array([binary.shape[0]], dtype=np.int32)
  inputs = np.arange(10, dtype=np.float32).reshape((1, 5, 2))
  outputs = np.ndarray(shape=(1, 5, 2), dtype=np.float32)

  machine.execute(binary, sizes, inputs, outputs)
  print(outputs)

def test_expression():
  assembly = symgen.assembly.Assembly(symgen.lib.core, symgen.lib.std)
  code = '(0) 1.5 add {0} [0] [0] mul'
  machine_code = assembly.assemble(code)
  print(machine_code)
  recovered = [assembly.op_names[i] for i in machine_code[:, 0]]
  assert recovered == ['input', 'const', 'add', 'store', 'memory', 'memory', 'mul']

  print(assembly.disassemble(machine_code))

def test_performance():
  import time
  machine = symgen.StackMachine(
    symgen.lib.core, symgen.lib.std, debug=False, source='test_eval.c', max_stack_size=2048
  )
  n_b, m, n = 16 * 1024, 64, 32
  code = ' '.join(f'({i}) ({n + i}) mul' for i in range(n)) + ' ' + ' '.join('add' for _ in range(n - 1))
  code = machine.assembly.assemble(code)

  print(4 * n)

  # inputs = np.random.normal(size=(n_b, 1024, 2 * n)).astype(np.float32)
  inputs = np.ones(shape=(n_b, m, 2 * n), dtype=np.float32) * np.arange(m, dtype=np.float32)[None, :, None]
  sizes = np.array([code.shape[0] for _ in range(n_b)], dtype=np.int32)
  code = np.concatenate([code for _ in range(n_b)], axis=0)

  outputs = np.ndarray(shape=(n_b, m, 1), dtype=np.float32)

  symgen_start_t = time.perf_counter()
  machine.execute(code, sizes, inputs, outputs)
  symgen_end_t = time.perf_counter()

  numpy_start_t = time.perf_counter()
  numpy_output = np.sum(inputs[:, :, :n] * inputs[:, :, n:], axis=-1, keepdims=True)
  numpy_end_t = time.perf_counter()

  ops = n_b * m * (2 * n + n - 1)

  eval_output = np.ndarray(shape=(n_b, m, 1), dtype=np.float32)
  code = ' + '.join(f'arg[{i}] * arg[{n + i}]' for i in range(n))
  ast = compile(code, filename='<string>', mode='eval')

  eval_start_t = time.perf_counter()
  for i in range(n_b):
    for j in range(m):
      eval_output[i, j, 0] = eval(ast, {}, {'arg': inputs[i, j]})
  eval_end_t = time.perf_counter()

  numpy_eval_output = np.ndarray(shape=(n_b, m, 1), dtype=np.float32)
  code = ' + '.join(f'arg[:, {i}] * arg[:, {n + i}]' for i in range(n))
  ast = compile(code, filename='<string>', mode='eval')

  np_eval_start_t = time.perf_counter()
  for i in range(n_b):
    numpy_eval_output[i, :, 0] = eval(ast, {}, {'arg': inputs[i]})
  np_eval_end_t = time.perf_counter()

  print(f'SymGen: {symgen_end_t - symgen_start_t:.3f} seconds')
  print(f'eval: {eval_end_t - eval_start_t:.3f} seconds')
  print(f'eval (vectorized): {np_eval_end_t - np_eval_start_t:.3f} seconds')
  print(f'NumPy: {numpy_end_t - numpy_start_t:.3f} seconds')
  print(f'Binary (estimated): {ops / 4.0e+9:.3f} seconds')

  errors = np.abs(numpy_output - outputs)

  print(np.max(errors))

  assert np.all(errors < 1.0e-3)