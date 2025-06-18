import math

import symgen
import numpy as np

def test_eval():
  machine = symgen.StackMachine(
    symgen.lib.core, symgen.lib.std, debug=True
  )

  def evaluate(expression, *args):
    result = machine.evaluate(expression, *args)
    if len(args) > 0:
      arguments = ', '.join(f'({i}) = {x:.2f}' for i, x in enumerate(args))
      print(f'{expression} = {result} where {arguments}')
    else:
      print(f'{expression} = {result}')
    return result

  evaluate('1.0 2.0 add')
  evaluate('(0) 1.5 add {0} [0] [0] mul', 1.0)
  evaluate('0.3989422804014327 2 (0) (0) mul div neg exp mul {0} [0] log [0] mul', 3.0)
  evaluate('(0) (0) mul (1) (1) mul add sqrt', 3.0, 4.0)

  binary = machine.assembly.assemble('(0) (0) mul (1) (1) mul add sqrt')
  print(binary.T)
  disassembled = machine.assembly.disassemble(binary)
  print(disassembled)

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