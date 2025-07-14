import symgen
import numpy as np

def test_eval():
  machine = symgen.StackMachine(
    symgen.lib.core, symgen.lib.std,
  )

  result = machine('(0) (0) mul (1) (1) mul add sqrt', np.array([3.0, 4.0]))

  print(result, result.shape)

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

  print()

def test_execute():
  from symgen import StackMachine, lib
  machine = StackMachine(lib.core, lib.std)

  outputs = machine.evaluate('(0) (0) mul (1) (1) mul add sqrt (0) (1) add', 2, 3)
  assert np.abs(outputs[0] - np.sqrt(4 + 9)) < 1.0e-3
  assert np.abs(outputs[1] - 5) < 1.0e-3

def test_expression():
  machine = symgen.StackMachine(symgen.lib.core, symgen.lib.std)
  code = '(0) 1.5 add {0} [0] [0] mul'
  parsed = machine.parse(code)

  recovered = [op for op, *_ in parsed]
  assert recovered == ['variable', 'const', 'add', 'store', 'load', 'load', 'mul']

  print(machine.parse('1.0 2.0 add'))


def test_performance():
  import time
  machine = symgen.StackMachine(symgen.lib.core, symgen.lib.std)
  n_b, n_repeat, n = 16 * 1024, 64, 32
  code = ' '.join(f'({i}) ({n + i}) mul' for i in range(n)) + ' ' + ' '.join('add' for _ in range(n - 1))
  code = machine.parse(code)

  inputs = np.arange(2 * n * n_b, dtype=np.float32).reshape((2 * n, n_b))
  outputs = np.ndarray(shape=(1, n_b), dtype=np.float32)

  symgen_start_t = time.perf_counter()
  for i in range(n_repeat):
    machine(code, inputs, out=outputs)
  symgen_end_t = time.perf_counter()

  np_outputs = np.ndarray(shape=(1, n_b), dtype=np.float32)
  numpy_start_t = time.perf_counter()
  for i in range(n_repeat):
    np.sum(inputs[:, :n] * inputs[:, n:], axis=0, keepdims=True, out=np_outputs)
  numpy_end_t = time.perf_counter()

  ops = n_b * (2 * n + n - 1)

  eval_output = np.ndarray(shape=(1, n_b), dtype=np.float32)
  code = ' + '.join(f'arg[{i}] * arg[{n + i}]' for i in range(n))
  ast = compile(code, filename='<string>', mode='eval')

  eval_start_t = time.perf_counter()
  for i in range(n_repeat):
    for j in range(n_b):
      eval_output[0, j] = eval(ast, {}, {'arg': inputs[j]})
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