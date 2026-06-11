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
  assert abs(evaluate('(0) 1.5 add [1] (1) (1) mul', 1.0) - 6.25) < 1.0e-6
  evaluate('0.3989422804014327 2 (0) (0) mul div neg exp mul [1] (1) log (1) mul', 3.0)
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
  code = '(0) 1.5 add [1] (1) (1) mul'
  parsed = machine.parse(code)

  recovered = [op for op, *_ in parsed]
  assert recovered == ['load', 'const', 'add', 'store', 'load', 'load', 'mul']

  print(machine.parse('1.0 2.0 add'))