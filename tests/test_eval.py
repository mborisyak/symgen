import symgen
import numpy as np

def test_compile():
  shared = symgen.machine.compile(symgen.lib.core, symgen.lib.std, output='sym_eval-test.so', source='sym_eval-test.c', debug=True)
  m = symgen.machine.link(shared)

  code = np.array([[1, 1], [1, 1], [2, 0]], dtype=np.int32)
  sizes = np.array([3], dtype=np.int32)
  inputs = np.zeros(shape=(1, 3, 1), dtype=np.float32)
  outputs = np.zeros(shape=(1, 3, 1), dtype=np.float32)

  m.stack_eval(code, sizes, inputs, outputs)
  print(outputs)