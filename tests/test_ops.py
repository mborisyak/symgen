import math

import symgen
import numpy as np

import matplotlib.pyplot as plt

def test_ops():
  import time

  n = 8 * 1024
  m = 1024
  x = np.arange(0, n * n, dtype=np.float32).reshape((n, n))
  index = np.random.randint(n, size=(m, ))

  z = np.zeros(shape=(n, ), dtype=np.float64)

  start_t = time.perf_counter()
  for i in index:
    np.add(x[:, i], z, out=z)
  end_t = time.perf_counter()
  print(end_t - start_t)

  start_t = time.perf_counter()
  for i in index:
    np.add(x[i, :], z, out=z)
  end_t = time.perf_counter()
  print(end_t - start_t)

def test_machine():
  from symgen import StackMachine, lib

  machine = StackMachine(lib.core, lib.std)

  expression = [
    ('const', 1),
    ('variable', 0),
    ('add', None),
    ('variable', 1),
    ('mul', None),
  ]
  n = 4
  inputs = np.arange(2 * n, dtype=int).reshape((2, n))

  result = machine(expression, inputs)
  trace = machine.trace(expression, inputs)

  print(trace)

  expected = ((inputs[0, :] + 1) * inputs[1, :])[None, :]

  assert np.all(np.abs(result - expected) < 1.0e-6)
  assert np.all(np.abs(trace[-1] - expected) < 1.0e-6)

def test_source():
  f = lambda i: i + 1
  g = lambda i: 1 + i
  import dis
  assert dis.Bytecode(f) == dis.Bytecode(g)