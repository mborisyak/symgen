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

def test_sampling():
  import random
  n = 32
  ps = [random.expovariate() for _ in range(n)]

  rng = random.Random(123)
  np_rng = np.random.default_rng(123)

  def s1(rng, likelihoods):

    u = rng.uniform(0, 1) * sum(likelihoods)
    c = 0.0
    for i in range(len(likelihoods)):
      c += likelihoods[i]

      if c >= u:
        return i

    raise ValueError

  N = 32 * 1024

  import time
  start_t = time.perf_counter()
  for _ in range(N):
    s1(rng, ps)
  end_t = time.perf_counter()
  print(f's1: {N / (end_t - start_t) / 1.0e+6} Miter/sec')

  def s2(likelihoods):
    return random.choices(range(n), weights=likelihoods)

  import time
  start_t = time.perf_counter()
  for _ in range(N):
    s2(ps)
  end_t = time.perf_counter()
  print(f's2: {N / (end_t - start_t)/ 1.0e+6} Miter/sec')

  def s3(rng, likelihoods):
    ls = np.array(likelihoods)
    p = ls / np.sum(ls)
    return rng.choice(n, p=p, replace=True)

  import time
  start_t = time.perf_counter()
  for _ in range(N):
    s3(np_rng, ps)
  end_t = time.perf_counter()
  print(f's3: {N / (end_t - start_t)/ 1.0e+6} Miter/sec')

def test_alloc():
  import random, time
  n, m = 1024, 128
  stack = np.ndarray(shape=(n, m), dtype=np.float32)

  stack[:2] = np.random.normal(size=(2, m))

  start_t = time.time()
  for i in range(2, n):
    if random.randint(0, 2) == 0:
      np.add(stack[i - 1], stack[i - 2], out=stack[i])
    else:
      np.multiply(stack[i - 1], stack[i - 2], out=stack[i])

  end_t = time.time()

  print(f'{(end_t - start_t) * 1.0e+3:.2f} millisec')

  stack = list()
  stack.append(np.random.normal(size=(m, )))
  stack.append(np.random.normal(size=(m, )))

  start_t = time.time()
  for i in range(2, n):
    if random.randint(0, 2) == 0:
      stack.append(
        np.add(stack[-1], stack[-2])
      )
    else:
      stack.append(
        np.multiply(stack[-1], stack[-2])
      )

  end_t = time.time()

  print(f'{(end_t - start_t) * 1.0e+3:.2f} millisec')

def test_scope():
  import inspect, time

  def get_scope(f):
    return tuple(name for name in inspect.signature(f).parameters)

  def f(x, y, z):
    return x * y - z

  def f2(x, y, z, **_):
    return x * y - z

  n = 4 * 1024 * 1024
  context = {'a': 16, 'b': 25, 'x': 2, 'y': 3, 'z': 4, 'zzz': 999, 'yyy': {1, 2, 3, 4, 5}}

  def scoped(f):
    _scope = get_scope(f)

    def g(**kwargs):
      return f(**{k: kwargs[k] for k in _scope})

    return g

  def scoped2(f):
    _scope = get_scope(f)

    def g(kwargs):
      return f(**{k: kwargs[k] for k in _scope})

    return g

  start_t = time.perf_counter()
  for _ in range(n):
    _ = f2(**context)
  end_t = time.perf_counter()

  print(f'kwargs: {n / (end_t - start_t) / 1.0e+6} Mops')

  scoped_f = scoped(f)
  scoped_f2 = scoped2(f)

  start_t = time.perf_counter()
  for _ in range(n):
    _ = scoped_f(**context)
  end_t = time.perf_counter()

  print(f'wrapper: {n / (end_t - start_t) / 1.0e+6} Mops')

  start_t = time.perf_counter()
  for _ in range(n):
    _ = scoped_f2(context)
  end_t = time.perf_counter()

  print(f'wrapper2: {n / (end_t - start_t) / 1.0e+6} Mops')

  scope = get_scope(f)

  start_t = time.perf_counter()
  for _ in range(n):
    _ = f(**{k: context[k] for k in scope})
  end_t = time.perf_counter()

  print(f'manual: {n / (end_t - start_t) / 1.0e+6} Mops')

  class Scoped(object):
    def __init__(self, f):
      self.f = f
      self.scope = get_scope(f)

    def __call__(self, **kwargs):
      return f(**{k: context[k] for k in scope})

  scoped_obj = Scoped(f)

  start_t = time.perf_counter()
  for _ in range(n):
    _ = scoped_obj(**context)
  end_t = time.perf_counter()

  print(f'class: {n / (end_t - start_t) / 1.0e+6} Mops')

  class Scoped2(object):
    def __init__(self, f):
      self.f = f
      self.scope = get_scope(f)

    def __call__(self, kwargs):
      return f(**{k: context[k] for k in scope})

  scoped_obj2 = Scoped2(f)

  start_t = time.perf_counter()
  for _ in range(n):
    _ = scoped_obj2(context)
  end_t = time.perf_counter()

  print(f'class2: {n / (end_t - start_t) / 1.0e+6} Mops')
