import numpy as np

import symgen

def _setup():
  """Build a program exercising add/mul/sub/log/sqrt/square over 2 positive inputs."""
  machine = symgen.StackMachine(symgen.lib.core, symgen.lib.std)
  program = machine.parse('(0) (1) add (0) mul (1) log add (0) sqrt mul (1) square add')
  n_in = 2
  inputs = np.random.default_rng(0).uniform(0.5, 2.0, size=(n_in, 16)).astype(np.float32)
  reference = machine(program, inputs)
  return machine, program, n_in, inputs, reference

def test_compile_matches_interpreter():
  machine, program, n_in, inputs, reference = _setup()
  f = machine.compile(program, n_in)
  out = np.asarray(f(inputs)).reshape(reference.shape)
  assert np.allclose(out, np.asarray(reference), atol=1.0e-5)

def test_compile_accepts_source():
  machine = symgen.StackMachine(symgen.lib.core, symgen.lib.std)
  code = '(0) (0) mul (1) (1) mul add sqrt'
  f = machine.compile(code, n_in=2)
  inputs = np.array([[3.0], [4.0]], dtype=np.float32)
  assert abs(float(np.asarray(f(inputs))[0, 0]) - 5.0) < 1.0e-5

def test_compile_is_jitted():
  import jax
  machine, program, n_in, _, _ = _setup()
  f = machine.compile(program, n_in)
  assert isinstance(f, jax.stages.Wrapped)
