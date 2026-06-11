"""Diagnostic: the scheme normalizes the O(1) bulk but doesn't hard-bound outputs (tail-amplifiers
still grow tails). Attribute each first small->large crossing to the op that originates it."""
import collections

import numpy as np
import jax.numpy as jnp

import symgen
from symgen import symbol, GeneratorMachine, StackMachine, RandomTopology
from symgen.grammars import normalized
from symgen.operation import bind

THRESHOLD = 100.0     # |value| this far above the O(1) bulk counts as a blow-up
CORE = {'const', 'load', 'store'}


def _grammar():
  """The grammar from test_self_normalizing_grammar (tests/test_gen.py)."""
  unbounded = symbol('unbounded').auto(depth=lambda depth: depth - 1)
  positive = symbol('positive').auto(depth=lambda depth: depth - 1)

  rules = {
    unbounded.when(lambda depth: depth > 0): {
      positive(depth=lambda depth: depth): 2.0,
      unbounded(): 1.0,
      unbounded() + unbounded() + normalized.add: 2.0,
      unbounded() + unbounded() + normalized.mul: 2.0,
      positive() + unbounded() + normalized.div: 0.5,
      unbounded() + normalized.exp: 0.5,
      positive() + normalized.log: 0.5,
      unbounded() + normalized.square: 1.0,
      unbounded() + normalized.tanh: 1.0,
    },
    positive.when(lambda depth: depth > 0): {
      positive(): 0.5,
      positive() + positive() + normalized.pos_add: 2.0,
      positive() + positive() + normalized.pos_mul: 2.0,
      unbounded() + normalized.pos_exp: 1.0,
      unbounded() + normalized.pos_square: 1.0,
      positive() + normalized.pos_log: 1.0,
      unbounded() + normalized.softplus: 1.0,
      positive() + normalized.sqrt: 1.0,
    },
    unbounded.when(lambda depth: depth <= 0): symgen.load,
    positive.when(lambda depth: depth <= 0): unbounded() + normalized.pos_exp,
  }
  return unbounded, rules


def _magnitude(a):
  """Return (max |finite value|, has_nonfinite); empty/all-nonfinite -> (0.0, True)."""
  a = np.asarray(a)
  finite = np.isfinite(a)
  mag = float(np.abs(a[finite]).max()) if finite.any() else 0.0
  return mag, not finite.all()


def _first_blowup(machine, program, inputs):
  """Run the program op-by-op; return (label, peak) where label names the op that first went large
  ('(bounded)' if none, '(via-load)' if the large value entered through a core load/const)."""
  n_in = inputs.shape[0]
  n_cells = machine._n_cells(program, n_in)
  memory = machine._seed_memory(jnp.asarray(inputs), n_cells)
  stack = []
  peak = 0.0

  for op, *args in program:
    arity, arguments = machine.properties[op]
    operands = [stack.pop() for _ in range(arity)]

    in_mag = max((_magnitude(o)[0] for o in operands), default=0.0)
    result = machine.library[op](*operands, **bind(arguments, args, memory))
    if result is None:
      continue
    stack.append(result)

    out_mag, out_bad = _magnitude(result)
    large = out_bad or out_mag > THRESHOLD
    peak = max(peak, np.inf if out_bad else out_mag)

    if large and in_mag <= THRESHOLD:
      return (op if op not in CORE else '(via-load)'), (np.inf if out_bad else out_mag)

  return '(bounded)', peak


def test_first_blowup_statistics():
  """Attribute blow-ups to originating ops; assert (bounded) is truly bounded and most outputs stay finite."""
  unbounded, rules = _grammar()
  libs = (symgen.lib.core, symgen.lib.std, symgen.lib.affine)
  machine = StackMachine(*libs)
  generator = GeneratorMachine(*libs, rules=rules, topology=RandomTopology(n_nodes=10, max_inputs=4))

  rng = np.random.default_rng(0)
  gx = np.linspace(-5, 5, num=129)
  gy = np.linspace(-5, 5, num=127)
  grid = np.stack(np.meshgrid(gx, gy, indexing='ij'), axis=0)
  grid = (grid - grid.mean(axis=(1, 2), keepdims=True)) / grid.std(axis=(1, 2), keepdims=True)

  N = 500
  first = collections.Counter()
  generated = 0
  worst = 0.0
  worst_bulk = 0.0            # the largest 5-95% bulk magnitude any final output reached
  bounded_peak = 0.0          # the largest magnitude any '(bounded)' expression ever reached
  nonfinite = 0               # expressions whose final output contains any inf/nan

  for _ in range(N):
    try:
      program = generator(rng, unbounded(depth=5), inputs=grid, n_out=1)
    except Exception:
      continue
    generated += 1
    out = np.asarray(machine(program, grid))
    if not np.all(np.isfinite(out)):
      nonfinite += 1
    else:
      lo, hi = np.quantile(out, normalized.Q_LO), np.quantile(out, normalized.Q_HI)
      worst_bulk = max(worst_bulk, abs(float(lo)), abs(float(hi)))
    label, mag = _first_blowup(machine, program, grid)
    first[label] += 1
    if np.isfinite(mag):
      worst = max(worst, mag)
    if label == '(bounded)':
      bounded_peak = max(bounded_peak, mag)

  blew = generated - first['(bounded)']
  print(f'\nHow each of {generated} generated expressions (of {N} attempts) first exceeds '
        f'|x| > {THRESHOLD:g}:\n')
  print(f'  {"cause":<14}{"# expressions":>15}{"share":>10}')
  for label, n in first.most_common():
    print(f'  {label:<14}{n:>15}{n / generated:>10.1%}')
  print(f'\n  (bounded)  = never exceeded {THRESHOLD:g} anywhere; an op name = that op is the '
        f'originator;\n  (via-load) = went large via a value minted in another node.')
  print(f'\n  blew up:                 {blew}/{generated} ({blew / generated:.1%})')
  print(f'  non-finite (inf/nan):    {nonfinite}/{generated} ({nonfinite / generated:.1%})')
  print(f'  largest 5-95% bulk seen: {worst_bulk:.3g}')
  print(f'  largest magnitude seen:  {worst:.3g}  (finite tail)')
  print(f'  peak within any (bounded) expr: {bounded_peak:.3g}  (must be <= {THRESHOLD:g})')

  assert generated > 0
  assert bounded_peak <= THRESHOLD
  assert nonfinite <= 0.2 * generated, f'{nonfinite}/{generated} expressions produced inf/nan'
