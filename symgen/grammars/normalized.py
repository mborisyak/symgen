"""Coefficient-sampling scheme: each op samples coefficients for the robustly-normalized operand and
folds the normalization back into them. Normalizes the O(1) bulk only -- tail-amplifying ops still
grow tails in deep DAGs (a separate, structural concern)."""
import numpy as np

from ..generator import op

__all__ = [
  'add', 'pos_add',
  'mul', 'pos_mul', 'div',
  'exp', 'pos_exp', 'log', 'pos_log',
  'square', 'pos_square',
  'tanh', 'softplus', 'sqrt', 'cbrt', 'sin', 'id',

  'add_normalization', 'pos_add_normalization',
  'mul_normalization', 'pos_mul_normalization', 'div_normalization',
  'exp_normalization', 'pos_exp_normalization', 'log_normalization', 'pos_log_normalization',
  'square_normalization', 'pos_square_normalization',
  'tanh_normalization', 'softplus_normalization', 'sqrt_normalization',
  'cbrt_normalization', 'sin_normalization', 'id_normalization'
]

PRIOR_DOF = 8.0     # prior pseudo-observations -> qrange/qspan floor of +DOF/n (justified, not eps)
Q_LO, Q_HI = 0.05, 0.95
IQ = 3.2897072539029457     # q95 - q05 of a standard normal (2 * 1.6448536...)

def bstd(z, dof=PRIOR_DOF):
  """Bayesian std (posterior-mean std under a weak scaled-inverse-chi^2 prior)."""
  mean = np.mean(z)
  ss = np.sum(np.square(z - mean))
  prior_var = np.mean(np.square(z))
  return np.sqrt((dof * prior_var + ss) / (dof + z.size))

def _mix(emp, ana):
  """Softmax-blend empirical and analytical scale estimates; a heavy tail (ana >> emp) leans analytical."""
  if emp <= 0.0:
    return ana                                   # constant operand: no empirical spread -> analytical
  r = ana / emp
  if not np.isfinite(r):
    return ana                                   # operand magnitude overflowed -> use analytical
  m = max(1.0, r)                                 # stable softmax over logits (1, r)
  e0, e1 = np.exp(1.0 - m), np.exp(r - m)
  return (e0 * emp + e1 * ana) / (e0 + e1)

def qrange(z):
  emp = float(np.quantile(z, Q_HI) - np.quantile(z, Q_LO))   # empirical 90% span
  ana = IQ * bstd(z)                                          # normal-model 90% span (tail-sensitive)
  return _mix(emp, ana) + PRIOR_DOF / z.size                 # +DOF/n: > 0 even for a constant operand

def qspan(z):
  a = np.abs(z)
  emp = float(np.quantile(a, Q_HI))
  ana = float(np.mean(a)) + 1.6448536269514722 * bstd(z)     # heavy tail -> large bstd -> large span
  return _mix(emp, ana) + PRIOR_DOF / a.size

def qmin(z):
  return np.median(z) - qrange(z) / 2.0          # edges from the median + tail-aware spread

def qmax(z):
  return np.median(z) + qrange(z) / 2.0

def add_normalization(wx=lambda rng: rng.normal(), wy=lambda rng: rng.normal()):
  def normalization(rng: np.random.Generator, stack):
    """Each term w*operand is O(1) (weight / robust std); center the sum at 0."""
    x, y = stack[-1], stack[-2]
    wx_ = wx(rng) / qrange(x)
    wy_ = wy(rng) / qrange(y)
    return wx_, wy_, -wx_ * np.median(x) - wy_ * np.median(y)

  return normalization

def pos_add_normalization(wx=lambda rng: rng.lognormal(0.0, 0.5), wy=lambda rng: rng.lognormal(0.0, 0.5)):
  def normalization(rng: np.random.Generator, stack):
    """Positive (LogNormal) weights on positive operands -> positive sum; /median keeps it O(1)."""
    x, y = stack[-1], stack[-2]
    return wx(rng) / np.median(x), wy(rng) / np.median(y), 0.0

  return normalization

add = op('affine_add', add_normalization())
pos_add = op('affine_add', pos_add_normalization())

def mul_normalization(cx=lambda rng: 0.5 * rng.normal(), cy=lambda rng: 0.5 * rng.normal()):
  def normalization(rng: np.random.Generator, stack):
    """Center each factor near its operand's median; ~centered factors keep the product scale-preserving."""
    x, y = stack[-1], stack[-2]
    cx_ = -np.median(x) + cx(rng) * qrange(x)
    cy_ = -np.median(y) + cy(rng) * qrange(y)
    return cx_, cy_

  return normalization

def pos_mul_normalization(cx=lambda rng: rng.lognormal(-1.0, 0.5), cy=lambda rng: rng.lognormal(-1.0, 0.5)):
  def normalization(rng: np.random.Generator, stack):
    """Positive shift (margin * median) keeps each factor > 0 -> positive product."""
    x, y = stack[-1], stack[-2]
    return cx(rng) * np.median(x), cy(rng) * np.median(y)

  return normalization

def div_normalization(cx=lambda rng: rng.normal(), margin=lambda rng: rng.lognormal(-1.0, 0.5)):
  def normalization(rng: np.random.Generator, stack):
    """Numerator centered; denominator shifted by margin*median(y) > 0 so y + cy > 0 everywhere (no /0)."""
    x, y = stack[-1], stack[-2]
    cx_ = -np.median(x) + cx(rng) * qrange(x)
    cy_ = margin(rng) * np.median(y)
    return cx_, cy_

  return normalization

mul = op('affine_mul', mul_normalization())
pos_mul = op('affine_mul', pos_mul_normalization())
div = op('affine_div', div_normalization())

def log_normalization(margin=lambda rng: rng.lognormal(-1.0, 0.5)):
  def normalization(rng: np.random.Generator, stack):
    """Positive margin c = margin*median keeps x + c > 0; b = -log(median + c) centers the output at 0."""
    x = stack[-1]
    m = np.median(x)
    c_ = margin(rng) * m
    return c_, -np.log(m + c_)

  return normalization

def pos_log_normalization(c=lambda rng: 1.0 + rng.lognormal(0.0, 0.5)):
  def normalization(rng: np.random.Generator, stack):
    """Shift by c >= 1 so log(operand + c) >= 0 (strictly positive output)."""
    return c(rng), 0.0

  return normalization

log = op('affine_log', log_normalization())
pos_log = op('affine_log', pos_log_normalization())

def square_normalization(gain=lambda rng: rng.lognormal(0.0, 0.5), shift=lambda rng: rng.normal()):
  def normalization(rng: np.random.Generator, stack):
    """(w*x + c)^2 scaled by qspan (robust magnitude) so the squared argument stays O(1)."""
    x = stack[-1]
    w_ = gain(rng) / qspan(x)
    return w_, shift(rng) - w_ * np.median(x)

  return normalization

pos_square_normalization = square_normalization      # square output is positive by construction

square = op('affine_square', square_normalization())
pos_square = op('affine_square', pos_square_normalization())

def _free(gain, shift):
  """w = gain / robust std, b centers the pre-activation so the bulk sits in the op's responsive region."""
  def normalization(rng: np.random.Generator, stack):
    x = stack[-1]
    w_ = gain(rng) / qrange(x)
    return w_, shift(rng) - w_ * np.median(x)
  return normalization

def tanh_normalization(gain=lambda rng: rng.lognormal(0.0, 0.5), shift=lambda rng: rng.normal()):
  return _free(gain, shift)

def softplus_normalization(gain=lambda rng: rng.lognormal(0.0, 0.5), shift=lambda rng: rng.normal()):
  return _free(gain, shift)

def cbrt_normalization(gain=lambda rng: rng.lognormal(0.0, 0.5), shift=lambda rng: rng.normal()):
  return _free(gain, shift)

def sqrt_normalization(gain=lambda rng: rng.lognormal(0.0, 0.5), b=lambda rng: rng.lognormal(-1.0, 0.5)):
  def normalization(rng: np.random.Generator, stack):
    """w = gain/median > 0 and b = positive margin > 0, so w*x + b > 0 everywhere (no NaN)."""
    x = stack[-1]
    return gain(rng) / np.median(x), b(rng)

  return normalization

def id_normalization(scale=lambda rng: rng.uniform(1.0, 3.0), shift=lambda rng: rng.normal()):
  def normalization(rng: np.random.Generator, stack):
    """Explicit standardizer: w*x + b with output center ~ shift, robust-std ~ scale."""
    x = stack[-1]
    sgn = 1.0 if rng.random() < 0.5 else -1.0
    w_ = sgn * scale(rng) / qrange(x)
    return w_, shift(rng) - w_ * np.median(x)

  return normalization

tanh = op('affine_tanh', tanh_normalization())
softplus = op('affine_softplus', softplus_normalization())
sqrt = op('affine_sqrt', sqrt_normalization())
cbrt = op('affine_cbrt', cbrt_normalization())
id = op('affine_id', id_normalization())

exp_normalization = softplus_normalization
exp = op('affine_softplus', softplus_normalization())

def pos_exp_normalization(gain=lambda rng: rng.lognormal(0.0, 0.5), c=lambda rng: rng.uniform(-1, 1)):
  def normalization(rng: np.random.Generator, stack):
    """Decaying exp (w < 0, x >= 0): w*x <= 0 so exp(w*x+c) is bounded in (0, e^c] -- no overflow."""
    x = stack[-1]
    w_ = -gain(rng) / qrange(x)
    c_ = c(rng)
    return w_, c_

  return normalization

pos_exp = op('affine_exp', pos_exp_normalization())

def sin_normalization(gain=lambda rng: rng.lognormal(0.0, 0.5), phase=lambda rng: rng.uniform(0.0, 2.0 * np.pi)):
  def normalization(rng: np.random.Generator, stack):
    """sin(w*x + b), bounded [-1,1]; w = gain/qrange sets frequency, b a random phase (periodic structure)."""
    x = stack[-1]
    w_ = gain(rng) / qrange(x)
    return w_, phase(rng) - w_ * np.median(x)

  return normalization

sin = op('affine_sin', sin_normalization())
