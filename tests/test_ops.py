import math

import symgen
import numpy as np

import matplotlib.pyplot as plt

def test_ops():
  import scipy.optimize as opt
  import scipy.special as sps
  import scipy.stats as stats

  np.random.seed(123456)

  mu_log_normal = 0.0
  sigma_log_normal = 1.0

  mean_log_normal = np.exp(0.5)
  var_log_normal = (np.exp(1) - 1) * np.exp(1)
  std_log_normal = np.sqrt((np.exp(1) - 1) * np.exp(1))

  print(sigma_log_normal)
  print(1 / sigma_log_normal)

  print()

  xs = np.random.normal(size=(1024 * 1024, ))
  xs_ = np.exp(np.random.normal(size=(1024 * 1024, )))

  n1 = np.random.normal(size=1024 * 1024, )
  n2 = np.random.normal(size=1024 * 1024, )

  p1 = np.exp(np.random.normal(size=1024 * 1024, ))
  p2 = np.exp(np.random.normal(size=1024 * 1024, ))

  a1, b1 = np.random.normal(size=2)
  a2, b2 = np.random.normal(size=2)

  ap, bp = np.random.exponential(size=2)
  an, bn = -np.random.exponential(size=2)

  print(a1, b1)
  print(a2, b2)

  def apply_op(op_name, op_f):
    operator, *args_sig = op_name.split('_')
    args = list()

    for arg in args_sig:
      if arg == 'c':
        args.append(np.random.normal())
      elif arg == 'cp':
        args.append(np.random.exponential())
      elif arg == 'n':
        args.append(np.random.normal(size=1024 * 1024, ))
      elif arg == 'p':
        args.append(np.exp(np.random.normal(size=1024 * 1024, )))
      elif arg == '1':
        pass
      else:
        raise ValueError()

    return op_f(*args)

  def pinv_p_cp(x1, a):
    std = np.exp(
      1.73978952 + 1.13020909 * np.log(a) + 0.13923955 * np.log(a) ** 2 - 0.00632442 * np.log(a) ** 3
    )
    # [-0.55992698 - 0.61744496 - 0.08477973  0.00216019  0.00083537]
    w1, w2 = 9.86342959e-01, 4.40520003e+00

    mean = np.exp(
      -np.log1p(a) + w1 * np.exp(-w2 * a)
    )
    return mean_log_normal / (x1 + a) / mean

  def ndiv_n_p_cp(x1, x2, a):
    w1, w2 = 0.99056203, 4.35480133

    std = np.exp(
      -np.log1p(a) + w1 * np.exp(-w2 * a)
    )
    return x1 / (x2 + a) / std

  def psqrt_p_cp(x, a):
    w1, w2 = -0.50227822, 0.21172428
    std = np.exp(
      -0.5 * np.log1p(a) + w1 * np.exp(-w2 * a)
    )
    w0, w1, w2, w3 = -0.23253458, 1., 0.342227, 5.
    mean = np.exp(
      w0 - 0.5 * np.log1p(a) + w2 * np.exp(-w3 * a)
    )
    return (np.sqrt(x + a) - np.sqrt(a)) * mean_log_normal / mean

  def psquare_p_cp(x, a):
    w0, w1 = 1.32698587, 13.79139551
    std = np.exp(w0 + np.log1p(w1 + a))
    return (np.square(x + a) - np.square(a)) * std_log_normal / std

  def ntanh_n_c(x, c):
    w0, w1 = 2.55590798, 0.52642439
    s = lambda x: (np.tanh(x) + 1) / 2
    std = w0 * s(w1 * c) * s(- w1 * c)
    mean = np.tanh(0.6372085 * c)

    return (np.tanh(x + c) - mean) / std

  def pexp_n_c_c(x, s, m):
    std = np.expm1(s ** 2) * np.exp(2 * m + s ** 2)
    mean = np.exp(m + 0.5 * s ** 2)

    if mean > std:
      return np.exp(x * s + m) * mean_log_normal / mean
    else:
      return np.exp(x * s + m) * std_log_normal / std

  def nlog_p_cp(x, m):
    mean = np.log(m + np.sqrt(2)) - 0.5 * np.log(2) * np.exp(-1.26318484 * m)
    std = 1.49033208 / (1.49033208 + m)
    return (np.log(x + m) - mean) / std

  ops = dict(
    nid_n=lambda x: x,
    pid_p=lambda x: x,
    nadd_p_p = lambda x1, x2: (x1 + x2 - 2 * mean_log_normal) / np.sqrt(2) / std_log_normal,
    padd_p_p = lambda x1, x2: (x1 + x2) / 2,

    nadd_n_n_1_c = lambda x1, x2, a:  (x1 + a * x2) / np.sqrt(1 + a ** 2),
    nadd_n_p_1_c = lambda x1, x2, a: (x1 + a * x2 - a * mean_log_normal) / np.sqrt(1 + a ** 2 * var_log_normal),
    nadd_p_p_1_c = lambda x1, x2, a: (x1 + a * x2 - (1 + a) * mean_log_normal) / std_log_normal / np.sqrt(1 + a ** 2),
    padd_p_p_1_cp=lambda x1, x2, a: (x1 + a * x2) / (1 + a),

    nmul_n_n_c_c = lambda x1, x2, a, b:  ((x1 + a) * (x2 + b) - a * b) / np.sqrt(1 + a ** 2 + b ** 2),
    nmul_n_p_c_c = lambda x1, x2, a, b: ((x1 + a) * (x2 + b) - a * (mean_log_normal + b)) / np.sqrt(
     a ** 2 * var_log_normal + (mean_log_normal + b) ** 2 + var_log_normal
    ),
    nmul_p_p_c_c = lambda x1, x2, a, b: ((x1 + a) * (x2 + b) - (mean_log_normal + a) * (mean_log_normal + b)) / std_log_normal / np.sqrt(
      (mean_log_normal + a) ** 2 + (mean_log_normal + b) ** 2 + var_log_normal
    ),
    pmul_p_p_cp_cp = lambda x1, x2, a, b: ((x1 + a) * (x2 + b) - a * b) / np.sqrt(
      (mean_log_normal + a) ** 2 + (mean_log_normal + b) ** 2 + var_log_normal
    ),
    pinv_p_cp=pinv_p_cp,
    ndiv_n_p_cp=ndiv_n_p_cp,

    nmul_n_n=lambda x1, x2: x1 * x2,
    nmul_n_p = lambda x1, x2: x1 * x2 / np.sqrt(mean_log_normal ** 2 + var_log_normal),
    nmul_p_p = lambda x1, x2: (x1 * x2 - mean_log_normal ** 2) / std_log_normal / np.sqrt(2 * mean_log_normal **2 + var_log_normal),
    pmul_p_p = lambda x1, x2: x1 * x2 / np.sqrt(2 * mean_log_normal **2 + var_log_normal),

    psquare_n_c = lambda x, c: np.square(x + c) * mean_log_normal / (1 + c ** 2),
    psquare_p_cp = psquare_p_cp,
    ntanh_n_c = ntanh_n_c,
    nlog_p_cp=nlog_p_cp
  )

  samples = dict()
  for op_name in ops:
    fs = apply_op(op_name, ops[op_name])
    samples[op_name] = fs

    m, s = np.mean(fs), np.std(fs)

    if op_name[0] == 'n':
      if abs(m) > 5.0e-2 or abs(s - 1) > 5.0e-2:
        warning = '(!)'
      else:
        warning = ''
    else:
      m, s = m / mean_log_normal, s / std_log_normal

      if abs(max(m, s) - 1) > 5.0e-2:
        warning = '(!)'
      else:
        warning = ''

    print(f'{op_name:<16}: {m:.2f} +- {s:.2f} {warning}')

  plt.figure(figsize=(16, 12))
  bins = np.linspace(-3, 6, num=100)
  for k, vs in samples.items():
    plt.hist(
      vs, bins=bins, histtype='step',
      label=f'{k} {np.mean(vs):.2f} ({np.mean(vs) / mean_log_normal:.2f} M), {np.std(vs):.2f} ({np.std(vs) / std_log_normal:.2f} V)'
    )
  plt.legend(loc='upper right')
  plt.savefig('ops.png')
  plt.close()

  biases = np.linspace(0, 16, num=128)
  means = np.ndarray(shape=biases.shape)
  stds = np.ndarray(shape=biases.shape)
  for i in range(biases.shape[0]):
      vs = np.log(p1 + biases[i])
      means[i] = np.mean(vs)
      stds[i] = np.std(vs)

  def f(p):
    a0, a1, a2, a3, a4 = p
    # 0.0007434044429235986
    # m = w0 - np.exp(-w1 - w2 * a_ - w3 * a_ ** 2)
    # 8.989977969166698e-05
    return a0 / (a0 + biases)

  def l(w):
    return np.mean(np.square(f(w) - stds)[16:-16]) #+ 1.0e-3 * np.sum(np.abs(w[1:]))

  import scipy.optimize as opt

  solution = opt.minimize(
    l, x0=[1, 1, 1, 1.0, 1.0], bounds=[(None, None), (0.0, None), (0.0, None), (None, None), (0.0, None)],
    tol=1.0e-9
  )
  print(solution.x, solution.fun)

  plt.figure(figsize=(9, 8))
  # plt.plot(biases, means)
  plt.plot(biases, stds)
  plt.plot(biases, f(solution.x), linestyle='--', color='black')
  plt.plot(biases, stds - f(solution.x), linestyle='--', color='black')
  plt.savefig('log-normal.png')
  plt.close()

def test_machine():
  library = (symgen.lib.core, symgen.lib.std, symgen.lib.stable)

  machine = symgen.StackMachine(*library, source='stable.c')

  signatures = dict(
    nadd_n_n='n=nn',
    nadd_n_p='n=np',
    nadd_p_p='n=pp',
    padd_p_p='p=pp',
    nadd_n_n_1_c='n=nnc',
    nadd_n_p_1_c='n=npc',
    nadd_p_p_1_w='n=ppc',
    padd_p_p_1_w='p=pp+',
    nmul_n_n='n=nn',
    nmul_n_p='n=np',
    nmul_p_p='n=pp',
    pmul_p_p='p=pp',
    nmul_n_n_c_c='n=nncc',
    nmul_n_p_c_c='n=npcc',
    nmul_p_p_c_c='n=ppcc',
    pmul_p_p_c_c='p=pp++',
    pinv_p_c='p=p+',
  )

  mean_log_normal = np.exp(0.5)
  var_log_normal = (np.exp(1) - 1) * np.exp(1)
  std_log_normal = np.sqrt((np.exp(1) - 1) * np.exp(1))

  n = 32
  m = 32 * 1024

  print(
    'C = ', machine.evaluate('(2) (1) (0) nadd_n_n_1_c', 2.0, 3.0, 5.0)
  )

  print()
  for op, sig in signatures.items():
    print('=' * 16 + f' {op:^16} ' + '=' * 16)
    result, args = sig.split('=')
    code = machine.assembly.assemble(' '.join([f'({i})' for i, _ in enumerate(args)][::-1]) + f' {op}')
    size = code.shape[0]
    code = np.concatenate([code] * n, axis=0)
    sizes = size * np.ones(n, dtype=np.int32)

    inputs = list()

    for arg in args:
      if arg == 'n':
        inputs.append(np.random.normal(size=(n, m)).astype(np.float32))
      elif arg == 'p':
        inputs.append(np.exp(np.random.normal(size=(n, m))).astype(np.float32))
      elif arg == '+':
        c = np.random.exponential(size=(n, )).astype(np.float32)
        c = np.broadcast_to(c[:, None], shape=(n, m))
        inputs.append(c)
      elif arg == 'c':
        c = np.random.normal(size=(n, )).astype(np.float32)
        c = np.broadcast_to(c[:, None], shape=(n, m))
        inputs.append(c)
      else:
        raise ValueError()

    inputs = np.stack(inputs, axis=-1)
    outputs = -999.0 * np.ones(shape=(n, m, 2), dtype=np.float32)

    machine.execute(code, sizes, inputs, outputs)

    assert np.all(outputs[:, :, 1] == -999.0)
    assert not np.any(outputs[:, :, 0] == -999.0)

    if op == 'nadd_n_n_1_c':
      assert np.all(np.abs(
        outputs[:, :, 0] - (inputs[:, :, 0] + inputs[:, :, 2] * inputs[:, :, 1]) / np.sqrt(1 + inputs[:, :, 2] * inputs[:, :, 2])
      ) < 1.0e-3)

    means, stds = np.mean(outputs[:, :, 0], axis=1), np.std(outputs[:, :, 0], axis=1)
    if result == 'p':
      means, stds = means / mean_log_normal, stds / std_log_normal
    print(f'{np.min(means):.2f} - {np.max(means):.2f} +- {np.min(stds):.2f} - {np.max(stds):.2f}')