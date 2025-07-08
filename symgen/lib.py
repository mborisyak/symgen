import math

__all__ = [
  'core', 'std',

  'merge'
]


core = dict(
  const='${argument}.number',
  integer='(number_t) ${argument}.integer',
  input='${input}',
  memory='${memory}',
  store='${memory} = POP();\n'
        'return;',
)

std = dict(
  add='const number_t a = POP();\n'
      'const number_t b = POP();\n'
      'a + b',
  sub='const number_t a = POP();\n'
      'const number_t b = POP();\n'
      'a - b',
  isub='const number_t a = POP();\n'
       'const number_t b = POP();\n'
       'b - a',
  neg='const number_t a = POP();\n'
      '-a',
  mul='const number_t a = POP();\n'
      'const number_t b = POP();\n'
      'a * b',
  div='const number_t a = POP();\n'
      'const number_t b = POP();\n'
      'a / b',
  idiv='const number_t a = POP();\n'
       'const number_t b = POP();\n'
       'b / a',
  inv='const number_t a = POP();\n'
      '1 / a',
  exp='const number_t a = POP();\n'
      'exp(a)',
  log='const number_t a = POP();\n'
      'log(a)',
  sqrt='const number_t a = POP();\n'
       'sqrt(a)',
  square='const number_t a = POP();\n'
         'a * a',
  cube='const number_t a = POP();\n'
       'a * a * a',
  softplus='const number_t a = POP();\n'
           'a > 0 ? a + log1p(exp(-a)) : log1p(exp(a))',
  tanh='const number_t a = POP();\n'
       'tanhf(a);',
  sigmoid='const number_t a = POP();\n'
          '0.5 + 0.5 * tanhf(a);',
  erf='erf(POP())',
)

stable = dict(
  nadd_n_n=(
    'const number_t n1 = POP();\n'
    'const number_t n2 = POP();\n'
    'INV_SQRT_2 * (n1 + n2)'
  ),
  nadd_n_p=(
    'const number_t n = POP();\n'
    'const number_t p = POP();\n'
    '(n + p - MEAN_LOG_NORMAL) / sqrt(1 + VAR_LOG_NORMAL)'
  ),
  nadd_p_p=(
    'const number_t p1 = POP();\n'
    'const number_t p2 = POP();\n'
    '(p1 + p2 - 2 * MEAN_LOG_NORMAL) * INV_SQRT_2 / STD_LOG_NORMAL'
  ),
  padd_p_p=(
    'const number_t p1 = POP();\n'
    'const number_t p2 = POP();\n'
    '0.5 * (p1 + p2)'
  ),

  nadd_n_n_1_c=(
    'const number_t n1 = POP();\n'
    'const number_t n2 = POP();\n'
    'const number_t w2 = POP();\n'
    '(n1 + w2 * n2) / sqrt(1 + w2 * w2)'
  ),
  nadd_n_p_1_c=(
    'const number_t n = POP();\n'
    'const number_t p = POP();\n'
    'const number_t w = POP();\n'
    '(n + w * p - w * MEAN_LOG_NORMAL) / sqrt(1 + w * w * VAR_LOG_NORMAL)'
  ),
  nadd_p_p_1_c=(
    'const number_t p1 = POP();\n'
    'const number_t p2 = POP();\n'
    'const number_t c = POP();\n'
    '(p1 + c * p2 - (1 + c) * MEAN_LOG_NORMAL) / STD_LOG_NORMAL / sqrt(1 + c * c)'
  ),
  padd_p_p_1_c=(
    'const number_t p1 = POP();\n'
    'const number_t p2 = POP();\n'
    'const number_t c = POP();\n'
    '(p1 + c * p2) / (1 + c)'
  ),
  nmul_n_n=(
    'const number_t n1 = POP();\n'
    'const number_t n2 = POP();\n'
    'n1 * n2'
  ),
  nmul_n_p=(
    'const number_t n = POP();\n'
    'const number_t p = POP();\n'
    'n * p / sqrt(MEAN_LOG_NORMAL * MEAN_LOG_NORMAL + VAR_LOG_NORMAL)'
  ),
  nmul_p_p=(
    'const number_t p1 = POP();\n'
    'const number_t p2 = POP();\n'
    'const number_t z = sqrt(2 * MEAN_LOG_NORMAL * MEAN_LOG_NORMAL + VAR_LOG_NORMAL);\n'
    '(p1 * p2 - MEAN_LOG_NORMAL * MEAN_LOG_NORMAL) / STD_LOG_NORMAL / z'
  ),
  pmul_p_p=(
    'const number_t p1 = POP();\n'
    'const number_t p2 = POP();\n'
    'p1 * p2 / sqrt(2 * MEAN_LOG_NORMAL * MEAN_LOG_NORMAL + VAR_LOG_NORMAL)'
  ),
  nmul_n_n_c_c=(
    'const number_t n1 = POP();\n'
    'const number_t n2 = POP();\n'
    'const number_t c1 = POP();\n'
    'const number_t c2 = POP();\n'
    '((n1 + c1) * (n2 + c2) - c1 * c2) / sqrt(1 + c1 * c1 + c2 * c2)'
  ),
  nmul_n_p_c_c=(
    'const number_t n = POP();\n'
    'const number_t p = POP();\n'
    'const number_t c_n = POP();\n'
    'const number_t c_p = POP();\n'
    'const number_t z = sqrt(\n'
    '  c_n * c_n * VAR_LOG_NORMAL + (MEAN_LOG_NORMAL + c_p) * (MEAN_LOG_NORMAL + c_p) + VAR_LOG_NORMAL\n'
    ');\n'
    '((n + c_n) * (p + c_p) - c_n * (MEAN_LOG_NORMAL + c_p)) / z'
  ),
  nmul_p_p_c_c=(
    'const number_t p1 = POP();\n'
    'const number_t p2 = POP();\n'
    'const number_t c1 = POP();\n'
    'const number_t c2 = POP();\n'
    'const number_t z = sqrt(\n'
    '  (MEAN_LOG_NORMAL + c1) * (MEAN_LOG_NORMAL + c1) +\n'
    '  (MEAN_LOG_NORMAL + c2) * (MEAN_LOG_NORMAL + c2) +\n'
    '  VAR_LOG_NORMAL\n'
    ');\n'
    '((p1 + c1) * (p2 + c2) - (MEAN_LOG_NORMAL + c1) * (MEAN_LOG_NORMAL + c2)) / STD_LOG_NORMAL / z'
  ),
  pmul_p_p_c_c=(
    'const number_t p1 = POP();\n'
    'const number_t p2 = POP();\n'
    'const number_t c1 = POP();\n'
    'const number_t c2 = POP();\n'
    'const number_t z = sqrt(\n'
    '  (MEAN_LOG_NORMAL + c1) * (MEAN_LOG_NORMAL + c1) +\n'
    '  (MEAN_LOG_NORMAL + c2) * (MEAN_LOG_NORMAL + c2) +\n'
    '  VAR_LOG_NORMAL\n'
    ');\n'
    '((p1 + c1) * (p2 + c2) - c1 * c2) / z'
  ),
  pinv_p_c=(
    'const number_t p = POP();\n'
    'const number_t c = POP();\n'
    'pinv_p_c(p, c)'
  ),
  ndiv_n_p_c=(
    'const number_t n = POP();\n'
    'const number_t p = POP();\n'
    'const number_t c = POP();\n'
    'ndiv_n_p_c(n, p, c)'
  ),
  psquare_n_c=(
    'const number_t n = POP();\n'
    'const number_t c = POP();\n'
    '(n + c) * (n + c) * MEAN_LOG_NORMAL / (1 + c * c)'
  ),
  psqrt_p_c=(
    'const number_t p = POP();\n'
    'const number_t c = POP();\n'
    'psqrt_p_c(p, c)'
  ),
  ntanh_n_c=(
    'const number_t n = POP();\n'
    'const number_t c = POP();\n'
    'ntanh_n_c(n, c)'
  ),

  nlog_p_c=(
    'const number_t n = POP();\n'
    'const number_t c = POP();\n'
    'nlog_p_c(n, c)'
  ),
)


def merge(*libraries: dict[str, str]):
  library = dict()

  for lib in libraries:
    for k in lib:
      k_lower = k.lower()

      if k_lower in library:
        raise ValueError(f'operator {k_lower} is already in the library')

      library[k_lower] = lib[k]

  return library