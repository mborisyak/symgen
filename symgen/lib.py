__all__ = [
  'core', 'std',

  'merge'
]

core = dict(
  const='${argument}.number',
  integer='(number_t) ${argument}.integer',
  input='${input}[${argument}.integer]',
  memory='${memory}[${argument}.integer]',
  store='${memory}[${argument}.integer] = pop(${stack});\n'
        'return;',
)

const = dict(
  one='1.0',
  pi='pi',
)

std = dict(
  add='const number_t a = pop(${stack});\n'
      'const number_t b = pop(${stack});\n'
      'a + b',
  sub='const number_t a = pop(${stack});\n'
      'const number_t b = pop(${stack});\n'
      'a - b',
  isub='const number_t a = pop(${stack});\n'
       'const number_t b = pop(${stack});\n'
       'b - a',
  neg='const number_t a = pop(${stack});\n'
      '-a',
  mul='const number_t a = pop(${stack});\n'
      'const number_t b = pop(${stack});\n'
      'a * b',
  div='const number_t a = pop(${stack});\n'
      'const number_t b = pop(${stack});\n'
      'a / b',
  idiv='const number_t a = pop(${stack});\n'
       'const number_t b = pop(${stack});\n'
       'b / a',
  inv='const number_t a = pop(${stack});\n'
      '1 / a',
  exp='const number_t a = pop(${stack});\n'
      'exp(a)',
  log='const number_t a = pop(${stack});\n'
      'log(a)',
  sqrt='const number_t a = pop(${stack});\n'
       'sqrt(a)',
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