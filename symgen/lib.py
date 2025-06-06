__all__ = [
  'core', 'std'
]

core = dict(
  const='${argument}.number',
  integer='(number_t) ${argument}.integer',
  input='${input}[${argument}.integer]',
  memory='${memory}[${argument}.integer]',
  store='${memory}[${argument}.integer] = pop(${stack});\n'
        'return;',
)

std = dict(
  add='const number_t a = pop(${stack});\n'
      'const number_t b = pop(${stack});\n'
      'a + b',
  sub='const number_t a = pop(${stack});\n'
      'const number_t b = pop(${stack});\n'
      'a - b',
  neg='const number_t a = pop(${stack});\n'
      '-a',
  mul='const number_t a = pop(${stack});\n'
      'const number_t b = pop(${stack});\n'
      'a * b',
  div='const number_t a = pop(${stack});\n'
      'const number_t b = pop(${stack});\n'
      'a / b',
  inv='const number_t a = pop(${stack});\n'
      '1 / a',
)