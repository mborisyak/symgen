import math
import numpy as np
import symgen

def test_gen():
  n = 1024
  max_variables = 2
  instructions = np.zeros(shape=(n, 2), dtype=np.int32)
  sizes = np.zeros(shape=(n, ), dtype=np.int32)

  symgen.sym_gen.expr_gen(11, 13, instructions, sizes, max_variables)

  m = sizes[0]

  print(
    instructions[:m]
  )

def test_gen_compile():
  libraries = (symgen.lib.core, symgen.lib.std)

  grammar = {
    'expr': {
      ('const', ): 3,
      ('expr', 'expr', 'add'): 1,
      ('expr', 'expr', 'mul'): 1,
    }
  }

  generator = symgen.GeneratorMachine(*libraries, grammar=grammar, seed_symbol='expr', source='sym_gen.c', debug=True)
  print()
  for i in range(10):
    output = generator.generate(i, 3)
    print(generator.assembly.disassemble(output))

def test_grammar_hash():
  from symgen.generator import Grammar, symbol

  libraries = (symgen.lib.core, symgen.lib.std)

  grammar = Grammar(
    signatures=[
      symbol('expr', 'i', 'j'),
      symbol('multiply', ),
      symbol('constant')
    ],
    transitions={
      symbol('expr', 'i > 0'): {
        symbol('expr', 'i - 1') + symbol('expr', 'i - 1') + symbol('add'): 1,
        symbol('expr', 'i - 1') + symbol('expr', 'i - 1') + symbol('multiply'): 1,
        symbol('constant'): 2,
      },
      symbol('expr', 'i == 0'): {
        symbol('constant') + symbol('constant') + symbol('add'): 1,
        symbol('const') + symbol('constant') + symbol('mul'): 1,
        symbol('const'): 3,
      },
      symbol('multiply'): {
        symbol('mul'): 1
      },
      symbol('constant'): {
        symbol('const', 0.0): 1,
        symbol('const', 1.0): 5,
        symbol('const', math.pi): 5,
      }
    }
  )

  generator = symgen.GeneratorMachine(
    *libraries, grammar=grammar,
    seed_symbol=symbol('expr', '4', '-1'),
    source='sym_gen.c', debug=True
  )

  instructions, instruction_sizes = generator.generate(1, 2, max_depth=10, instruction_limit=1024, expression_limit=8)
  print(instructions)
  print(instruction_sizes)
  c = 0
  for s in instruction_sizes:
    print(generator.assembly.disassemble(instructions[c:c+s]))
    c += s

def test_tmp():
  import tempfile

  file = tempfile.NamedTemporaryFile('w', prefix='test', delete=True, delete_on_close=False)
  print(file.name)

  with open(file.name, 'w') as f:
    f.write('Hello!')