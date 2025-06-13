import symgen
import numpy as np

def test_eval():
  machine = symgen.StackMachine(
    symgen.lib.core, symgen.lib.std, debug=True
  )
  result = machine.evaluate('(0) 1.5 add {0} [0] [0] mul', 1.0)

  print(result)

def test_expression():
  assembly = symgen.assembly.Assembly(symgen.lib.core, symgen.lib.std)
  code = '(0) 1.5 add {0} [0] [0] mul'
  machine_code = assembly.assemble(code)
  print(machine_code)
  recovered = [assembly.op_names[i] for i in machine_code[:, 0]]
  assert recovered == ['input', 'const', 'add', 'store', 'memory', 'memory', 'mul']

  print(assembly.disassemble(machine_code))
