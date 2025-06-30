import os

ROOT = os.path.dirname(__file__)

LIB_FLAGS = ['--shared', '-fPIC']
OPT_FLAGS = ['-O3', '-march=native', '-mtune=native']
#OPT_FLAGS = ['-O2']

__all__ = [
  'compile',
  'link',
  'ensure'
]

def compile(source_file: str, shared: bytes | str):
  import sysconfig
  import subprocess as sp

  import numpy as np

  include_dirs = [*sysconfig.get_paths()['include'].split(' '), np.get_include()]
  include_dirs = [f'-I{l}' for l in include_dirs]
  include_dirs.append(f'-I{ROOT}')

  cflags = sysconfig.get_config_vars().get('CFLAGS').split(' ')
  ldflags = sysconfig.get_config_vars().get('LDFLAGS').split(' ')

  call_arguments = ['gcc', source_file, *LIB_FLAGS, *OPT_FLAGS, *include_dirs, *ldflags, '-o', shared]

  process = sp.Popen(
    executable='gcc',
    args=call_arguments,
    stdout=sp.PIPE, stderr=sp.PIPE, stdin=None
  )

  stdout, stderr = process.communicate()
  if process.returncode != 0:
    print('===== arguments =====')
    print(' '.join(call_arguments))
    print('===== stdout =====')
    print(stdout.decode('utf-8'))
    print('===== stderr =====')
    print(stderr.decode('utf-8'))
    raise Exception('Compilation failed')

  return shared

def link(shared: bytes | str | None, name: str):
  import importlib.util
  spec = importlib.util.spec_from_file_location(name, shared)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)

  return module

def ensure(generate, get_hash, name, shared: bytes | str | None=None, source: bytes | str | None=None):
  import tempfile

  if shared is None or not os.path.exists(shared):
    code_hash, code = generate()

    if source is None:
      source_file = tempfile.NamedTemporaryFile(
        'w', suffix=f'{code_hash}.c', prefix=f'{name}-', delete=True
      )
      source = source_file.name

    with open(source, 'w') as f:
      f.write(code)

    if shared is None:
      shared_file = tempfile.NamedTemporaryFile('w+b', suffix=f'{code_hash}.so', prefix=f'{name}-', delete=True)
      shared = shared_file.name

    _ = compile(source, shared=shared)
  else:
    code_hash = get_hash()

  module = link(shared, name)
  assert module.hash() == code_hash, (
    'machine returned a different hash, it is likely it has been compiled with different flags'
  )

  return shared, module