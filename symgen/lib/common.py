from ..operation import Operation

__all__ = [
  'merge'
]

def merge(*libraries: dict[str, Operation]) -> dict[str, Operation]:
  library = dict()

  for lib in libraries:
    for k in lib:
      k_lower = k.lower()

      if k_lower in library:
        raise ValueError(f'operator {k_lower} is already in the library')

      library[k_lower] = lib[k]

  return library