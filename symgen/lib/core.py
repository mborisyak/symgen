import jax.numpy as jnp

__all__ = [
  'core'
]

def const(*, value):
  return jnp.asarray(value)

def load(*, memory, address):
  return memory[address]

def store(x, *, memory, address):
  memory[address] = x
  return None

core = dict(
  const=const,
  load=load,
  store=store
)
