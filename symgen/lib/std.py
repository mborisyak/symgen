import jax.numpy as jnp

__all__ = [
  'std'
]

std = dict(
  add=lambda x, y: jnp.add(x, y),
  sub=lambda x, y: jnp.subtract(x, y),
  mul=lambda x, y: jnp.multiply(x, y),
  div=lambda x, y: jnp.divide(x, y),
  neg=lambda x: jnp.negative(x),
  inv=lambda x: jnp.reciprocal(x),
  log=lambda x: jnp.log(x),
  exp=lambda x: jnp.exp(x),
  sqrt=lambda x: jnp.sqrt(x),
  square=lambda x: jnp.square(x),
)
