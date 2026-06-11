import jax.nn
import jax.numpy as jnp

__all__ = [
  'affine'
]

def affine_add(x, y, *, wx, wy, b):
  return wx * x + wy * y + b

def affine_mul(x, y, *, cx, cy):
  return (x + cx) * (y + cy)

def affine_div(x, y, *, cx, cy):
  return (x + cx) / (y + cy)

def affine_exp(x, *, w, c):
  return jnp.exp(w * x + c)

def affine_log(x, *, c, b):
  return jnp.log(x + c) + b

def affine_square(x, *, w, c):
  return jnp.square(w * x + c)

def affine_tanh(x, *, w, b):
  return jnp.tanh(w * x + b)

def affine_softplus(x, *, w, b):
  return jax.nn.softplus(w * x + b)

def affine_sqrt(x, *, w, b):
  return jnp.sqrt(w * x + b)

def affine_cbrt(x, *, w, b):
  return jnp.cbrt(w * x + b)

def affine_sin(x, *, w, b):
  return jnp.sin(w * x + b)

def affine_id(x, *, w, b):
  return w * x + b

affine = {
  'affine_add': affine_add,
  'affine_mul': affine_mul,
  'affine_div': affine_div,
  'affine_exp': affine_exp,
  'affine_log': affine_log,
  'affine_square': affine_square,
  'affine_tanh': affine_tanh,
  'affine_softplus': affine_softplus,
  'affine_sqrt': affine_sqrt,
  'affine_cbrt': affine_cbrt,
  'affine_sin': affine_sin,
  'affine_id': affine_id,
}
