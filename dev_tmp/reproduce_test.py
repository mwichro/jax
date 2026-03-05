
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
dim=8
def kernel(a_ref, b_ref, out_ref):
    out_ref[...] = jnp.dot(a_ref[...], b_ref[...])
a = jnp.ones((dim, dim), dtype=jnp.float64)
b = jnp.ones((dim, dim), dtype=jnp.float64)
pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((dim, dim), jnp.float64), grid=(), compiler_params=plt.CompilerParams())(a, b)

