
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
dim=8
def kernel(a_ref, b_ref, out_ref):
    a = a_ref[...]
    b = b_ref[...]
    # manual dot
    out_ref[...] = jnp.sum(a[:, None, :] * b.T[None, ...], axis=-1)
a = jnp.ones((dim, dim), dtype=jnp.float64)
b = jnp.ones((dim, dim), dtype=jnp.float64)
pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((dim, dim), jnp.float64), grid=(), compiler_params=plt.CompilerParams())(a, b)
print("SUCCESS manual dot!")

