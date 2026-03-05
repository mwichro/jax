
import jax.numpy as jnp
x_flat = jnp.arange(12).reshape(3, 4)
mat_T = jnp.arange(8).reshape(4, 2)
mat = mat_T.T # shape 2, 4
res1 = jnp.dot(x_flat, mat_T)
res2 = jnp.sum(x_flat[:, None, :] * mat[None, ...], axis=-1)
print(jnp.allclose(res1, res2))

