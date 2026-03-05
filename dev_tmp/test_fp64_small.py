import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

def test_fp64_small_matmul():
    # p=7 means 2*P = 16. Let's try 2*P = 8 (p=3)
    p = 3
    P = p + 1
    dim = 2 * P # 8
    
    def kernel(a_ref, b_ref, out_ref):
        # [8, 8] @ [8, 8] -> [8, 8]
        a = a_ref[...]
        b = b_ref[...]
        out_ref[...] = jnp.dot(a, b)

    a = jnp.ones((dim, dim), dtype=jnp.float64)
    b = jnp.ones((dim, dim), dtype=jnp.float64)
    
    print(f"Running fp64 matmul with dim={dim}...")
    try:
        res = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((dim, dim), jnp.float64),
            grid=(),
            compiler_params=plt.CompilerParams()
        )(a, b)
        print("Result sample:", res[0, 0])
        assert jnp.allclose(res, dim)
        print("SUCCESS")
    except Exception as e:
        print("FAILED")
        print(e)

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    test_fp64_small_matmul()
