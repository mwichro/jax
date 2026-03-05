import time
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from functools import partial

# -----------------------------------------------------------------------------
# 1. The problematic exact contraction logic
# -----------------------------------------------------------------------------
def dot_along_axis(mat, x, axis, precision=None):
    """Core swapaxes+reshape+dot contraction. Works for any dtype that Triton supports.

    Args:
        precision: JAX matmul precision (`None` keeps the platform default,
            which on NVIDIA Ampere+ GPUs means TF32 for fp32 operands).
            Pass `jax.lax.Precision.HIGHEST` to force true fp32 arithmetic.
    """
    ndim = x.ndim
    if axis == ndim - 1:
        x_flat = x.reshape(-1, x.shape[-1])
        res = jnp.dot(x_flat, mat.T, precision=precision)
        return res.reshape(x.shape)
    else:
        x_trans = jnp.swapaxes(x, axis, -1)
        x_flat = x_trans.reshape(-1, x_trans.shape[-1])
        res = jnp.dot(x_flat, mat.T, precision=precision)
        res_trans = res.reshape(x_trans.shape)
        return jnp.swapaxes(res_trans, axis, -1)


# -----------------------------------------------------------------------------
# 2. Minimal Pallas Kernel Structure
# -----------------------------------------------------------------------------
def pallas_patch_kernel(u_ref, f_ref, out_ref, p, block_size):
    pid = pl.program_id(0)
    start_idx = pid * block_size
    
    # 1. READ: Load to SRAM
    b_idx = pl.dslice(start_idx, block_size)
    u_patch = u_ref[b_idx, ...]
    f_patch = f_ref[b_idx, ...]
    
    P = p + 1
    # u_patch shape: [block_size, 8, P*P*P]
    # Reshape and transpose to memory-contiguous 3D layouts
    u_vol = u_patch.reshape(-1, 2, 2, 2, P, P, P)
    u_vol = u_vol.transpose(0, 1, 4, 2, 5, 3, 6)
    u_patch_3d = u_vol.reshape(-1, 2 * P, 2 * P, 2 * P)
    
    f_vol = f_patch.reshape(-1, 2, 2, 2, P, P, P)
    f_vol = f_vol.transpose(0, 1, 4, 2, 5, 3, 6)
    f_patch_3d = f_vol.reshape(-1, 2 * P, 2 * P, 2 * P)

    # Static matrices for application
    D = jnp.ones((2 * P, 2 * P), dtype=u_patch.dtype)
    M = jnp.ones((2 * P, 2 * P), dtype=u_patch.dtype)

    # 2. COMPUTE: 9x apply_1d (simulates tensor-product evaluation)
    t1 = dot_along_axis(M, dot_along_axis(M, dot_along_axis(D, u_patch_3d, 3), 2), 1)
    t2 = dot_along_axis(M, dot_along_axis(D, dot_along_axis(M, u_patch_3d, 3), 2), 1)
    t3 = dot_along_axis(D, dot_along_axis(M, dot_along_axis(M, u_patch_3d, 3), 2), 1)
    
    res_vol = (t1 + t2 + t3) - f_patch_3d

    # Reverse permutations
    res_back = res_vol.reshape(-1, 2, P, 2, P, 2, P)
    res_back = res_back.transpose(0, 1, 3, 5, 2, 4, 6)
    correction = res_back.reshape(-1, 8, P * P * P)
    
    # 3. WRITE: back to global memory
    out_ref[b_idx, ...] = correction

@partial(jax.jit, static_argnums=(1, 2))
def run_minimal_smoother(u_global, p, block_size):
    n_patches = u_global.shape[0]
    grid_size = n_patches // block_size
    
    # Dummy RHS f vector directly represented
    f_global = jnp.ones_like(u_global)

    kernel = lambda u, f, out: pallas_patch_kernel(u, f, out, p=p, block_size=block_size)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(u_global.shape, u_global.dtype),
        grid=(grid_size,),
        interpret=False,
        compiler_params=plt.CompilerParams()
    )(u_global, f_global)

# -----------------------------------------------------------------------------
# 3. Benchmark driver
# -----------------------------------------------------------------------------
def main():
    # Setup problem exactly parameters as the original failing bench
    p = 3
    n_patches = 1000 * 64  # Scale to generate enough work (8000 patches)
    block_size = 1
    dtype = jnp.float32    # Issue frequently surfaces in high-precision unrolling

    # jax.config.update("jax_enable_x64", True)
    
    P = p + 1
    print(f"JAX Version: {jax.__version__}")
    print(f"Patches: {n_patches}, Polynomial Degree: P={p}")
    print(f"Running compilation...")

    # Allocate variables
    u_global = jnp.ones((n_patches, 8, P**3), dtype=dtype)

    # Warmup and compile
    res = run_minimal_smoother(u_global, p, block_size).block_until_ready()
    print("Compilation finished. Running benchmarks...")

    # Timings
    n_runs = 50
    start_time = time.perf_counter()
    for _ in range(n_runs):
        res = run_minimal_smoother(u_global, p, block_size)
    res.block_until_ready()
    end_time = time.perf_counter()

    avg_time_ms = (end_time - start_time) * 1000 / n_runs
    print(f"Average execution time over {n_runs} runs: {avg_time_ms:.3f} ms")

if __name__ == "__main__":
    main()