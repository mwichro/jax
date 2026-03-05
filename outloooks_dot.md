# Low-Level Investigation: `dot` / `dot_general` Lowering in Triton Pallas

## The Core Problem
The restriction comes from an artificial, hardcoded barrier in JAX’s lowering logic. In [jax/_src/pallas/triton/lowering.py](jax/_src/pallas/triton/lowering.py#L2321-L2322), the `_dot_general_lowering` function explicitly throws a `ValueError` for any matrices with dimensions less than 16:
```python
  if min(*b_type.shape) < 16:
    raise ValueError("all dimensions of b must be >= 16 ")
```
This check indiscriminately rejects perfectly valid smaller block sizes (and is also missing a check for `m` in `a_type.shape`).

## Upstream Triton natively supports this (especially fp64)
Triton's MLIR compiler backend is completely equipped to handle dimensions `< 16`:
* **FP64 Support**: In upstream Triton's `TritonGPUToLLVM` dialect, the hardware `MMA v2` generation uses smaller instruction blocks for FP64 compared to FP16. In `AccelerateMatmul.cpp` and `MMAv2.cpp`, Triton explicitly supports configurations like `m8n8k4` (`mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64`). So for FP64, the minimum dimensional requirement is actually **8 and 4**, not 16.
* **FMA Fallback**: Even if dimensions do not map smoothly onto Tensor Cores, Triton natively falls back to scalar FMA loops (`DotOpToLLVM/FMA.cpp`) to handle any arbitrary fractional sizes without crashing.

## What Needs to Be Done (The "Easy Way Out")

### A. Shallowest Fix: Relax the Thresholds
Since Triton handles smaller blocks, you can simply adjust the python MLIR emission rules in JAX. Update the check so that for `fp64`, it allows `< 16` dimensions (or drop it completely and let Triton handle the fallback to FMA if it cannot use MMA). 

### B. Optimal Fix: Pad inside the MLIR Lowering
If we want to guarantee that hardware Tensor Cores are always hit (padding up to 16 for standard precision, or multiples of 8/16 for FP64), you can pad the incoming Triton variables *before* calling `tt_dialect.dot`.
The changes would be confined solely to Python inside `jax/_src/pallas/triton/lowering.py`:
1. Use JAX's MLIR generator to instantiate a zero-tensor snippet (`tt_dialect.splat` or `arith.constant`).
2. Use **`tt_dialect.cat`** (Triton's `TT_CatOp`) to concatenate the block of zeros onto your original inputs `A` and `B` respectively until their dimensions hit the required hardware multiples.
3. Compute the `tt_dialect.dot` on the padded arrays.
4. *Slicing out the result*: Triton MLIR lacks a native in-place block-slice operation like `tensor.extract_slice`. Instead, you can leave the result padded in the registers. Since Pallas maps to `store` operations further down, as long as the memory write uses coordinate masks corresponding to the original unpadded dimensions (`m` and `n` via `tt_dialect.make_range`), the dummy padded data computes efficiently but is simply dumped and never touches memory.

## Summary
The required changes are **very shallow** (confined mostly to `jax/_src/pallas/triton/lowering.py`). No upstream modifications to Triton C++ MLIR internals are necessary. You can just bypass the JAX `< 16` safeguard or elegantly assemble `splat` + `cat` MLIR nodes to pad the boundaries and achieve peak fp64 shapes will naturally map to their proper Tensor Core mappings.

# Status Update (March 5, 2026)
* **Environment**: 4x A100 environment is configured and verified with JAX 0.9.x and local Triton/JAX builds.
* **FP64 Support**: Lowering rules in `jax/_src/pallas/triton/lowering.py` have been updated to support `fp64` accumulators, fixing precision loss where values were previously truncated to `f32`.
* **Dimension Relaxing**: The artificial 16x16x16 barrier was removed and replaced with a data-type-aware check.
* **Segfault Guard**: Identified a crash in Triton's `MMAv2.cpp` when `M < 16` or `K < 16` for `fp64`. Added a precise `ValueError` to JAX to prevent compiler segfaults while allowing valid smaller tiles for other dtypes.

## Analysis of $I \otimes M \otimes I \cdot u$ Packing
Yes, your understanding is correct. If you are applying a 1D operator $M$ (size 8x8) over a 3D grid $u$ (8x8x8), you can reshape $u$ to avoid the `dim < 16` hardware limitation without adding zero-padding.

Since the operator acts on one dimension and is an identity on the others, you can "fold" one of the identity dimensions into the batch or the contraction dimension:

*   **Batch Folding ($M > 16$):** Instead of treating each slice as an 8x8 matrix, you can reshape your input $u$ from `(8, 8, 8)` to `(8*8, 8)` i.e., `(64, 8)`. Now your $M \cdot u_{folded}$ operation has:
    *   $M = 64$ (Greater than 16)
    *   $K = 8$ (Native `m8n8k4` compatible)
    *   $N = 8$ (Native `m8n8k4` compatible)
*   **No Padding Needed:** Because the "Logical $M$" dimension is now 64, it satisfies the Triton `MMAv2` warp-tile requirement ($M \ge 16$). The hardware naturally parallelizes the 64 rows across the warps.
*   **Result:** This allows you to utilize the `fp64` Tensor Cores (`mma.m8n8k4`) at full throughput without any dummy calculations or `cat` operations.
*   **Implementation**: In Pallas, you can achieve this by using `at[i, j].load()` where the viewed block size is configured to be 16 or 32 instead of 8, effectively packing multiple logical elements into one Triton `tt.dot` call.

## The Bug in JAX primitive `dot`
During the investigation it was discovered that the `pl.dot` primitive systematically degraded `fp64` computations down to `fp32` accumulators within the lowering configuration. In `jax/_src/pallas/primitives.py`, the return element-type logic (`out_dtype`) was naively configured as `jnp.int32` for integer payloads, but blindly defaulted everything else to `jnp.float32`. This caused our test kernels attempting to execute valid FP64 dot products to generate an intermediate compilation output requiring `{f64, f64} -> f32`, which lacked a direct `mma` backend mapping on existing NVIDIA hardware, failing with: `error: Unsupported MMA instruction for the given mma type`. 

By patching `jax/_src/pallas/primitives.py` so the `out_dtype` calculation explicitly preserves `jnp.float64` out of an `fp64` dot product, Triton compiles straightforwardly down to `mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64` using `NumRegisters = {2, 1, 4}` within `MMAv2.cpp`.


