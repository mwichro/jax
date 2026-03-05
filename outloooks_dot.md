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
