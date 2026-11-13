# Attention benchmark

Results for the [PyTorch attention benchmarking problem](../cs336_assignment2_systems.pdf),
using `Attention` from `model.py` and both classes in `flashattention.py`.

**Main finding:** with either causal or noncausal masking, `model.py` ran out
of memory at 16,384 tokens for every tested dimension. Triton FlashAttention
completed the whole grid. Causal masking substantially reduced Triton's
forward time at long sequence lengths.

## Setup and measurements

- **GPU:** NVIDIA GeForce RTX 3090 (24 GiB).
- **Software:** Python 3.13.15, PyTorch 2.11.0+cu130, Triton 3.6.0.
- **Inputs:** random float32 Q, K, V of shape `(8, N, D)`; no head dimension.
- **Attention:** both causal and noncausal for `model.py` and Triton. Causal
  means a query at position `i` may attend only to keys at positions `j ≤ i`.
  `model.py` receives a lower-triangular boolean mask; Triton receives
  `is_causal=True`. Noncausal `model.py` receives an all-true mask. The tiled
  PyTorch FlashAttention implementation does not support causal mode.
- **Timing:** two untimed forward/backward warmups, then the stated number
  of forward and backward repetitions. Each pass ends with
  `torch.cuda.synchronize()`. Each backward uses a fresh, untimed forward.

The tables report mean time per pass. **Pre-BW** is live tensor memory
allocated immediately after forward and before backward. **Delta** is the
increase from immediately before that forward. Both use
`torch.cuda.memory_allocated()` and are measured in MiB.
Pre-BW includes the boolean mask held by `model.py`; Triton does not
allocate an explicit mask. The mask is created before the memory baseline,
so it is excluded from Delta.

## Causal: `model.py` versus Triton

All successful rows use **100 forward and 100 backward passes**. Each pair
in the last column reports live memory immediately before backward, first
for `model.py`, then for Triton.

| N × D | Model fwd / bwd (ms) | Triton fwd / bwd (ms) | Pre-BW model / Triton (MiB) |
|---:|---:|---:|---:|
| 256 × 16 | 0.60 / 2.34 | 0.15 / 0.97 | 23.02 / 0.63 |
| 256 × 32 | 0.60 / 2.34 | 0.15 / 0.97 | 23.64 / 1.26 |
| 256 × 64 | 0.48 / 2.37 | 0.14 / 0.72 | 24.89 / 2.51 |
| 256 × 128 | 0.65 / 2.38 | 0.15 / 0.99 | 27.39 / 5.01 |
| 1,024 × 16 | 0.91 / 2.70 | 0.19 / 0.72 | 116.81 / 2.53 |
| 1,024 × 32 | 0.90 / 2.86 | 0.20 / 0.82 | 119.31 / 5.03 |
| 1,024 × 64 | 0.93 / 2.67 | 0.24 / 1.06 | 124.31 / 10.03 |
| 1,024 × 128 | 0.96 / 2.88 | 0.32 / 2.02 | 134.31 / 20.03 |
| 4,096 × 16 | 11.69 / 35.21 | 0.88 / 4.90 | 1594.50 / 10.12 |
| 4,096 × 32 | 11.89 / 35.49 | 1.11 / 4.79 | 1604.50 / 20.12 |
| 4,096 × 64 | 12.17 / 35.84 | 1.47 / 8.89 | 1624.50 / 40.12 |
| 4,096 × 128 | 13.06 / 36.83 | 2.63 / 22.60 | 1664.50 / 80.12 |
| 8,192 × 16 | 45.39 / 137.29 | 3.09 / 18.99 | 6308.75 / 20.25 |
| 8,192 × 32 | 46.31 / 138.40 | 3.83 / 18.68 | 6328.75 / 40.25 |
| 8,192 × 64 | 47.53 / 140.14 | 5.47 / 34.58 | 6368.75 / 80.25 |
| 8,192 × 128 | 51.56 / 144.79 | 10.12 / 91.63 | 6448.75 / 160.25 |
| 16,384 × 16 | **OOM** | 12.16 / 76.34 | — / 40.50 |
| 16,384 × 32 | **OOM** | 14.89 / 75.27 | — / 80.50 |
| 16,384 × 64 | **OOM** | 21.52 / 140.19 | — / 160.50 |
| 16,384 × 128 | **OOM** | 45.72 / 372.68 | — / 320.50 |

At N = 8,192 and D = 64, causal Triton was **8.68× faster forward**
and **4.05× faster backward** than causal `model.py`. Its pre-backward
allocation was **80.25 MiB**, compared with **6,368.75 MiB** for the model.
At N = 16,384 and D = 64, the model failed, while Triton completed a
forward in **21.52 ms** and a backward in **140.19 ms**.

The lower-triangular mask does not shrink the dense score tensors made by
`model.py`, so its causal and noncausal timings and memory are similar.
The Triton forward kernel stops visiting key tiles after the last tile that
could contain an allowed key. At N = 8,192 and D = 64, its forward time fell
from **10.47 ms noncausal** to **5.47 ms causal**. Its two backward kernels
still visit all query/key tile pairs and mask disallowed scores, so their
times stayed close: **34.75 ms noncausal** and **34.58 ms causal**.

Before benchmarking, the two causal implementations' outputs and Q/K/V
gradients were checked on the same random `(8, 256, 64)` inputs with
`rtol=1e-2, atol=1e-2`; all comparisons passed. The full causal measurements,
including forward memory deltas and OOM errors, are in
[`attention_causal_results_100.csv`](attention_causal_results_100.csv).

## Noncausal: `model.py`

All successful rows use **100 forward and 100 backward passes**.

| N × D | Fwd / bwd (ms) | Pre-BW (MiB) | Delta (MiB) |
|---:|---:|---:|---:|
| 256 × 16 | 0.47 / 1.95 | 23.02 | 6.20 |
| 256 × 32 | 0.47 / 1.66 | 23.64 | 6.33 |
| 256 × 64 | 0.49 / 2.29 | 24.89 | 6.58 |
| 256 × 128 | 0.49 / 2.26 | 27.39 | 7.08 |
| 1,024 × 16 | 0.87 / 2.82 | 116.81 | 97.56 |
| 1,024 × 32 | 0.87 / 2.82 | 119.31 | 98.06 |
| 1,024 × 64 | 0.89 / 2.84 | 124.31 | 99.06 |
| 1,024 × 128 | 0.95 / 2.87 | 134.31 | 101.06 |
| 4,096 × 16 | 11.73 / 35.40 | 1594.50 | 1554.25 |
| 4,096 × 32 | 11.93 / 35.62 | 1604.50 | 1556.25 |
| 4,096 × 64 | 12.19 / 36.03 | 1624.50 | 1560.25 |
| 4,096 × 128 | 13.10 / 36.78 | 1664.50 | 1568.25 |
| 8,192 × 16 | 45.56 / 137.92 | 6308.75 | 6212.50 |
| 8,192 × 32 | 46.69 / 139.25 | 6328.75 | 6216.50 |
| 8,192 × 64 | 47.68 / 140.83 | 6368.75 | 6224.50 |
| 8,192 × 128 | 51.39 / 145.59 | 6448.75 | 6240.50 |
| 16,384 × 16 | **OOM** | — | — |
| 16,384 × 32 | **OOM** | — | — |
| 16,384 × 64 | **OOM** | — | — |
| 16,384 × 128 | **OOM** | — | — |

## Noncausal: `flashattention.py` Triton

All rows use **100 forward and 100 backward passes**.

| N × D | Fwd / bwd (ms) | Pre-BW (MiB) | Delta (MiB) |
|---:|---:|---:|---:|
| 256 × 16 | 0.15 / 0.57 | 0.63 | 0.13 |
| 256 × 32 | 0.15 / 0.90 | 1.26 | 0.26 |
| 256 × 64 | 0.15 / 0.57 | 2.51 | 0.51 |
| 256 × 128 | 0.14 / 0.60 | 5.01 | 1.01 |
| 1,024 × 16 | 0.22 / 0.76 | 2.53 | 0.53 |
| 1,024 × 32 | 0.24 / 0.75 | 5.03 | 1.03 |
| 1,024 × 64 | 0.30 / 1.01 | 10.03 | 2.03 |
| 1,024 × 128 | 0.46 / 2.01 | 20.03 | 4.03 |
| 4,096 × 16 | 1.51 / 4.81 | 10.12 | 2.12 |
| 4,096 × 32 | 1.85 / 4.76 | 20.12 | 4.12 |
| 4,096 × 64 | 2.64 / 8.87 | 40.12 | 8.12 |
| 4,096 × 128 | 4.94 / 22.90 | 80.12 | 16.12 |
| 8,192 × 16 | 5.90 / 18.69 | 20.25 | 4.25 |
| 8,192 × 32 | 7.17 / 18.46 | 40.25 | 8.25 |
| 8,192 × 64 | 10.47 / 34.75 | 80.25 | 16.25 |
| 8,192 × 128 | 19.49 / 92.52 | 160.25 | 32.25 |
| 16,384 × 16 | 24.14 / 74.72 | 40.50 | 8.50 |
| 16,384 × 32 | 29.14 / 73.78 | 80.50 | 16.50 |
| 16,384 × 64 | 42.19 / 141.02 | 160.50 | 32.50 |
| 16,384 × 128 | 78.68 / 369.85 | 320.50 | 64.50 |

Raw data for both full-grid tables:
[`attention_results_100.csv`](attention_results_100.csv).

## Noncausal: `flashattention.py` tiled PyTorch

The tile loops are much slower, so larger cases have fewer passes.
The **Passes** column shows the count used for both forward and backward.

| N × D | Passes | Fwd / bwd (ms) | Pre-BW (MiB) | Delta (MiB) |
|---:|---:|---:|---:|---:|
| 256 × 16 | 100 | 33.08 / 76.47 | 16.88 | 0.13 |
| 256 × 32 | 100 | 31.42 / 71.08 | 17.51 | 0.26 |
| 256 × 64 | 100 | 32.16 / 71.46 | 18.76 | 0.51 |
| 256 × 128 | 100 | 31.63 / 74.32 | 21.26 | 1.01 |
| 1,024 × 16 | 10 | 474.04 / 1063.73 | 18.78 | 0.53 |
| 1,024 × 32 | 10 | 483.39 / 1096.38 | 21.28 | 1.03 |
| 1,024 × 64 | 10 | 499.24 / 1120.12 | 26.28 | 2.03 |
| 1,024 × 128 | 10 | 480.53 / 1166.47 | 36.28 | 4.03 |
| 4,096 × 64 | 1 | 7395.97 / 18599.45 | 56.38 | 8.12 |

Raw data:
[256 tokens](flash_torch_256_100.csv),
[1,024 tokens](flash_torch_1024_10.csv), and
[4,096 tokens](flash_torch_4096_1.csv).

At 4,096 × 64, a single forward took 7.40 seconds and a single backward
took 18.60 seconds. A 100-pass measurement would take roughly an hour
once the extra forwards needed to create backward graphs are included.
Larger tiled PyTorch cases were not run.

## Memory accounting

The first OOM in the requested grid is **N = 16,384, D = 16** for
`model.py`. One float32 score tensor contains
`8 × 16,384² × 4` bytes = **8,192 MiB (8 GiB)**. The boolean mask
adds **256 MiB**; Q, K, and V together add **24 MiB** at D = 16.
`model.py` also creates score-sized tensors for masking and its explicit
softmax. The OOM occurred while trying to allocate another 8 GiB tensor.

At N = 8,192 and D = 16, the measured forward delta was **6,212.5 MiB**:
roughly three 2,048 MiB score tensors plus smaller allocations. Doubling
N approximately quadruples this retained memory, taking it beyond the
GPU capacity. This is why backward-related memory grows approximately
with sequence length squared. Tiled attention avoids retaining the full
score matrix. At N = 16,384 and D = 16, Triton retained only **8.5 MiB**
beyond its input baseline.

These memory figures describe tensors tracked by the PyTorch allocator.
They exclude some CUDA context and external-library allocations; see
[PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management).

## Reproduce

Run from `assignment2-systems` with its uv environment. Each case runs
in a separate process, and each finished result is written to CSV.

Full 100-pass grid for `model.py` and Triton:

```sh
uv run --frozen python benchmarks/attention_benchmark.py \
  --output benchmarks/attention_results_100.csv
```

Full 100-pass causal grid for `model.py` and Triton:

```sh
uv run --frozen python benchmarks/attention_benchmark.py --causal \
  --output benchmarks/attention_causal_results_100.csv
```

Tiled PyTorch sample grids:

```sh
uv run --frozen python benchmarks/attention_benchmark.py \
  --implementations flash_torch --lengths 256 \
  --output benchmarks/flash_torch_256_100.csv

uv run --frozen python benchmarks/attention_benchmark.py \
  --implementations flash_torch --lengths 1024 --repetitions 10 \
  --output benchmarks/flash_torch_1024_10.csv

uv run --frozen python benchmarks/attention_benchmark.py \
  --implementations flash_torch --lengths 4096 --dims 64 \
  --repetitions 1 --warmup 1 \
  --output benchmarks/flash_torch_4096_1.csv
```
