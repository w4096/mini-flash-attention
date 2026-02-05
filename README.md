# Mini Flash Attention

A minimal, educational implementation of Flash Attention v2 in CUDA. This project demonstrates the core concepts of Flash Attention with a focus on code clarity and understanding.

## What's This?

Flash Attention is a fast and memory-efficient attention algorithm that makes training large models more practical. This is a simplified version that keeps the core ideas while being easier to understand and modify.

**Key features:**

- Flash Attention v2 algorithm with tiling
- CUDA Tensor Cores for matrix operations
- FP16 and BF16 support
- Works seamlessly with PyTorch

## Getting Started

**Requirements:**

- CUDA 11.8+ (tested on CUDA 12.8)
- Python 3.8+
- PyTorch 2.0+
- NVIDIA Ampere GPU or newer (SM 80+)

**Build:**
```bash
git clone https://github.com/w4096/mini-flash-attention.git
cd mini-flash-attention
git submodule update --init --recursive

python setup.py build
export PYTHONPATH=$(pwd)/build/lib.linux-x86_64-cpython-312
```

## Quick Example

```python
import torch
from mini_flash_attention import flash_attn_func

# Standard batched attention
q = torch.randn(2, 1024, 8, 128, device='cuda', dtype=torch.float16)
k = torch.randn(2, 1024, 8, 128, device='cuda', dtype=torch.float16)
v = torch.randn(2, 1024, 8, 128, device='cuda', dtype=torch.float16)

output = flash_attn_func(q, k, v)
```

## Performance

Benchmarked on NVIDIA GPU by running the benchmark script in `benchmark/run.py`.

Here are the results for a forward pass with `batch_size=10`, `seqlen=4096`, `dim=128`, and `heads=28`:

| Implementation | CPU Time | CUDA Time |
|----------------|----------|-----------|
| **Mini Flash Attention** | 41.606ms | **41.547ms** |
| PyTorch Attention | 44.098ms | 42.882ms |
| Flash Attention (official) | 43.890ms | 42.692ms |

All implementations produce nearly identical results:

```
Max difference between mini-flash-attn and flash-attn: 3.814697265625e-06
Max difference between torch and flash-attn: 3.814697265625e-06
Max difference between torch and mini-flash-attn: 3.814697265625e-06
```

Run `python benchmark/run.py` to test on your hardware.

## How It Works

The implementation splits Q, K, V into tiles and processes them in chunks:

1. Load a tile of Q into shared memory
2. Loop through K/V tiles, computing attention scores incrementally  
3. Use online softmax to maintain numerical stability
4. Accumulate the output on-the-fly

This approach keeps memory usage low while being fast thanks to:
- CUDA Tensor Cores for matrix multiplications
- Shared memory tiling to reduce bandwidth
- Async memory copies to overlap computation
- Swizzled memory layouts to avoid bank conflicts

The code uses NVIDIA CuTe library for clean tensor operations.

## What's Next

- [ ] Support arbitrarily sequence lengths
- [ ] Implement causal masking
- [ ] Support variable sequence lengths
- [ ] Support KV cache

## References

Based on the Flash Attention v2 paper by Tri Dao (2023). Uses NVIDIA Cutlass/CuTe libraries.

- [Flash Attention v2 Paper](https://arxiv.org/abs/2307.08691)
- [Official Implementation](https://github.com/Dao-AILab/flash-attention)
