# Mini Flash Attention

A minimal, educational implementation of Flash Attention v2 in CUDA. This project demonstrates the core concepts of Flash Attention with a focus on code clarity and understanding.

## Features

- ✅ **Flash Attention v2 Algorithm**: Implements the tiling and online softmax algorithm
- ✅ **Tensor Core Acceleration**: Utilizes CUDA Tensor Cores for fast matrix multiplications
- ✅ **FP16/BF16 Support**: Supports both half precision data types
- ✅ **Memory Efficient**: Uses shared memory tiling to reduce HBM bandwidth
- ✅ **PyTorch Integration**: Seamless integration with PyTorch tensors

## Architecture

The implementation follows the Flash Attention v2 algorithm with these key components:

### Core Components

- **Query/Key/Value Tiles**: Efficient global memory to shared memory copying with `cp.async`
- **Score Computation**: Q@K^T matrix multiplication using Tensor Core MMA instructions
- **Online Softmax**: Numerically stable softmax with rescaling for incremental computation
- **Output Accumulation**: P@V computation with FMA instructions for precision

## Installation

### Prerequisites

- CUDA Toolkit 11.8+ (tested with CUDA 12.8)
- Python 3.8+
- PyTorch 2.0+
- GCC/G++ with C++20 support
- NVIDIA GPU with Compute Capability 8.0+ (Ampere or newer)

### Build from Source

```bash
git clone https://github.com/w4096/mini-flash-attention.git
cd mini-flash-attention
git submodule update --init --recursive

# Install Python package
python setup.py install
```

## Usage

```python
import torch
from mini_flash_attention import mini_flash_attn_func

# Create random input tensors
batch_size = 1
seqlen = 4096
heads = 8
head_dim = 128

q = torch.randn(batch_size, seqlen, heads, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn(batch_size, seqlen, heads, head_dim, device='cuda', dtype=torch.float16)
v = torch.randn(batch_size, seqlen, heads, head_dim, device='cuda', dtype=torch.float16)

# Compute attention
output = mini_flash_attn_func(q, k, v)[0]
```

## Performance

Benchmarked on NVIDIA GPU with the following configuration:

| Parameter | Value |
|-----------|-------|
| Batch Size | 10 |
| Sequence Length | 4096 |
| Head Dimension | 128 |
| Number of Heads | 28 |

by running the benchmark script in `benchmark/run.py`.

### Timing Results

| Implementation | CPU Time | CUDA Time |
|----------------|----------|-----------|
| **Mini Flash Attention** | 41.606ms | **41.547ms** |
| PyTorch Attention | 44.098ms | 42.882ms |
| Flash Attention (official) | 43.890ms | 42.692ms |

### Accuracy Comparison

All implementations produce nearly identical results:

```
Max difference between mini-flash-attn and flash-attn: 3.814697265625e-06
Max difference between torch and flash-attn: 3.814697265625e-06
Max difference between torch and mini-flash-attn: 3.814697265625e-06
```

**Key Takeaways:**
- Mini Flash Attention achieves **competitive performance** with the official implementation
- Numerical accuracy is within expected FP16 precision bounds
- ~6% speedup compared to standard PyTorch attention

## Technical Details

### Algorithm Overview

1. **Tiling**: Split Q into blocks of size `(BlockM, HeadDim)` and K,V into blocks of size `(BlockN, HeadDim)`
2. **Score Computation**: For each Q tile, compute attention scores with all K tiles
3. **Online Softmax**: Update running maximum and exponential sum incrementally
4. **Output Accumulation**: Accumulate P@V products with proper rescaling

### Key Optimizations

- **Swizzled Shared Memory**: Bank conflict reduction using XOR-based swizzling
- **Async Copy**: Non-blocking global-to-shared memory transfers with `cp.async`
- **Fused Operations**: FMA instructions (`__fmaf_rn`) for reduced rounding errors
- **Vectorized Loads/Stores**: 128-bit memory transactions

### CUDA Kernel Configuration

- **Block Size**: 128 threads (4 warps)
- **Tile Dimensions**: BlockM=64, BlockN=64 (configurable)
- **Shared Memory**: ~24KB per block
- **MMA Shape**: 16x8x16 (M x N x K)

## Project Structure

```
mini-flash-attention/
├── csrc/
│   ├── api.cpp                 # Python binding entry point
│   └── mfa/
│       ├── api.cpp             # Forward pass API
│       ├── api.h
│       ├── flash.cu            # Kernel launcher
│       ├── flash.h             # Forward parameters
│       ├── fwd.cuh             # Main kernel implementation
│       ├── traits.h            # Kernel configuration
│       └── utils.cuh           # Utility functions
├── mini_flash_attention/
│   ├── __init__.py
│   └── interface.py            # PyTorch custom op interface
├── tests/
│   └── test.py                 # Accuracy and performance tests
├── 3rd/
│   └── cutlass/                # Cutlass library (submodule)
├── setup.py
└── README.md
```

## Limitations

- **Causal Masking**: Not currently implemented
- **Dropout**: Not supported
- **Compute Capability**: Requires SM 8.0+ (Ampere architecture or newer)
- **GQA/MQA**: Grouped-query attention support is basic

## Development

### Running Tests

```bash
python tests/test.py
```

### Building with Debug Info

```bash
python setup.py build --debug
```

## Contributing

Contributions are welcome! Areas for improvement:

- [x] Implement causal masking
- [ ] Support variable sequence lengths
- [ ] Support KV cache
- [ ] Add more kernel configurations for different head dimensions

## References

- [Flash Attention v2 Paper](https://arxiv.org/abs/2307.08691) (Dao, 2023)
- [Official Flash Attention Implementation](https://github.com/Dao-AILab/flash-attention)

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [NVIDIA Cutlass](https://github.com/NVIDIA/cutlass) and CuTe
- Inspired by the original [Flash Attention](https://github.com/Dao-AILab/flash-attention) implementation
