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

Tested on NVIDIA GPU with sequence length 4096, head dimension 128:

```
Mini Flash Attention Profiling Results:
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                   mini_flash_attn::_flash_attn_forward         3.46%     108.825us        99.94%       3.144ms       3.144ms     654.799us       100.00%       3.274ms       3.274ms             1  
                                       cudaLaunchKernel         0.92%      28.948us        41.64%       1.310ms       1.310ms       0.000us         0.00%       1.964ms       1.964ms             1  
                       Runtime Triggered Module Loading        40.23%       1.265ms        40.23%       1.265ms     632.710us       1.310ms       200.00%       1.310ms     654.799us             2  
                                Activity Buffer Request        33.24%       1.046ms        33.24%       1.046ms       1.046ms     654.799us       100.00%     654.799us     654.799us             1  
                                  Lazy Function Loading         0.50%      15.577us         0.50%      15.577us      15.577us     654.799us       100.00%     654.799us     654.799us             1  
void mfa::flash_attention_fwd_kernel<mfa::ForwardKer...         0.00%       0.000us         0.00%       0.000us       0.000us     654.799us       100.00%     654.799us     654.799us             1  
                                       aten::empty_like         0.29%       9.278us         0.81%      25.564us      25.564us       0.000us         0.00%       0.000us       0.000us             1  
                                    aten::empty_strided         0.52%      16.286us         0.52%      16.286us      16.286us       0.000us         0.00%       0.000us       0.000us             1  
                                  cudaStreamSynchronize        20.78%     653.650us        20.78%     653.650us     653.650us       0.000us         0.00%       0.000us       0.000us             1  
                                  cudaDeviceSynchronize         0.06%       1.963us         0.06%       1.963us       1.963us       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 3.146ms
Self CUDA time total: 654.799us

Flash Attention Profiling Results:
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                          FlashAttnFunc         1.65%      68.555us        82.29%       3.414ms       3.414ms       0.000us         0.00%       3.310ms       3.310ms             1  
                        flash_attn::_flash_attn_forward         7.31%     303.449us        80.40%       3.336ms       3.336ms     827.467us       100.00%       3.310ms       3.310ms             1  
                                   cudaFuncSetAttribute         4.98%     206.744us        50.03%       2.076ms       2.076ms       0.000us         0.00%       1.655ms       1.655ms             1  
                                Activity Buffer Request        22.29%     925.022us        22.29%     925.022us     925.022us     827.467us       100.00%     827.467us     827.467us             1  
                       Runtime Triggered Module Loading        42.44%       1.761ms        42.44%       1.761ms       1.761ms     827.467us       100.00%     827.467us     827.467us             1  
                                  Lazy Function Loading         2.60%     108.038us         2.60%     108.038us     108.038us     827.467us       100.00%     827.467us     827.467us             1  
void flash::flash_fwd_kernel<Flash_fwd_kernel_traits...         0.00%       0.000us         0.00%       0.000us       0.000us     827.467us       100.00%     827.467us     827.467us             1  
                                 cudaDeviceGetAttribute         0.01%       0.598us         0.01%       0.598us       0.120us       0.000us         0.00%       0.000us       0.000us             5  
                                       aten::empty_like         0.09%       3.741us         0.32%      13.271us      13.271us       0.000us         0.00%       0.000us       0.000us             1  
                                    aten::empty_strided         0.23%       9.530us         0.23%       9.530us       9.530us       0.000us         0.00%       0.000us       0.000us             1  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 4.149ms
Self CUDA time total: 827.467us
```

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

## Accuracy Notes

The implementation achieves ~1e-3 maximum absolute error compared to the reference PyTorch implementation when using FP16. This is within the expected numerical precision for half-precision floating-point operations:

- FP16 mantissa precision: ~3 decimal digits (ε ≈ 0.001)
- Error sources: fp16 conversion of softmax weights, cumulative rounding in long sequences
- Mitigation: Uses FMA instructions and careful online softmax rescaling

For applications requiring higher precision, consider using the official Flash Attention implementation or adjusting the kernel to use higher intermediate precision.

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

- [ ] Implement causal masking
- [ ] Add backward pass
- [ ] Support variable sequence lengths
- [ ] Add more kernel configurations for different head dimensions

## References

- [Flash Attention v2 Paper](https://arxiv.org/abs/2307.08691) (Dao, 2023)
- [Official Flash Attention Implementation](https://github.com/Dao-AILab/flash-attention)

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [NVIDIA Cutlass](https://github.com/NVIDIA/cutlass) and CuTe
- Inspired by the original [Flash Attention](https://github.com/Dao-AILab/flash-attention) implementation
