#pragma once

#include <cuda_runtime.h>
namespace mfa {

template<typename T, int THREADS>
__device__ __forceinline__ T warp_reduce_sum(T x) {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4 || THREADS == 2);

    for (int offset = THREADS / 2; offset > 0; offset /= 2) {
        x += __shfl_xor_sync(0xffffffff, x, offset);
    }
    return x;
};

template<typename T, int THREADS>
__device__ __forceinline__ T warp_reduce_max(T x) {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4 || THREADS == 2);

    for (int offset = THREADS / 2; offset > 0; offset /= 2) {
        x = max(x, __shfl_xor_sync(0xffffffff, x, offset));
    }
    return x;
}

// Block-level reduction using shared memory for inter-warp communication
template<typename T, typename Op>
__device__ __forceinline__ T block_reduce_sum(T* smem, T val, Op op) {
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = blockDim.x / 32;
    
    // Step 1: Reduce within each warp
    val = warp_reduce_sum<T, 32>(val);
    
    // Step 2: First thread of each warp writes to shared memory
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    
    // Step 3: First warp reduces the warp results
    if (warp_id == 0) {
        val = (lane < num_warps) ? smem[lane] : T(0);
        val = warp_reduce_sum<T, 32>(val);
    }
    
    // Step 4: Broadcast result to all threads
    if (threadIdx.x == 0) {
        smem[0] = val;
    }
    __syncthreads();
    
    return smem[0];
}

template<typename T, typename Op>
__device__ __forceinline__ T block_reduce_max(T* smem, T val, Op op) {
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = blockDim.x / 32;
    
    // Step 1: Reduce within each warp
    val = warp_reduce_max<T, 32>(val);
    
    // Step 2: First thread of each warp writes to shared memory
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    
    // Step 3: First warp reduces the warp results
    if (warp_id == 0) {
        val = (lane < num_warps) ? smem[lane] : -INFINITY;
        val = warp_reduce_max<T, 32>(val);
    }
    
    // Step 4: Broadcast result to all threads
    if (threadIdx.x == 0) {
        smem[0] = val;
    }
    __syncthreads();
    
    return smem[0];
}

} // namespace mfa
