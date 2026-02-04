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

} // namespace mfa
